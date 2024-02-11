import gc
import itertools
import multiprocessing as mp
import os
import pickle
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator
from gymnasium.vector import SyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import envs
import wandb
from src.action_mapper import ActionMapper
from src.gridworld.common.data import SequenceDataset
from src.gridworld.common.generate import generate_dataset, solve_env
from src.gridworld.common.prepare_envs import make_envs
from src.gridworld.common.transformer import Model
from src.tiny_llama.config import Config as ModelConfig
from src.utils.misc import Timeit, set_seed
from src.utils.schedule import cosine_annealing_with_warmup
from src.utils.wandb_logging import log_in_context, log_list, log_raw_regrets

if False:
    envs


@dataclass
class Config:
    # wandb params
    project: str = "Headless-AD"
    group: str = "gridworld-headless_ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "GridWorld"
    num_train_envs: int = 10_000
    learning_histories_path: Optional[str] = "trajectories"

    num_eval_envs: int = 100
    eval_every: int = 1_000
    log_every: int = 100

    num_train_steps: int = 30_000

    # Model Params
    seq_len: int = 100
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 64
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Training params
    batch_size: int = 64
    learning_rate: float = 3e-3
    beta1: float = 0.9
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    tau: float = 2.0
    sim_measure: str = "dot"
    get_action_type: str = "sample"
    use_action_set_prompt: bool = True
    rand_select_in_ucb: bool = True
    loss_type: str = "contrastive"
    rand_emb_type: str = "orthogonal"

    # New
    rotary_percentage: float = 1.0  # is default in the llama's configs at the bottom
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"

    # Device
    device: str = "cuda"
    autocast_dtype: str = "bf16"

    # Where to save data for experiment visualizations
    logs_dir: str = "logs"

    # New
    action_seq_len: int = 3
    train_frac_acts: float = 0.4
    train_frac_goals: float = 0.85
    grid_size: int = 9
    num_episodes: int = 200
    q_learning_lr: float = 0.9933
    q_learning_discount: float = 0.6238
    check_cached: bool = False

    action_space_type: str = "for_headless"
    num_in_context_episodes: Optional[int] = None

    def __post_init__(self):
        self.job_type = self.action_space_type

        if self.num_in_context_episodes is None:
            self.num_in_context_episodes = 2 * self.num_episodes

        self.eval_seed = 1000 + self.train_seed


def make_env(env_name: str, acts: np.ndarray, goal: np.ndarray, act_seq_len: int):
    """
    This function creates an darkroom environment parameterized by the action sequences
    the length of action sequnce and a goal position.
    """

    def helper():
        return gym.make(
            env_name,
            goal_pos=goal.copy(),
            available_actions=acts.copy(),
            action_seq_len=act_seq_len,
        )

    return helper


@torch.no_grad()
def evaluate_in_context(
    config: Config,
    model: Model,
    acts: np.ndarray,
    goals: np.ndarray,
):
    model.eval()
    # Create env
    vec_env = SyncVectorEnv(
        [
            make_env(
                env_name=config.env_name,
                acts=a,
                goal=g,
                act_seq_len=config.action_seq_len,
            )
            for (a, g) in zip(acts, goals)
        ]
    )
    init_state, _ = vec_env.reset(seed=config.eval_seed)

    # reassign some variables
    num_envs = vec_env.num_envs

    # Create action embeddings
    act_mapper = ActionMapper(
        action_embed_dim=config.token_embed_dim,
        num_actions=acts.shape[1],
        device=config.device,
        sim_measure=config.sim_measure,
        rand_emb_type=config.rand_emb_type,
    )
    act_mapper.regenerate(seed=config.eval_seed)

    # Create context windows for storing the interaction histories
    states = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    states[-1, :] = torch.from_numpy(init_state).to(config.device).type(torch.long)
    actions = torch.zeros(
        (config.seq_len, num_envs, config.token_embed_dim),
        dtype=torch.float32,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    num_actions_per_env = torch.full(
        size=(num_envs,), fill_value=acts.shape[1], device=config.device
    )

    actions_list = act_mapper._get_action_map_as_context(
        num_actions_per_env=num_actions_per_env
    )

    tried_action_inds = []
    # create a list for accumulation of regrets
    all_returns = [[] for _ in range(num_envs)]
    all_lengths = [[] for _ in range(num_envs)]
    current_lengths = np.zeros(num_envs)
    current_returns = np.zeros(num_envs)
    entropies = []

    num_dones = np.zeros(num_envs, dtype=np.int32)
    tried_action_sets = [list() for _ in range(num_envs)]
    interm_tried_action_sets = [set() for _ in range(num_envs)]
    for istep in tqdm(itertools.count(start=1), desc="Eval ..."):
        inp = (
            states.T[:, -istep:],
            actions.transpose(0, 1)[:, -istep:],
            rewards.T[:, -istep:],
        )
        # check for validity
        assert (istep < config.seq_len and inp[0].shape[1] == istep) or (
            istep >= config.seq_len and inp[0].shape[1] == config.seq_len
        ), (
            inp[0].shape[1],
            istep,
        )
        # make prediction
        pred = model(*inp, actions_list=actions_list)
        pred = pred[:, -1]
        # map predictions to action indices
        action_sample, action_mode, entropy = act_mapper.get_action(
            pred.unsqueeze(1),
            num_actions_per_env=num_actions_per_env,
            with_entropy=True,
        )

        if config.get_action_type == "sample":
            action = action_sample
        elif config.get_action_type == "mode":
            action = action_mode
        else:
            raise NotImplementedError

        action = action.squeeze(-1)
        tried_action_inds.append(action.cpu().numpy())
        entropies.append(entropy.mean().item())

        # Env step
        state, reward, term, trunc, _ = vec_env.step(action.cpu().numpy())
        assert len(reward) == num_envs, (len(reward), num_envs)
        current_returns += reward
        current_lengths += 1
        num_dones += (term | trunc).astype(np.int32)

        for j in range(num_envs):
            interm_tried_action_sets[j].add(action[j].item())

        for j in np.where(term | trunc)[0]:
            if num_dones[j] <= config.num_in_context_episodes:
                all_returns[j].append(current_returns[j])
                all_lengths[j].append(current_lengths[j])
                if len(tried_action_sets[j]) == 0:
                    tried_action_sets[j].append(interm_tried_action_sets[j])
                else:
                    tried_action_sets[j].append(
                        tried_action_sets[j][-1].union(interm_tried_action_sets[j])
                    )
                interm_tried_action_sets[j] = set()

                current_returns[j] = 0
                current_lengths[j] = 0

        # record the actions and rewards seen
        states = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # record the _embeddings_ of the chosen actions
        states[-1] = torch.from_numpy(state).type(torch.long).to(config.device)
        actions[-2] = act_mapper(action)
        rewards[-2] = torch.from_numpy(reward).type(torch.long).to(config.device)

        if np.min(num_dones) >= config.num_in_context_episodes:
            break

    tried_action_inds = np.vstack(tried_action_inds).T

    num_unique_in_each_batch = [
        len(np.unique(batch_element, axis=0)) for batch_element in tried_action_inds
    ]
    num_unique_in_each_batch = np.mean(num_unique_in_each_batch)

    model.train()

    returns = np.vstack(all_returns)
    lengths = np.vstack(all_lengths)
    entropies = np.array(entropies)[None, ...]

    return returns, lengths, num_unique_in_each_batch, entropies, tried_action_sets


def wandb_define_metrics() -> None:
    wandb.define_metric("data_gen/step")
    wandb.define_metric("data_gen/*", step_metric="data_gen/step")

    wandb.define_metric("final/step")
    wandb.define_metric("final/*", step_metric="final/step")

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("times/step")
    wandb.define_metric("times/*", step_metric="times/step")

    wandb.define_metric("num_uniques/step")
    wandb.define_metric("num_uniques/*", step_metric="num_uniques/step")
    wandb.define_metric("returns/step")
    wandb.define_metric("returns/*", step_metric="returns/step")

    wandb.define_metric("tried/step")
    wandb.define_metric("tried/*", step_metric="tried/step")


def loss_fn(
    pred_embs: torch.Tensor,
    action_embs: torch.Tensor,
    actions: torch.Tensor,
    tau: float,
    sim_measure: str,
):
    # pred_embs: b x t x d
    # action_embs: num_acts x d
    # actions: b x t
    # num_actions: b
    b, t, d1 = pred_embs.shape
    num_acts, d2 = action_embs.shape
    assert d1 == d2, (d1, d2)

    # exp_cosine_sim: b x t x num_acts
    # as the result, we have a similarity score between each prediction and each action embedding
    pred_norm = torch.norm(pred_embs, p=2, dim=-1)

    if sim_measure == "cosine":
        pred_embs = pred_embs / pred_norm.unsqueeze(-1)
    elif sim_measure == "dot":
        pass
    else:
        raise NotImplementedError

    sim = pred_embs @ action_embs.T / tau

    sim = sim - sim.max(-1).values.unsqueeze(-1)
    assert sim.shape == (b, t, num_acts), (
        sim.shape,
        (b, t, num_acts),
    )

    # The similarity between prediction and ground truth action should be promoted
    self_dist = torch.gather(sim, 2, actions.unsqueeze(-1)).squeeze(-1)
    # All other actions will be negatives
    neg_dist = torch.exp(sim).sum(-1)

    # loss = -(self_dist / neg_dist).log().mean()
    loss = -(self_dist - neg_dist.log()).mean()

    return loss, pred_norm.detach().mean()


def next_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def regrets_to_wandb(config: Config, regrets: Dict[str, np.ndarray], name: str):
    filename = os.path.join(config.logs_dir, f"{name}.pickle")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "+wb") as f:
        pickle.dump(regrets, f)

    wandb.save(filename)


def base_algo_scores(config: Config, envs: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    base_algo_final_returns = {}
    with mp.Pool(processes=os.cpu_count()) as pool:
        for key, (acts, goals) in envs.items():
            returns, _ = solve_env(
                config=config, pool=pool, goals=goals, actions=acts, savedir=None
            )
            final_return = returns.mean(0)[-1]

            base_algo_final_returns[key] = final_return

    return base_algo_final_returns


def random_model_scores(
    config: Config,
    model_config: ModelConfig,
    accelerator: Accelerator,
    envs: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    real_num_in_context_episodes = config.num_in_context_episodes
    config.num_in_context_episodes = 10
    random_model = Model(
        config=model_config,
        n_token=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
        action_seq_len=config.action_seq_len,
        num_states=config.grid_size**2,
    ).to(config.device)

    random_model = accelerator.prepare(random_model)

    returns = {}

    for key, (acts, goals) in tqdm(envs.items()):
        (returns[key], _, _, _, _) = evaluate_in_context(
            config,
            model=random_model,
            acts=acts,
            goals=goals,
        )

        returns[key] = returns[key].mean(0)[-1]

    config.num_in_context_episodes = real_num_in_context_episodes
    return returns


def get_all_test_metric(
    tried_action_sets: List[Set[int]], train_size: int, test_size: int
):
    train_tried = np.zeros(
        (len(tried_action_sets), len(tried_action_sets[0])), dtype=np.float32
    )
    test_tried = np.zeros(
        (len(tried_action_sets), len(tried_action_sets[0])), dtype=np.float32
    )

    for env in range(len(tried_action_sets)):
        for t in range(len(tried_action_sets[env])):
            actions_so_far = tried_action_sets[env][t]
            unique_acts = np.array(list(actions_so_far))
            train_acts = unique_acts[unique_acts < train_size]
            test_acts = unique_acts[unique_acts > train_size]

            train_tried[env][t] = len(train_acts) / float(train_size)
            test_tried[env][t] = len(test_acts) / float(test_size)

    return train_tried, test_tried


@pyrallis.wrap()
def train(config: Config):
    # Clean up and then create directory
    if os.path.exists(config.logs_dir):
        shutil.rmtree(config.logs_dir)
    os.makedirs(config.logs_dir)

    # config.device = "cuda" if torch.cuda.is_available() else "cpu"

    config.autocast_dtype = (
        "bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )
    accelerator = Accelerator(mixed_precision=config.autocast_dtype)
    config.device = accelerator.device

    wandb.init(
        project=config.project,
        group=config.group,
        job_type=config.job_type,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    # if (
    #     config.token_embed_dim < config.num_heads
    #     or config.token_embed_dim // config.num_heads < 8
    #     or config.token_embed_dim // config.num_heads > 256
    #     or not config.parallel_residual
    #     and config.shared_attention_norm
    # ):
    #     sys.exit(0)

    # if config. == 2048:
    #     config.batch_size = 64
    # elif config.action_embed_dim == 64:
    #     config.batch_size = 512

    wandb_define_metrics()

    (train_acts, train_goals), eval_envs = make_envs(config=config)
    generate_dataset(config=config, actions=train_acts, goals=train_goals)
    q_learning_scores = base_algo_scores(config=config, envs=eval_envs)
    for key, value in q_learning_scores.items():
        wandb.log({f"q_learning/{key}": value})

    set_seed(seed=config.train_seed)

    dataset = SequenceDataset(
        runs_path=config.learning_histories_path, seq_len=config.seq_len
    )
    shape0s = np.array(
        [len(dataset._states), len(dataset._actions), len(dataset._rewards)]
    )
    assert np.all(shape0s == config.num_train_envs), (
        shape0s,
        config.num_train_envs,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        persistent_workers=True,
        drop_last=True,
    )

    # model & optimizer & scheduler setup
    model_config = ModelConfig(
        block_size=3 * config.seq_len + 5**config.action_seq_len,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_embd=config.d_model,
        bias=config.layer_norm_bias,
        rotary_percentage=config.rotary_percentage,
        parallel_residual=config.parallel_residual,
        shared_attention_norm=config.shared_attention_norm,
        _norm_class=config._norm_class,
        _mlp_class=config._mlp_class,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
    )

    random_scores = random_model_scores(
        config=config,
        model_config=model_config,
        accelerator=accelerator,
        envs=eval_envs,
    )
    for key, value in random_scores.items():
        wandb.log({f"random_model/{key}": value})

    model = Model(
        config=model_config,
        n_token=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
        action_seq_len=config.action_seq_len,
        num_states=config.grid_size**2,
    ).to(config.device)

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, 0.999),
    )
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=int(config.num_train_steps * config.warmup_ratio),
        total_steps=config.num_train_steps,
    )

    model, optim, dataloader, scheduler = accelerator.prepare(
        model, optim, dataloader, scheduler
    )
    dataloader = next_dataloader(dataloader)

    act_mapper = ActionMapper(
        action_embed_dim=config.token_embed_dim,
        num_actions=train_acts.shape[1],
        device=config.device,
        sim_measure=config.sim_measure,
        rand_emb_type=config.rand_emb_type,
    )

    best_opt_value = None
    # Start training
    for global_step in trange(1, config.num_train_steps + 2, desc="Training"):
        with Timeit() as batch_timer:
            states, actions, rewards = next(dataloader)
            # Prepare input
            states = states.to(torch.long)
            actions = actions.to(torch.long)
            rewards = rewards.to(torch.long)
            num_actions_per_env = torch.full(
                size=(config.batch_size,),
                fill_value=train_acts.shape[1],
                device=config.device,
            )
            assert actions.shape[1] == config.seq_len, (
                actions.shape[1],
                config.seq_len,
            )
            assert states.shape[1] == config.seq_len, (
                states.shape[1],
                config.seq_len,
            )
            assert rewards.shape[1] == config.seq_len, (
                rewards.shape[1],
                config.seq_len,
            )

        with Timeit() as act_mapper_timer:
            # Map actions to their embeddings
            act_mapper.regenerate()
            action_embeds = act_mapper(actions)

            actions_list = act_mapper._get_action_map_as_context(
                num_actions_per_env=num_actions_per_env
            )
        with Timeit() as pred_timer:
            # Make prediction
            # [batch_size, seq_len, action_embed_dim] and [batch_size, seq_len, 1]
            # to [batch_size, seq_len, action_embed_dim]
            pred = model(
                states=states,
                act_emb=action_embeds,
                rewards=rewards,
                actions_list=actions_list,
            )
            assert pred.shape[1] == config.seq_len, (pred.shape[1], config.seq_len)

        # Calculate loss
        # Compare predictions at timestep t against the ground truth action at timestep 't + 1'
        with Timeit() as loss_timer:
            with accelerator.autocast():
                if config.loss_type == "contrastive":
                    loss, norm = loss_fn(
                        pred_embs=pred,
                        action_embs=act_mapper.action_map,
                        actions=actions,
                        tau=config.tau,
                        sim_measure=config.sim_measure,
                    )
                elif config.loss_type == "mse":
                    loss = torch.pow(pred - action_embeds, 2).mean(-1).mean()
                    norm = torch.norm(pred, p=2, dim=-1).detach().mean()
                else:
                    raise NotImplementedError

        with Timeit() as back_timer:
            # Make optimization step
            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optim.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()

        with Timeit() as metrics_timer:
            # Calculate action prediction accuracy
            with torch.no_grad():
                act_sample, act_mode = act_mapper.get_action(
                    pred, num_actions_per_env=num_actions_per_env
                )
                accuracy_sample = torch.mean(
                    (act_sample == actions).type(torch.float32)
                )
                accuracy_mode = torch.mean((act_mode == actions).type(torch.float32))

                mse = torch.nn.functional.mse_loss(input=pred, target=action_embeds)
                cossim = torch.nn.functional.cosine_similarity(
                    x1=pred.flatten(end_dim=1),
                    x2=action_embeds.flatten(end_dim=1),
                ).mean()

        if global_step % config.log_every == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/norm": norm.item(),
                    "train/accuracy_sample": accuracy_sample.item(),
                    "train/accuracy_mode": accuracy_mode.item(),
                    "train/mse": mse.item(),
                    "train/cosine_similarity": cossim.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
            )

            total_time = (
                batch_timer.elapsed_time_cpu
                + act_mapper_timer.elapsed_time_cpu
                + pred_timer.elapsed_time_cpu
                + loss_timer.elapsed_time_cpu
                + back_timer.elapsed_time_cpu
                + metrics_timer.elapsed_time_cpu
            )

            wandb.log(
                {
                    "times/batch_sampling": batch_timer.elapsed_time_cpu,
                    "times/act_mapping": act_mapper_timer.elapsed_time_cpu,
                    "times/pred": pred_timer.elapsed_time_cpu,
                    "times/loss": loss_timer.elapsed_time_cpu,
                    "times/backward": back_timer.elapsed_time_cpu,
                    "times/metrics": metrics_timer.elapsed_time_cpu,
                    "times/step": global_step,
                    "times/throughput": config.batch_size / total_time,
                }
            )

        if global_step % config.eval_every == 0:
            torch.cuda.empty_cache()
            gc.collect()
            eval_step = global_step // config.eval_every
            # max envs is used to save memory usage
            returns = {}
            lengths = {}
            num_uniques = {}
            entropies = {}
            tried_action_sets = {}

            for key, (acts, goals) in eval_envs.items():
                (
                    returns[key],
                    lengths[key],
                    num_uniques[key],
                    entropies[key],
                    tried_action_sets[key],
                ) = evaluate_in_context(
                    config,
                    model=model,
                    acts=acts,
                    goals=goals,
                )

            log_in_context(values_dict=returns, name="in-context/returns")
            log_in_context(values_dict=lengths, name="in-context/lengths")
            log_in_context(values_dict=entropies, name="in-context/entropy")

            log_raw_regrets(regrets=returns, name="returns", step=eval_step)
            regrets_to_wandb(config, returns, f"ours_{eval_step}")

            # Log the average number of unique actions in all batches
            # during first 'seq_len' steps of evaluation
            log_list(values_dict=num_uniques, name="num_uniques", step=eval_step)

            # take the 'all_test' setting and measure how much of the train and test sets are used
            # during in-context work
            if "all_test" in tried_action_sets:
                train_tried, test_tried = get_all_test_metric(
                    tried_action_sets=tried_action_sets["all_test"],
                    train_size=len(train_acts[0]),
                    test_size=len(eval_envs["test_train"][0][0]),
                )
                log_in_context(
                    values_dict={"train_tried": train_tried},
                    name="in-context/train_tried",
                )
                log_in_context(
                    values_dict={"test_tried": test_tried}, name="in-context/test_tried"
                )

                regrets_to_wandb(
                    config, {"train_tried": train_tried}, f"train_tried_{eval_step}"
                )
                regrets_to_wandb(
                    config, {"test_tried": test_tried}, f"test_tried_{eval_step}"
                )

                wandb.log(
                    {
                        "tried/train": train_tried.mean(0)[-1],
                        "tried/test": test_tried.mean(0)[-1],
                        "tried/step": eval_step,
                    }
                )

            # more actions is more important
            # and normalized regret should be close to 1
            opt_value = 0
            for key in returns:
                opt_value += returns[key].mean(0)[-1]

            opt_value /= len(returns)

            if best_opt_value is None:
                best_opt_value = opt_value
            else:
                best_opt_value = max(best_opt_value, opt_value)

            wandb.log(
                {
                    "final/value": best_opt_value,
                    "final/step": eval_step,
                }
            )

    wandb.finish()


if __name__ == "__main__":
    train()
