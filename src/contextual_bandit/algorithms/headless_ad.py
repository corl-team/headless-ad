import gc
import os
import shutil
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator
from gymnasium.vector import SyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import trange

import envs
import wandb
from src.action_mapper import ActionMapper
from src.contextual_bandit.common.data import SequenceDataset
from src.contextual_bandit.common.generate import generate_dataset, solve_bandits
from src.contextual_bandit.common.prepare_envs import make_envs
from src.contextual_bandit.common.transformer import Model
from src.tiny_llama.config import Config as ModelConfig
from src.utils.misc import Timeit, index_mask, set_seed
from src.utils.schedule import cosine_annealing_with_warmup
from src.utils.wandb_logging import (
    arrays_to_wandb,
    log_in_context,
    log_list,
    log_normalized_regrets,
    log_raw_regrets,
)

if False:
    envs


@dataclass
class Config:
    # wandb params
    project: str = "Headless-AD"
    group: str = "contextual_bandit-headless_ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "ContextualBandit"
    num_train_envs: int = 10_000
    num_env_steps: int = 300
    num_in_context_steps: int = 300
    train_min_arms: int = 4
    train_max_arms: int = 20
    eval_more_arms_list: List[int] = field(
        default_factory=lambda: [20, 25, 30, 40, 50, 100]
    )
    data_generation_algo: str = "linucb"
    ucb_alpha: float = 0.3  # ucb
    learning_histories_path: Optional[str] = "trajectories"
    bandit_context_dim: int = 2

    num_eval_envs: int = 200
    eval_every: int = 1000
    log_every: int = 100

    num_train_steps: int = 100_000

    use_aux_loss: bool = False

    # Model Params
    seq_len: int = 300
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Training params
    batch_size: int = 32
    learning_rate: float = 3e-3
    beta1: float = 0.9
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    tau: float = 2.0
    sim_measure: str = "dot"
    get_action_type: str = "mode"
    use_action_set_prompt: bool = True
    train_on_mixed: bool = True
    rand_select_in_ucb: bool = True
    loss_type: str = "contrastive"  # ["contrastive", "mse"]
    rand_emb_type: str = "orthogonal"

    # New
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"
    shared_attention_norm: bool = False

    # Device
    device: str = "cuda"
    autocast_dtype: str = "bf16"

    # Where to save data for experiment visualizations
    logs_dir: str = "logs"

    def __post_init__(self):
        self.job_type = f"arms_{self.train_max_arms}"

        self.eval_seed = 1000 + self.train_seed


def make_env(env_name, context_dim: int, arm_embeds: np.ndarray, num_arms: int):
    """
    This function creates an bandit environment parameterized by the number of arms
    and the mean rewards assigned to each arm.
    """

    def helper():
        return gym.make(
            env_name,
            context_dim=context_dim,
            arm_embeds=arm_embeds.copy(),
            num_arms=num_arms,
        )

    return helper


@torch.no_grad()
def evaluate_in_context(
    config: Config,
    model: Model,
    arm_embeds: np.ndarray,
    num_actions_per_env: np.ndarray,
):
    model.eval()
    # Create env
    vec_env = SyncVectorEnv(
        [
            make_env(
                env_name=config.env_name,
                context_dim=config.bandit_context_dim,
                arm_embeds=ae,
                num_arms=na,
            )
            for (ae, na) in zip(arm_embeds, num_actions_per_env)
        ]
    )
    init_state, _ = vec_env.reset(seed=config.eval_seed)

    # reassign some variables
    num_envs = vec_env.num_envs
    num_actions_per_env = torch.from_numpy(num_actions_per_env).to(config.device)

    # Create action embeddings
    act_mapper = ActionMapper(
        action_embed_dim=config.token_embed_dim,
        num_actions=num_actions_per_env.max().item(),
        device=config.device,
        sim_measure=config.sim_measure,
        rand_emb_type=config.rand_emb_type,
    )
    act_mapper.regenerate(seed=config.eval_seed)

    # Create context windows for storing the interaction histories
    states = torch.zeros(
        (config.seq_len, num_envs, config.bandit_context_dim),
        dtype=torch.float32,
        device=config.device,
    )
    # Insert initial state
    states[-1] = torch.tensor(init_state, dtype=torch.float32, device=config.device)

    actions = torch.zeros(
        (config.seq_len, num_envs, config.token_embed_dim),
        dtype=torch.float32,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.float32, device=config.device
    )
    action_set_prompt = act_mapper._get_action_map_as_context(
        num_actions_per_env=num_actions_per_env
    )

    # create a list for accumulation of regrets
    regrets = []
    entropies = []
    for istep in trange(1, config.num_in_context_steps + 1, desc="Eval ..."):
        sliced_states = states.transpose(0, 1)[:, -istep:]
        sliced_actions = actions.transpose(0, 1)[:, -istep:]
        sliced_rewards = rewards.T[:, -istep:]

        # check for validity
        assert (istep < config.seq_len and sliced_states.shape[1] == istep) or (
            istep >= config.seq_len and sliced_states.shape[1] == config.seq_len
        ), (
            sliced_states.shape[1],
            istep,
        )
        # make prediction
        pred = model(
            states=sliced_states,
            act_emb=sliced_actions,
            rewards=sliced_rewards,
            action_set_prompt=action_set_prompt,
        )
        pred = pred[:, -1]

        # map predictions to action indices
        # there are two options how to obtain the indices -
        # sampling proportional to the similarities between action embeddings and the prediction
        # and taking the action with the closest embedding
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
        # check that all actions sampled are lower than the amount of arms in each
        # corresponding bandit
        assert (action < num_actions_per_env).all(), (
            action[action >= num_actions_per_env],
            num_actions_per_env[action >= num_actions_per_env],
        )
        entropies.append(entropy.mean().item())

        # Env step
        state, reward, _, _, info = vec_env.step(action.cpu().numpy())
        assert len(info["regret"]) == num_envs, (len(info["regret"]), num_envs)
        assert len(reward) == num_envs, (len(reward), num_envs)

        # record the actions and rewards seen
        states = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # record the _embeddings_ of the chosen actions
        states[-1] = torch.tensor(state, dtype=torch.float32, device=config.device)
        actions[-2] = act_mapper(action)
        rewards[-2] = torch.tensor(reward, dtype=torch.float32, device=config.device)

        # record the regret on the current step
        regrets.append(info["regret"])

        # Record the amount of unique actions in each sequence
        if istep == config.seq_len - 1:
            num_unique_in_each_batch = [
                len(torch.unique(batch_element, dim=0))
                for batch_element in actions.transpose(0, 1)
            ]
            num_unique_in_each_batch = np.mean(num_unique_in_each_batch)

    model.train()

    regrets = np.vstack(regrets).T
    entropies = np.vstack(entropies).T

    return regrets, num_unique_in_each_batch, entropies


def wandb_define_metrics() -> None:
    """
    This function makes the wandb page prettier
    """
    wandb.define_metric("data_gen/step")
    wandb.define_metric("data_gen/regret", step_metric="data_gen/step")

    wandb.define_metric("final/step")
    wandb.define_metric("final/*", step_metric="final/step")

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("times/step")
    wandb.define_metric("times/*", step_metric="times/step")

    wandb.define_metric("num_uniques/step")
    wandb.define_metric("num_uniques/*", step_metric="num_uniques/step")
    wandb.define_metric("raw_regret/step")
    wandb.define_metric("raw_regret/*", step_metric="raw_regret/step")
    wandb.define_metric("ucb_regret/step")
    wandb.define_metric("ucb_regret/*", step_metric="ucb_regret/step")
    wandb.define_metric("random_regrets/step")
    wandb.define_metric("random_regret/*", step_metric="random_regret/step")
    wandb.define_metric("normalized_regret/step")
    wandb.define_metric("normalized_regret/*", step_metric="normalized_regret/step")
    wandb.define_metric("normalized_regret_means/step")
    wandb.define_metric(
        "normalized_regret_means/*", step_metric="normalized_regret_means/step"
    )


def contrastive_loss_fn(
    pred_embs: torch.Tensor,
    action_embs: torch.Tensor,
    actions: torch.Tensor,
    num_actions_per_env: torch.Tensor,
    tau: float,
    sim_measure: str,
):
    # pred_embs: b x t x d
    # action_embs: num_acts x d
    # actions: b x t
    # num_actions: b
    b, t, d1 = pred_embs.shape
    num_acts, d2 = action_embs.shape
    (b2,) = num_actions_per_env.shape
    assert d1 == d2, (d1, d2)
    assert b == b2, (b, b2)

    # calculate the norm of each prediction vector
    pred_norm = torch.norm(pred_embs, p=2, dim=-1)

    # calculate the similarities
    if sim_measure == "cosine":
        pred_embs = torch.nn.functional.normalize(pred_embs, p=2, dim=2)
    elif sim_measure == "dot":
        pass
    else:
        raise NotImplementedError

    sim = pred_embs @ action_embs.T / tau

    # Mask will multiply unavailable actions with 0 which will stop gradient flow from those embeds
    mask = index_mask(num_actions=num_actions_per_env, num_total_actions=num_acts)
    sim = sim * mask.unsqueeze(1)
    sim = sim - sim.max(-1).values.unsqueeze(-1)
    assert sim.shape == (b, t, num_acts), (
        sim.shape,
        (b, t, num_acts),
    )

    # The similarity between prediction and ground truth action should be promoted
    self_dist = torch.gather(sim, 2, actions.unsqueeze(-1)).squeeze(-1)
    # All other actions will be negatives
    neg_dist = torch.exp(sim).sum(-1)

    loss = -(self_dist - neg_dist.log()).mean()

    return loss, pred_norm.detach().mean()


def mse_loss_fn(
    pred_embs: torch.Tensor,
    action_embs: torch.Tensor,
):
    loss = ((pred_embs - action_embs) ** 2).sum(2).mean()

    with torch.no_grad():
        pred_norm = torch.norm(pred_embs, p=2, dim=-1)

    return loss, pred_norm.detach().mean()


def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch


def make_lower_and_upper_regrets(
    config: Config, dists: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    This function calculates the performance of contextual bandit-specific algorithms
    for a later comparison with AD.
    """
    ucb_regrets = {}
    random_regrets = {}

    ucb_actions = {}
    random_actions = {}

    for key, (arm_embeds, num_arms) in dists.items():
        # Disabling saving of the histories
        ucb_regrets[key], _, ucb_actions[key], _, _ = solve_bandits(
            config=config,
            arm_embeds=arm_embeds,
            num_arms=num_arms,
            basedir=None,
            data_generation_algo="linucb",
            num_env_steps=config.num_env_steps,
            seed=config.eval_seed,
        )
        random_regrets[key], _, random_actions[key], _, _ = solve_bandits(
            config=config,
            arm_embeds=arm_embeds,
            num_arms=num_arms,
            basedir=None,
            data_generation_algo="random",
            num_env_steps=config.num_env_steps,
            seed=config.eval_seed,
        )

    return random_regrets, ucb_regrets, random_actions, ucb_actions


@pyrallis.wrap()
def train(config: Config):
    # Clean up and then create directory
    if os.path.exists(config.logs_dir):
        shutil.rmtree(config.logs_dir)
    os.makedirs(config.logs_dir)

    config.autocast_dtype = (
        "bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )
    accelerator = Accelerator(mixed_precision=config.autocast_dtype)
    config.device = accelerator.device

    set_seed(seed=config.train_seed)

    wandb.init(
        project=config.project,
        group=config.group,
        job_type=config.job_type,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    wandb_define_metrics()

    # create the contextual bandits for train and test
    (train_arm_embeds, train_num_arms), eval_dists = make_envs(config=config)

    # Run the data generation algorithm and log the training histories
    generate_dataset(
        config=config,
        arm_embeds=train_arm_embeds,
        num_arms=train_num_arms,
        seed=config.train_seed,
    )

    # Evaluate some contextual bandit-specific algorithms and log their regrets.
    # That will later be used for comparison with AD
    (
        random_regrets,
        ucb_regrets,
        random_actions,
        ucb_actions,
    ) = make_lower_and_upper_regrets(config=config, dists=eval_dists)
    arrays_to_wandb(logs_dir=config.logs_dir, arrays=random_regrets, name="random")
    arrays_to_wandb(logs_dir=config.logs_dir, arrays=ucb_regrets, name="linucb")

    log_raw_regrets(regrets=ucb_regrets, name="ucb_regrets", step=0)
    log_raw_regrets(regrets=random_regrets, name="random_regrets", step=0)

    for key, value in random_actions.items():
        num_uniques = len(np.unique(value))
        wandb.log({f"random_actions/{key}": num_uniques})

    for key, value in ucb_actions.items():
        num_uniques = len(np.unique(value))
        wandb.log({f"ucb_actions/{key}": num_uniques})

    dataset = SequenceDataset(
        runs_path=config.learning_histories_path, seq_len=config.seq_len
    )
    shape0s = np.array(
        [dataset._states.shape[0], dataset._actions.shape[0], dataset._rewards.shape[0]]
    )
    assert np.all(shape0s == config.num_train_envs), (
        shape0s,
        config.num_train_envs,
    )
    shape1s = np.array(
        [dataset._states.shape[1], dataset._actions.shape[1], dataset._rewards.shape[1]]
    )
    assert np.all(shape1s == config.num_env_steps), (
        shape1s,
        config.num_env_steps,
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
        block_size=3 * config.seq_len + 100,
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

    model = Model(
        config=model_config,
        token_dim=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
        state_dim=config.bandit_context_dim,
    ).to(config.device)
    model.apply(partial(model._init_weights, n_layer=model_config.n_layer))

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

    # wrap everything into an accelerator
    model, optim, dataloader, scheduler = accelerator.prepare(
        model, optim, dataloader, scheduler
    )
    # the dataloader is effectively a consistent stream of random data samples
    # from the dataset
    dataloader = next_dataloader(dataloader)

    act_mapper = ActionMapper(
        action_embed_dim=config.token_embed_dim,
        num_actions=config.train_max_arms,
        device=config.device,
        sim_measure=config.sim_measure,
        rand_emb_type=config.rand_emb_type,
    )

    # Start training
    for global_step in trange(1, config.num_train_steps + 1, desc="Training"):
        with Timeit() as batch_timer:
            # sample a new batch
            states, actions, rewards, num_actions_per_env = next(dataloader)
            # Prepare input
            states = states.to(torch.float32)
            actions = actions.to(torch.long)
            rewards = rewards.to(torch.float32)
            num_actions_per_env = num_actions_per_env.to(torch.long)
            assert actions.shape[1] == config.seq_len, (
                actions.shape[1],
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

            action_set_prompt = act_mapper._get_action_map_as_context(
                num_actions_per_env=num_actions_per_env
            )
        with Timeit() as pred_timer:
            # Make prediction
            pred = model(
                states=states,
                act_emb=action_embeds,
                rewards=rewards,
                action_set_prompt=action_set_prompt,
            )
            assert pred.shape[1] == config.seq_len, (pred.shape[1], config.seq_len)

        # Calculate loss
        # Compare predictions at timestep t against the ground truth action at timestep 't + 1'
        with Timeit() as loss_timer:
            with accelerator.autocast():
                if config.loss_type == "contrastive":
                    loss, norm = contrastive_loss_fn(
                        pred_embs=pred,
                        action_embs=act_mapper.action_map,
                        actions=actions,
                        num_actions_per_env=num_actions_per_env,
                        tau=config.tau,
                        sim_measure=config.sim_measure,
                    )
                elif config.loss_type == "mse":
                    loss, norm = mse_loss_fn(pred_embs=pred, action_embs=action_embeds)
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

        # log the training metrics
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

        # start evaluation
        if global_step % config.eval_every == 0:
            # free the memory
            torch.cuda.empty_cache()
            gc.collect()

            eval_step = global_step // config.eval_every

            regrets = {}
            num_uniques = {}
            entropies = {}

            # evaluate on each evaluation set
            for key, (arm_embeds, num_arms_per_env) in eval_dists.items():
                (
                    regrets[key],
                    num_uniques[key],
                    entropies[key],
                ) = evaluate_in_context(
                    config,
                    model=model,
                    arm_embeds=arm_embeds,
                    num_actions_per_env=num_arms_per_env,
                )

            # create plots with in-context performance and log the to wandb
            log_in_context(values_dict=regrets, name="in-context/regret")
            log_in_context(values_dict=entropies, name="in-context/entropy")

            # log regret curves as arrays to wandb
            arrays_to_wandb(
                logs_dir=config.logs_dir, arrays=regrets, name=f"ours_{eval_step}"
            )

            # Log the average number of unique actions in all batches
            # during first 'seq_len' steps of evaluation
            log_list(values_dict=num_uniques, name="num_uniques", step=eval_step)

            # Choose the algorithm which performance will correspond to 1
            # when the regrets are normalized
            if config.data_generation_algo == "linucb":
                upper_regrets = ucb_regrets
            else:
                raise NotImplementedError

            log_normalized_regrets(
                raw_regrets=regrets,
                lower_regrets=random_regrets,
                upper_regrets=upper_regrets,
                step=eval_step,
            )

            log_raw_regrets(regrets=regrets, name="raw_regrets", step=eval_step)

            # Average regrets from all evaluations sets and calculate the bayes objective
            opt_value = 0
            for key, value in regrets.items():
                opt_value += value.mean(0)[-1]

            opt_value /= len(regrets)

            wandb.log(
                {
                    "final/value": opt_value,
                    "final/step": eval_step,
                }
            )

    wandb.finish()


if __name__ == "__main__":
    train()
