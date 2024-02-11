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
from src.bandit.common.data import SequenceDataset
from src.bandit.common.generate import generate_dataset, solve_bandits
from src.bandit.common.prepare_envs import make_envs
from src.bandit.common.transformer import Model
from src.tiny_llama.config import Config as ModelConfig
from src.utils.misc import Timeit, index_mask, norm_regret, set_seed
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
    group: str = "bernoulli_bandit-headless_ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "MultiArmedBanditBernoulli"
    num_train_envs: int = 10_000
    num_env_steps: int = 300
    num_in_context_steps: int = 300
    train_min_arms: int = 4
    train_max_arms: int = 20
    eval_more_arms_list: List[int] = field(
        default_factory=lambda: [10, 20, 25, 30, 40, 50]
    )
    data_generation_algo: str = "thompson"
    ucb_alpha: float = 0.3  # ucb
    learning_histories_path: Optional[str] = "trajectories"

    num_eval_envs: int = 100
    eval_every: int = 1_000
    log_every: int = 100

    num_train_steps: int = 200_000

    # Model Params
    seq_len: int = 300
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 64
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
    get_action_type: str = "sample"
    use_action_set_prompt: bool = True
    train_on_mixed: bool = True
    rand_select_in_ucb: bool = True
    loss_type: str = "contrastive"
    rand_emb_type: str = "orthogonal"

    # New
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"
    shared_attention_norm: bool = False
    # intermediate_size: int = 102

    # Device
    device: str = "cuda"
    autocast_dtype: str = "bf16"

    # Where to save data for experiment visualizations
    logs_dir: str = "logs"

    def __post_init__(self):
        self.job_type = "job"

        self.eval_seed = 1000 + self.train_seed


def make_env(env_name: str, arms_mean: np.ndarray, num_arms: int):
    """
    This function creates an bandit environment parameterized by the number of arms
    and the mean rewards assigned to each arm.
    """

    def helper():
        return gym.make(env_name, arms_mean=arms_mean.copy(), num_arms=num_arms)

    return helper


def make_input_for_eval(
    actions: torch.Tensor,
    rewards: torch.Tensor,
    istep: int,
):
    """
    The creation of input is put here for compactness. If the evaluation has just started
    then the history is empty. Otherwise, only the filled part of the context window is cut out.
    """

    if istep == 0:
        num_envs = actions.shape[1]
        token_embed_dim = actions.shape[2]
        inp = (
            torch.empty(
                num_envs,
                0,
                token_embed_dim,
                dtype=actions.dtype,
                device=actions.device,
            ),
            torch.empty(num_envs, 0, dtype=rewards.dtype, device=rewards.device),
        )
    else:
        inp = (actions.transpose(0, 1)[:, -istep:], rewards.T[:, -istep:])

    return inp


@torch.no_grad()
def evaluate_in_context(
    config: Config,
    model: Model,
    arms_means: np.ndarray,
    num_actions_per_env: np.ndarray,
):
    model.eval()
    # Create env
    vec_env = SyncVectorEnv(
        [
            make_env(config.env_name, d, na)
            for (d, na) in zip(arms_means, num_actions_per_env)
        ]
    )
    vec_env.reset(seed=config.eval_seed)

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

    # create context windows
    # Insert all actions to the beginning of the context
    # [batch_size, seq_len]
    actions = torch.zeros(
        (config.seq_len, num_envs, config.token_embed_dim),
        dtype=torch.float32,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    action_set_prompt = act_mapper._get_action_map_as_context(
        num_actions_per_env=num_actions_per_env
    )

    # create a list for accumulation of regrets
    regrets = []
    entropies = []
    for istep in trange(config.num_in_context_steps, desc="Eval ..."):
        sliced_actions, sliced_rewards = make_input_for_eval(
            actions=actions,
            rewards=rewards,
            istep=istep,
        )
        # check for validity
        assert (istep < config.seq_len and sliced_actions.shape[1] == istep) or (
            istep >= config.seq_len and sliced_actions.shape[1] == config.seq_len
        ), (
            sliced_actions.shape[1],
            istep,
        )
        # make prediction
        pred = model(
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
        _, reward, _, _, info = vec_env.step(action.cpu().numpy())
        assert len(info["regret"]) == num_envs, (len(info["regret"]), num_envs)
        assert len(reward) == num_envs, (len(reward), num_envs)

        # record the actions and rewards seen
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # record the _embeddings_ of the chosen actions
        actions[-1] = act_mapper(action)
        rewards[-1] = torch.from_numpy(reward).type(torch.long).to(config.device)

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


def loss_fn(
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
        pred_embs = pred_embs / pred_norm.unsqueeze(-1)
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
    This function calculates the performance of bandit-specific algorithms
    for a later comparison with AD.
    """
    ucb_regrets = {}
    random_regrets = {}
    ts_regrets = {}
    for key, (arms_means, num_arms_per_env) in dists.items():
        # Disabling saving of the histories
        ucb_regrets[key], _, _, _, _ = solve_bandits(
            config=config,
            arms_means=arms_means,
            num_arms=num_arms_per_env,
            basedir=None,
            data_generation_algo="ucb",
            num_env_steps=config.num_in_context_steps,
            seed=config.eval_seed,
        )
        random_regrets[key], _, _, _, _ = solve_bandits(
            config=config,
            arms_means=arms_means,
            num_arms=num_arms_per_env,
            basedir=None,
            data_generation_algo="random",
            num_env_steps=config.num_in_context_steps,
            seed=config.eval_seed,
        )
        ts_regrets[key], _, _, _, _ = solve_bandits(
            config=config,
            arms_means=arms_means,
            num_arms=num_arms_per_env,
            basedir=None,
            data_generation_algo="thompson",
            num_env_steps=config.num_in_context_steps,
            seed=config.eval_seed,
        )

    return random_regrets, ucb_regrets, ts_regrets


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

    # create the bandits for train and test
    (train_dist, train_num_arms), eval_dists = make_envs(config=config)

    # Run the data generation algorithm and log the training histories
    generate_dataset(
        config=config,
        arms_means=train_dist,
        num_arms=train_num_arms,
        seed=config.train_seed,
    )

    # Evaluate some bandit-specific algorithms and log their regrets.
    # That will later be used for comparison with AD
    random_regrets, ucb_regrets, ts_regrets = make_lower_and_upper_regrets(
        config=config, dists=eval_dists
    )
    arrays_to_wandb(logs_dir=config.logs_dir, arrays=random_regrets, name="random")
    arrays_to_wandb(logs_dir=config.logs_dir, arrays=ucb_regrets, name="ucb")
    arrays_to_wandb(logs_dir=config.logs_dir, arrays=ts_regrets, name="ts")

    log_raw_regrets(regrets=ucb_regrets, name="ucb_regrets", step=0)
    log_raw_regrets(regrets=random_regrets, name="random_regrets", step=0)
    log_raw_regrets(regrets=ts_regrets, name="ts_regrets", step=0)

    # Create dataset and dataloader
    dataset = SequenceDataset(
        runs_path=config.learning_histories_path, seq_len=config.seq_len
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
        block_size=2 * config.seq_len + 100,
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
        n_token=config.token_embed_dim,
        use_action_set_prompt=config.use_action_set_prompt,
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
            _, actions, rewards, num_actions_per_env = next(dataloader)
            # Prepare input
            actions = actions.to(torch.long)
            rewards = rewards.to(torch.long)
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
                act_emb=action_embeds,
                rewards=rewards,
                action_set_prompt=action_set_prompt,
            )
            pred = pred[:, :-1]
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
                        num_actions_per_env=num_actions_per_env,
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
            for key, (arms_means, num_arms_per_env) in eval_dists.items():
                (
                    regrets[key],
                    num_uniques[key],
                    entropies[key],
                ) = evaluate_in_context(
                    config,
                    model=model,
                    arms_means=arms_means,
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
            if config.data_generation_algo == "ucb":
                upper_regrets = ucb_regrets
            elif config.data_generation_algo == "thompson":
                upper_regrets = ts_regrets
            else:
                raise NotImplementedError

            log_normalized_regrets(
                raw_regrets=regrets,
                lower_regrets=random_regrets,
                upper_regrets=upper_regrets,
                step=eval_step,
            )

            log_raw_regrets(regrets=regrets, name="raw_regrets", step=eval_step)

            opt_value = 0
            for key in regrets:
                normalized_regret = norm_regret(
                    regrets=regrets[key],
                    lower_regrets=random_regrets[key],
                    upper_regrets=upper_regrets[key],
                )

                opt_value += (num_uniques[key] / eval_dists[key][1].max()) - (
                    normalized_regret - 1
                ) ** 2

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
