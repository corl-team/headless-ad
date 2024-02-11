import gc
import itertools
import os
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

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
from src.gridworld.common.ad_transformer import Model
from src.gridworld.common.data import SequenceDataset
from src.gridworld.common.generate import generate_dataset
from src.gridworld.common.prepare_envs import make_envs
from src.tiny_llama.config import Config as ModelConfig
from src.utils.misc import Timeit, set_seed
from src.utils.schedule import cosine_annealing_with_warmup
from src.utils.wandb_logging import (
    arrays_to_wandb,
    log_in_context,
    log_list,
    log_raw_regrets,
)

if False:
    envs


@dataclass
class Config:
    # wandb params
    project: str = "Headless-AD"
    group: str = "gridworld-ad"
    job_type: str = "debug"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "GridWorld"
    num_train_envs: int = 10_000
    num_in_context_episodes: Optional[int] = None
    learning_histories_path: Optional[str] = "trajectories"

    num_eval_envs: int = 50
    eval_every: int = 1000
    log_every: int = 100

    num_train_steps: int = 100_000

    # Model Params
    seq_len: int = 100
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Training params
    batch_size: int = 64
    learning_rate: float = 3e-3
    beta1: float = 0.9
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    get_action_type: str = "mode"
    rand_select_in_ucb: bool = True
    label_smoothing: float = 0.1

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

    action_space_type: str = "train"

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

    # Create context windows for storing the interaction histories
    states = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )
    states[-1] = torch.from_numpy(init_state).to(config.device).type(torch.long)
    actions = torch.zeros(
        (config.seq_len, num_envs),
        dtype=torch.long,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, num_envs), dtype=torch.long, device=config.device
    )

    # create a list for accumulation of regrets
    all_returns = [[] for _ in range(num_envs)]
    all_lengths = [[] for _ in range(num_envs)]
    current_lengths = np.zeros(num_envs)
    current_returns = np.zeros(num_envs)
    entropies = []
    tried_action_inds = []

    num_dones = np.zeros(num_envs, dtype=np.int32)
    for istep in tqdm(itertools.count(start=1), desc="Eval ..."):
        sliced_states = states.T[:, -istep:]
        sliced_actions = actions.T[:, -istep:]
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
            states=sliced_states, actions=sliced_actions, rewards=sliced_rewards
        )
        pred = pred[:, -1]

        # map predictions to action indices
        # there are two options how to obtain the indices -
        # sampling proportional to the similarities between action embeddings and the prediction
        # and taking the action with the closest embedding
        dist = torch.distributions.Categorical(logits=pred)
        action_sample = dist.sample()
        entropy = dist.entropy()
        action_mode = pred.argmax(dim=-1)

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

        # check whether some environments have finished
        # if yes, add metrics to lists and increment the counters
        for i in np.where(term | trunc)[0]:
            if num_dones[i] < config.num_in_context_episodes:
                all_returns[i].append(current_returns[i])
                all_lengths[i].append(current_lengths[i])

                current_returns[i] = 0
                current_lengths[i] = 0

        # record the actions and rewards seen
        states = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # record the _embeddings_ of the chosen actions
        states[-1] = torch.from_numpy(state).type(torch.long).to(config.device)
        actions[-2] = action
        rewards[-2] = torch.from_numpy(reward).type(torch.long).to(config.device)

        if np.min(num_dones) >= config.num_in_context_episodes:
            break

    # Calculate the average amount of unique actions used by an agent
    # in each environment
    tried_action_inds = np.vstack(tried_action_inds).transpose(0, 1)
    num_unique_in_each_batch = [
        len(np.unique(batch_element, axis=0)) for batch_element in tried_action_inds
    ]
    num_unique_in_each_batch = np.mean(num_unique_in_each_batch)

    model.train()

    returns = np.vstack(all_returns)
    lengths = np.vstack(all_lengths)
    entropies = np.array(entropies)[None, ...]

    return returns, lengths, num_unique_in_each_batch, entropies


def wandb_define_metrics() -> None:
    """
    This function makes the wandb page prettier
    """
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


def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch


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

    wandb.init(
        project=config.project,
        group=config.group,
        job_type=config.job_type,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    wandb_define_metrics()

    # Create the environments for train and test
    (train_acts, train_goals), eval_envs = make_envs(config=config)

    # Run the data generation algorithm and log the training histories
    generate_dataset(config=config, actions=train_acts, goals=train_goals)

    set_seed(seed=config.train_seed)

    # Create a dataset
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
        block_size=3 * config.seq_len + train_acts.shape[1],
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
        actions_per_env=train_acts.shape[1],
        num_states=config.grid_size**2,
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

    # Start training
    for global_step in trange(1, config.num_train_steps + 1, desc="Training"):
        with Timeit() as batch_timer:
            states, actions, rewards = next(dataloader)
            # Prepare input
            states = states.to(torch.long)
            actions = actions.to(torch.long)
            rewards = rewards.to(torch.long)
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

        with Timeit() as pred_timer:
            # Make prediction
            pred = model(
                states=states,
                actions=actions,
                rewards=rewards,
            )
            assert pred.shape[1] == config.seq_len, (pred.shape[1], config.seq_len)

        # Calculate loss
        # Compare predictions at timestep t against the ground truth action at timestep 't + 1'
        with Timeit() as loss_timer:
            with accelerator.autocast():
                loss = torch.nn.functional.cross_entropy(
                    input=pred.flatten(0, 1),
                    target=actions.flatten(0, 1),
                    label_smoothing=config.label_smoothing,
                )

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
                dist = torch.distributions.Categorical(logits=pred)
                act_sample = dist.sample()
                act_mode = pred.argmax(dim=-1)

                accuracy_sample = torch.mean(
                    (act_sample == actions).type(torch.float32)
                )
                accuracy_mode = torch.mean((act_mode == actions).type(torch.float32))

        # log the training metrics
        if global_step % config.log_every == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/accuracy_sample": accuracy_sample.item(),
                    "train/accuracy_mode": accuracy_mode.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
            )

            total_time = (
                batch_timer.elapsed_time_cpu
                + pred_timer.elapsed_time_cpu
                + loss_timer.elapsed_time_cpu
                + back_timer.elapsed_time_cpu
                + metrics_timer.elapsed_time_cpu
            )

            wandb.log(
                {
                    "times/batch_sampling": batch_timer.elapsed_time_cpu,
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

            returns = {}
            lengths = {}
            num_uniques = {}
            entropies = {}

            for key, (acts, goals) in eval_envs.items():
                # Evaluate only on action sets with the same size as
                # the train set's size
                if acts.shape[1] != train_acts.shape[1]:
                    continue
                (
                    returns[key],
                    lengths[key],
                    num_uniques[key],
                    entropies[key],
                ) = evaluate_in_context(
                    config,
                    model=model,
                    acts=acts,
                    goals=goals,
                )

            # create plots with in-context performance and log the to wandb
            log_in_context(values_dict=returns, name="in-context/returns")
            log_in_context(values_dict=lengths, name="in-context/lengths")
            log_in_context(values_dict=entropies, name="in-context/entropy")

            log_raw_regrets(regrets=returns, name="returns", step=eval_step)

            # log regret curves as arrays to wandb
            arrays_to_wandb(
                logs_dir=config.logs_dir, arrays=returns, name=f"ours_{eval_step}"
            )

            # Log the average number of unique actions in all batches
            # during first 'seq_len' steps of evaluation
            log_list(values_dict=num_uniques, name="num_uniques", step=eval_step)

            # Average regrets from all evaluations sets and calculate the bayes objective
            opt_value = 0
            if config.action_space_type == "train":
                keys = ["train_train", "train_test"]
            else:
                keys = returns.keys()

            for key in keys:
                opt_value += returns[key].mean(0)[-1]

            opt_value /= len(returns)

            wandb.log(
                {
                    "final/value": opt_value,
                    "final/step": eval_step,
                }
            )

    wandb.finish()


if __name__ == "__main__":
    train()
