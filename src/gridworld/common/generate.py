import json
import multiprocessing as mp
import os
import random
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

import envs
import wandb

if False:
    envs


@dataclass
class Config:
    train_seed: int
    env_name: str
    num_train_goals: int
    num_histories: int
    num_episodes: int
    learning_histories_path: str
    visualize: bool
    env_size: int
    action_seq_len: int
    q_learning_lr: float
    q_learning_discount: float
    num_train_envs: int

    check_cached: bool


def q_learning(
    env,
    lr=0.01,
    discount=0.9,
    num_episodes=int(1e7),
    save_every=1000,
    savedir="tmp",
    seed=None,
):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(size=(env.size * env.size, env.action_space.n))
    state, _ = env.reset(seed=seed)

    trajectories = defaultdict(list)
    save_filenames = []
    all_returns = []
    epsilons = []
    current_return = 0
    # creating lists to contain total rewards and steps per episode
    eps = 1.0
    eps_diff = 1.0 / (0.9 * num_episodes)
    episode_i = 0
    term, trunc = False, False
    i = 0
    while True:
        i += 1

        if term or trunc:
            all_returns.append(current_return)
            epsilons.append(eps)

            episode_i += 1
            eps = max(0, eps - eps_diff)
            current_return = 0
            # Get trajectories with optimal actions
            state, _ = env.reset()

        if random.random() < eps:
            a = rng.choice(env.action_space.n)
        else:
            a = Q[state, :].argmax()

        next_state, r, term, trunc, _ = env.step(a)
        current_return += r

        if term:
            Q[next_state, :] = 0

        # Collect trajectories with exploratory actions
        trajectories["states"].append(state)
        trajectories["actions"].append(a)
        trajectories["rewards"].append(r)
        trajectories["terminateds"].append(term)
        trajectories["truncateds"].append(trunc)
        trajectories["qtables"].append(Q)

        # Update Q-Table with new knowledge
        Q[state, a] += lr * (r + discount * np.max(Q[next_state, :]) - Q[state, a])
        state = next_state

        # dump training trajectories
        if savedir is not None and (i % save_every == 0 or episode_i == num_episodes):
            filename = dump_trajectories(savedir, i, trajectories)
            save_filenames.append(os.path.basename(filename))
            trajectories = defaultdict(list)

        if episode_i == num_episodes:
            break

    if savedir is not None:
        save_metadata(savedir, env.goal_pos, save_filenames)

    all_returns = np.array(all_returns)
    epsilons = np.array(epsilons)

    return Q, all_returns, epsilons


def dump_trajectories(savedir, i, trajectories):
    filename = os.path.join(savedir, f"trajectories_{i}.npz")
    np.savez(
        filename,
        states=np.array(trajectories["states"], dtype=float).reshape(-1, 1),
        actions=np.array(trajectories["actions"]).reshape(-1, 1),
        rewards=np.array(trajectories["rewards"], dtype=float).reshape(-1, 1),
        dones=np.int32(
            np.array(trajectories["terminateds"]) | np.array(trajectories["truncateds"])
        ).reshape(-1, 1),
    )

    return os.path.basename(filename)


def save_metadata(savedir, goal_pos, save_filenames):
    metadata = {
        "algorithm": "Q-learning",
        "label": "label",
        "ordered_trajectories": save_filenames,
        "goal": goal_pos.tolist(),
    }
    with open(os.path.join(savedir, "metadata.metadata"), mode="w") as f:
        json.dump(metadata, f, indent=2)


class Worker:
    def __init__(self, config: Config, savedir: Optional[str]):
        self.config = config
        self.savedir = savedir

    def __call__(self, inp):
        goal, actions, inp_i = inp
        env = gym.make(
            self.config.env_name,
            goal_pos=goal,
            available_actions=actions,
            action_seq_len=self.config.action_seq_len,
        )

        id = uuid.uuid4()
        if self.savedir is not None:
            savedir = os.path.join(self.savedir, f"tabularQ-{id}")
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = None

        _, returns, epsilons = q_learning(
            env,
            num_episodes=self.config.num_episodes,
            savedir=savedir,
            lr=self.config.q_learning_lr,
            seed=self.config.train_seed + inp_i,
            discount=self.config.q_learning_discount,
        )

        return returns, epsilons


def solve_env(
    config: Config,
    pool: mp.Pool,
    goals: np.ndarray,
    actions: np.ndarray,
    savedir: Optional[str],
):
    out = pool.map(
        Worker(config, savedir=savedir), zip(goals, actions, range(len(goals)))
    )
    returns, epsilons = zip(*out)

    returns = np.array(returns)
    epsilons = np.array(epsilons)

    return returns, epsilons


def generate_dataset(config: Config, goals: np.ndarray, actions: np.ndarray):
    """
    This function accepts the configurations of the training contextual bandits
    and runs the data generation algorithms. Also it logs some metrics for later inspection.
    :param arm_embeds: shape [num_envs, max_arms] - specifies the embedding assigned to each bandit arm
    :param num_arms: shape [num_envs] - number of arms in each bandit
    :param seed: a random seed
    """

    # Clean up and then create directory where data will be stored
    if os.path.exists(config.learning_histories_path):
        shutil.rmtree(config.learning_histories_path)
    os.makedirs(config.learning_histories_path, exist_ok=True)

    # Generate trajectories
    with mp.Pool(processes=os.cpu_count()) as pool:
        returns, epsilons = solve_env(
            config,
            pool=pool,
            goals=goals,
            actions=actions,
            savedir=config.learning_histories_path,
        )
    assert returns.shape == (config.num_train_envs, config.num_episodes), (
        returns.shape,
        (config.num_train_envs, config.num_episodes),
    )
    assert epsilons.shape == (config.num_train_envs, config.num_episodes), (
        epsilons.shape,
        (config.num_train_envs, config.num_episodes),
    )

    returns_ = np.mean(returns, axis=0)
    returns_lb, returns_ub = np.percentile(returns, q=[5, 95], axis=0)
    wandb.log(
        {
            "data_gen/return": wandb.plot.line_series(
                xs=np.arange(len(returns_lb)),
                ys=[returns_lb, returns_, returns_ub],
                keys=["5%-percentile", "mean", "95%-percentile"],
                title="Data Generation Return",
                xname="Step",
            )
        }
    )

    wandb.log({"data_gen/final_return": returns_[-1]})

    wandb.log({"data_gen/final_return_hist": wandb.Histogram(returns[:, -1])})
    for i, e in enumerate(epsilons.mean(0)):
        wandb.log({"data_gen/epsilon": e, "data_gen/step": i})
