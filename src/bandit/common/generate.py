import json
import multiprocessing as mp
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

import envs
import wandb

# my linter removes variables that are not explicitly used
if False:
    envs


@dataclass
class Config:
    seed: int
    env_name: str
    ucb_alpha: float
    rand_select_in_ucb: bool
    num_env_steps: int
    learning_histories_path: str
    data_generation_algo: str


# If not doing it in this way, have to deal with warnings
def calc_delta(alpha, num_pulled, t):
    if num_pulled > 0:
        return np.sqrt(alpha * np.log(t) / num_pulled)
    else:
        return 1e20


def calc_av_reward(sum_rewards, num_pulled):
    if num_pulled > 0:
        return sum_rewards / num_pulled
    else:
        return 0


calc_delta = np.vectorize(calc_delta)
calc_av_reward = np.vectorize(calc_av_reward)


class UCB:
    def __init__(self, alpha: float, num_arms: int, rand_select: bool):
        self.num_arms = num_arms

        self.num_pulled = np.zeros(num_arms)
        self.sum_rewards = np.zeros(num_arms)
        self.t = 0
        self.alpha = alpha

        self.rand_select = rand_select

    def select_arm(self):
        delta = calc_delta(self.alpha, self.num_pulled, self.t)
        av_reward = calc_av_reward(self.sum_rewards, self.num_pulled)

        # TODO:
        value = av_reward + delta
        if self.rand_select:
            max_arm_value = np.max(value)
            max_arms = np.argwhere(value == max_arm_value).flatten()
            one_max_arm = np.random.choice(max_arms)
        else:
            one_max_arm = np.argmax(value)

        return one_max_arm

    def update_state(self, arm, reward):
        self.num_pulled[arm] += 1
        self.sum_rewards[arm] += reward
        self.t += 1


class ThompsonSampling:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)

    def select_arm(self):
        thetas = np.random.beta(self.alphas, self.betas)

        max_arm = np.argmax(thetas)

        return max_arm

    def update_state(self, arm, reward):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward


class EpsGreedy:
    def __init__(self, max_steps: int, num_arms: int):
        self.num_arms = num_arms
        self.max_steps = max_steps
        self.num_pulled = np.zeros(num_arms)
        self.sum_rewards = np.zeros(num_arms)
        self.eps = 1

    def select_arm(self):
        av_reward = np.where(self.num_pulled > 0, self.sum_rewards / self.num_pulled, 0)

        p = np.random.uniform(low=0, high=1)
        best_act = av_reward.argmax()
        random_act = np.random.randint(low=0, high=self.num_arms)
        act = np.where(p > self.eps, best_act, random_act)

        return act.item()

    def update_state(self, arm, reward):
        self.num_pulled[arm] += 1
        self.sum_rewards[arm] += reward
        # So that the last 10% steps are made without exploration
        self.eps = max(0, self.eps - 1 / (2 / 3 * self.max_steps))


class Random:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def select_arm(self):
        act = np.random.randint(low=0, high=self.num_arms)

        return act

    def update_state(self, arm, reward):
        pass


def save_metadata(savedir, save_filenames):
    metadata = {
        "algorithm": "LinUCB",
        "label": "label",
        "ordered_trajectories": save_filenames,
    }
    with open(os.path.join(savedir, "metadata.metadata"), mode="w") as f:
        json.dump(metadata, f, indent=2)


def dump_trajs(savedir: str, idx: int, trajs: list):
    filename = os.path.join(savedir, f"trajectories_{idx}.npz")
    np.savez(
        filename,
        states=np.array(trajs["states"], dtype=np.float32).reshape(-1, 1),
        actions=np.array(trajs["actions"], dtype=np.int32).reshape(-1, 1),
        rewards=np.array(trajs["rewards"], dtype=np.float32).reshape(-1, 1),
        dones=np.array(
            np.array(trajs["terminateds"]) | np.array(trajs["truncateds"]),
            dtype=np.int32,
        ).reshape(-1, 1),
        num_actions=np.array(trajs["num_actions"], dtype=np.float32),
    )
    return os.path.basename(filename)


def calc_all_actions_i(actions, num_actions):
    """
    Calculates the step when all actions are tried
    """
    acc = np.zeros(num_actions)
    for i, a in enumerate(actions):
        acc[a] += 1
        if np.all(acc > 0):
            return i

    return np.sum(acc > 0)


def solve_bandit(
    env: gym.Env, algo: UCB, savedir: Optional[str], max_steps: int, seed: int
):
    trajs = defaultdict(list)

    state, _ = env.reset(seed=seed)
    regrets = []
    alphas = []
    for _ in range(max_steps):
        action = algo.select_arm()
        new_state, reward, term, trunc, info = env.step(action)
        assert not (term or trunc)

        regrets.append(info["regret"])

        algo.update_state(action, reward)

        # save transitions
        trajs["states"].append(state)
        trajs["actions"].append(action)
        trajs["rewards"].append(reward)
        trajs["terminateds"].append(term)
        trajs["truncateds"].append(trunc)

        state = new_state

        if hasattr(algo, "alpha"):
            alphas.append(algo.alpha)

    alphas = np.array(alphas)

    # fraction of steps when the optimal action was used
    frac_optimal = np.mean(np.array(trajs["actions"]) == info["opt_act"])

    trajs["num_actions"].append(algo.num_arms)
    if savedir is not None:
        filename = dump_trajs(savedir, max_steps, trajs)
        save_metadata(savedir=savedir, save_filenames=[os.path.basename(filename)])

    # record the step index when all actions are tried at least once
    all_actions_i = calc_all_actions_i(trajs["actions"], env.unwrapped.action_space.n)
    return (
        np.array(regrets),
        np.array(alphas),
        np.array(trajs["actions"]),
        frac_optimal,
        all_actions_i,
    )


class Worker:
    """
    This class is used to run the data generation algorithm on a single bandit instance.
    :param data_generation_algo: which algorithm is used for data generation
    :param num_env_steps: number of steps in the environment that an algorithm performs
    :param basedir: where to save the data
    :param seed: a random seed
    """

    def __init__(
        self,
        config: Config,
        data_generation_algo: str,
        num_env_steps: int,
        basedir: str,
        seed: int,
    ):
        self.config = config
        self.basedir = basedir
        self.data_generation_algo = data_generation_algo
        self.num_env_steps = num_env_steps
        self.seed = seed

    def __call__(self, inp):
        means, num_arms = inp
        # Create environment
        env = gym.make(self.config.env_name, arms_mean=means, num_arms=num_arms)
        # Create a random name for this history's logs
        id = uuid.uuid4()
        if self.basedir is not None:
            savedir = os.path.join(self.basedir, f"LinUCB-{id}")
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = None

        # Choose a data generation algorithm
        if self.data_generation_algo == "ucb":
            algo = UCB(
                alpha=self.config.ucb_alpha,
                num_arms=num_arms,
                rand_select=self.config.rand_select_in_ucb,
            )
        elif self.data_generation_algo == "thompson":
            algo = ThompsonSampling(num_arms=num_arms)
        elif self.data_generation_algo == "eps":
            algo = EpsGreedy(max_steps=self.num_env_steps, num_arms=num_arms)
        elif self.data_generation_algo == "random":
            algo = Random(num_arms=num_arms)
        else:
            raise NotImplementedError

        # Run the data generation algorithm
        regrets, alphas, actions, frac_optimal, all_actions_i = solve_bandit(
            env=env,
            algo=algo,
            savedir=savedir,
            max_steps=self.num_env_steps,
            seed=self.seed,
        )

        return regrets, alphas, actions, frac_optimal, all_actions_i


def solve_bandits(
    config: Config,
    arms_means: np.ndarray,
    num_arms: np.ndarray,
    basedir: str,
    data_generation_algo: str,
    num_env_steps: int,
    seed: int,
):
    """
    Run the data generation algorithm on each bandit and get the metrics back.
    """

    # Generate trajectories
    with mp.Pool(processes=os.cpu_count()) as pool:
        out = pool.map(
            Worker(
                config=config,
                basedir=basedir,
                data_generation_algo=data_generation_algo,
                num_env_steps=num_env_steps,
                seed=seed,
            ),
            zip(arms_means, num_arms),
        )
        regrets, alphas, actions, frac_optimal, all_actions_i = zip(*out)

    regrets = np.asarray(regrets)
    actions = np.asarray(actions)
    alphas = np.asarray(alphas)
    frac_optimal = np.asarray(frac_optimal)
    all_actions_i = np.asarray(all_actions_i)

    return regrets, alphas, actions, frac_optimal, all_actions_i


def generate_dataset(
    config: Config, arms_means: np.ndarray, num_arms: np.ndarray, seed: int
):
    """
    This function accepts the configurations of the training bandits
    and runs the data generation algorithms. Also it logs some metrics for later inspection.
    :param arms_means: shape [num_envs, max_arms] - specifies the mean assigned to each bandit arm
    :param num_arms: shape [num_envs] - number of arms in each bandit
    :param seed: a random seed
    """

    # Clean up and then create directory where data will be stored
    if os.path.exists(config.learning_histories_path):
        shutil.rmtree(config.learning_histories_path)
    os.makedirs(config.learning_histories_path)

    # run the algorithm. The data will be saved to disk inside of this function.
    # All returned values are used for logging and ensuring data generation's health.
    regrets, alphas, actions, frac_optimal, all_actions_i = solve_bandits(
        config=config,
        arms_means=arms_means,
        num_arms=num_arms,
        basedir=config.learning_histories_path,
        data_generation_algo=config.data_generation_algo,
        num_env_steps=config.num_env_steps,
        seed=seed,
    )

    def func(hist):
        # This function calculates what is the fraction of the most used action in a history 'hist'
        _, counts = np.unique(hist, return_counts=True)
        max_counts = counts.max()
        max_counts_frac = max_counts / len(hist)

        return max_counts_frac

    max_same_action_frac = np.mean([func(acts) for acts in actions])

    # Log an averaged regret curves along with the percentiles
    regrets_ = np.mean(regrets, axis=0)
    regrets_lb, regrets_ub = np.percentile(regrets, q=[5, 95], axis=0)

    wandb.log(
        {
            "data_gen/regret": wandb.plot.line_series(
                xs=np.arange(len(regrets_lb)),
                ys=[regrets_lb, regrets_, regrets_ub],
                keys=["5%-percentile", "mean", "95%-percentile"],
                title="Data Generation Regret",
                xname="Step",
            )
        }
    )

    wandb.log({"data_gen/max_same_action_frac": max_same_action_frac})

    wandb.log({"data_gen/frac_optimal": wandb.Histogram(frac_optimal)})

    wandb.log({"data_gen/all_actions_i": wandb.Histogram(all_actions_i)})
    wandb.log({"data_gen/last_regret": regrets_[-1]})

    for step, (alpha_min, alpha_max) in enumerate(zip(alphas.min(0), alphas.max(0))):
        wandb.log(
            {
                "data_gen/alpha_min": alpha_min,
                "data_gen/alpha_max": alpha_max,
                "data_gen/step": step,
            }
        )
