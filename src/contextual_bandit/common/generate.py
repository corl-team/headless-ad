import json
import multiprocessing as mp
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
from numba import njit

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
    bandit_context_dim: int


@njit()
def get_linucb_scores(
    context_histories: List[np.ndarray],
    reward_histories: List[np.ndarray],
    current_context: np.ndarray,
    alpha: float,
):
    upper_bounds = []
    for i in range(len(context_histories)):
        cont = context_histories[i][1:]
        rew = reward_histories[i][1:]
        if len(cont) == 0:
            upper_bounds.append(1e20)
        else:
            A = cont.T @ cont + np.eye(len(current_context))
            A_inv = np.linalg.inv(A)
            b = cont.T @ rew
            theta_estimate = A_inv @ b

            upper_conf = np.sqrt(current_context.T @ A_inv @ current_context)

            reward_estimate = np.dot(current_context, theta_estimate)
            upper_bound = reward_estimate + alpha * upper_conf
            upper_bounds.append(upper_bound)

    upper_bounds = np.array(upper_bounds)

    return upper_bounds


# https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture10/lecture10.pdf
class LinUCB:
    def __init__(self, alpha: float, num_arms: int):
        self.num_arms = num_arms
        self.alpha = alpha

        self.context_histories = [np.zeros((1, 2)) for _ in range(num_arms)]
        self.reward_histories = [np.zeros((1,)) for _ in range(num_arms)]

    def select_arm(self, context):
        upper_bounds = get_linucb_scores(
            context_histories=self.context_histories,
            reward_histories=self.reward_histories,
            current_context=context,
            alpha=self.alpha,
        )

        max_arm_value = np.max(upper_bounds)
        max_arms = np.argwhere(upper_bounds == max_arm_value).flatten()
        one_max_arm = np.random.choice(max_arms)

        return one_max_arm

    def update_state(self, arm, context, reward):
        self.context_histories[arm] = np.concatenate(
            [self.context_histories[arm], context[None, ...]], axis=0
        )
        self.reward_histories[arm] = np.concatenate(
            [self.reward_histories[arm], np.array(reward)[None, ...]], axis=0
        )


class ThompsonSampling:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)

    def select_arm(self, context):
        thetas = np.random.beta(self.alphas, self.betas)

        max_arm = np.argmax(thetas)

        return max_arm

    def update_state(self, arm, context, reward):
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

    def select_arm(self, context):
        act = np.random.randint(low=0, high=self.num_arms)

        return act

    def update_state(self, arm, context, reward):
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
        states=np.array(trajs["states"], dtype=np.float32).reshape(-1, 2),
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
    env: gym.Env, algo: LinUCB, savedir: Optional[str], max_steps: int, seed: int
):
    trajs = defaultdict(list)

    state, _ = env.reset(seed=seed)
    regrets = []
    alphas = []
    for _ in range(max_steps):
        action = algo.select_arm(state)
        new_state, reward, term, trunc, info = env.step(action)
        assert not (term or trunc)

        regrets.append(info["regret"])

        algo.update_state(action, state, reward)

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
        arm_embeds, num_arms = inp
        # Create environment
        env = gym.make(
            self.config.env_name,
            context_dim=self.config.bandit_context_dim,
            arm_embeds=arm_embeds,
            num_arms=num_arms,
        )
        # Create a random name for this history's logs
        id = uuid.uuid4()
        if self.basedir is not None:
            savedir = os.path.join(self.basedir, f"LinUCB-{id}")
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = None

        # Choose a data generation algorithm
        if self.data_generation_algo == "linucb":
            algo = LinUCB(
                alpha=self.config.ucb_alpha,
                num_arms=num_arms,
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
    arm_embeds: np.ndarray,
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
            zip(arm_embeds, num_arms),
        )
        regrets, alphas, actions, frac_optimal, all_actions_i = zip(*out)

    regrets = np.asarray(regrets)
    actions = np.asarray(actions)
    alphas = np.asarray(alphas)
    frac_optimal = np.asarray(frac_optimal)
    all_actions_i = np.asarray(all_actions_i)

    return regrets, alphas, actions, frac_optimal, all_actions_i


def generate_dataset(
    config: Config,
    arm_embeds: np.ndarray,
    num_arms: np.ndarray,
    seed: int,
):
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

    # run the algorithm. The data will be saved to disk inside of this function.
    # All returned values are used for logging and ensuring data generation's health.
    regrets, alphas, actions, frac_optimal, all_actions_i = solve_bandits(
        config=config,
        arm_embeds=arm_embeds,
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

    # compare with the regret bound
    theoretical_bound = np.sqrt(
        config.bandit_context_dim
        * config.train_max_arms
        * np.arange(1, len(regrets_) + 1)
    )
    for i in range(len(regrets_)):
        wandb.log(
            {
                "data_gen/raw_regret": regrets_[i],
                "data_gen/theoretical_bound": theoretical_bound[i],
                "data_gen/frac": theoretical_bound[i] / regrets_[i],
                "data_gen/regrets_step": i,
            }
        )
