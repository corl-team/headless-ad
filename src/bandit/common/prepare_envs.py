from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Config:
    train_seed: int
    eval_seed: int

    train_on_mixed: int
    train_max_arms: int
    train_min_arms: int
    num_train_envs: int
    num_eval_envs: int

    eval_more_arms_list: List[int]


def skewed_reward_dist(rng: np.random.Generator, max_arms: int, num_envs: int):
    """
    This function creates a set of bandits which have the higher rewards distributed over the even arms.
    """
    num_even_arms = max_arms // 2
    num_odd_arms = max_arms - num_even_arms
    means = np.zeros((num_envs, max_arms))
    # 95% of the time - even arms have higher return
    means[:, ::2] = rng.uniform(size=(num_envs, num_odd_arms), low=0.0, high=0.5)
    means[:, 1::2] = rng.uniform(
        size=(num_envs, num_even_arms),
        low=0.5,
        high=1.0,
    )

    return means


def mixed_skewed_reward_dist(
    rng: np.random.Generator, max_arms: int, num_envs: int, frac_first: float
):
    """
    This function creates two sets of bandits. The first one contains bandits with the higher rewards
    distributed over the odd arms. The second set is similar but with the even arms.
    :param max_arms: controls the maximum amount of arms in all bandits
    :param num_envs: the total amount of envs in two sets combined
    :param frac_first: the relative size of the first set compared to the num_envs
    """
    offset = int(num_envs * frac_first)

    # create the first 'odd' set
    means1 = skewed_reward_dist(rng, max_arms=max_arms, num_envs=offset)
    # create the second 'even' set
    means2 = skewed_reward_dist(rng, max_arms=max_arms, num_envs=num_envs - offset)
    means2 = 1 - means2

    # check that the bandits in the sets are correct
    assert means1[0, 0] < means1[0, 1], (means1[0, 0], means1[0, 1])
    assert means2[0, 0] > means2[0, 1], (means2[0, 0], means2[0, 1])

    # combine the two sets
    means = np.concatenate([means1, means2], axis=0)
    # shuffle the bandits
    means = rng.permutation(means)

    return means


def get_num_arms(
    config: Config,
    rng: np.random.Generator,
    num_envs: int,
):
    """
    This function generates the amount of arms for each bandit.
    If 'train_on_mixed' is True, then the amount of arms will be random in the specified range.
    Otherwise, the amount of arms is fixed
    """

    if config.train_on_mixed:
        num_arms = rng.integers(
            low=config.train_min_arms,
            high=config.train_max_arms,
            size=(num_envs,),
            endpoint=True,
        )
    else:
        num_arms = np.full(fill_value=config.train_max_arms, shape=(num_envs,))

    return num_arms


def make_envs(
    config: Config,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    This function will create arm distribution for train and test envs,
    a single one for each episode.
    Instead of putting this logic in the environment itself,
    I do it in the beginning of the program for a better control over
    the kinds of distributions and the relationships between train and test distributions
    I want to get. Also for reproducibility.
    """

    # Higher rewards are more likely to be distributed under
    # odd arms 95% of the time during training
    rng = np.random.default_rng(config.train_seed)

    train_means = mixed_skewed_reward_dist(
        rng,
        max_arms=config.train_max_arms,
        num_envs=config.num_train_envs,
        frac_first=0.95,
    )
    train_num_arms = get_num_arms(
        config=config, rng=rng, num_envs=config.num_train_envs
    )

    eval_dists = {}
    rng = np.random.default_rng(config.eval_seed)
    # Higher rewards are more likely to be distributed under
    # even arms 95% of the time during eval
    means = mixed_skewed_reward_dist(
        rng,
        max_arms=config.train_max_arms,
        num_envs=config.num_eval_envs,
        frac_first=0.05,
    )

    num_arms = get_num_arms(config=config, rng=rng, num_envs=config.num_eval_envs)
    eval_dists["inverse"] = (means, num_arms)

    # The rewards are distributed uniformly over the arms
    means = rng.uniform(
        size=(config.num_eval_envs, config.train_max_arms), low=0.0, high=1.0
    )
    num_arms = get_num_arms(config=config, rng=rng, num_envs=config.num_eval_envs)
    eval_dists["all_new"] = (means, num_arms)

    # Other evaluation sets contain bandits with a fixed and increased amount of arms
    # and a uniform distribution of rewards
    for i in config.eval_more_arms_list:
        means = rng.uniform(size=(config.num_eval_envs, i), low=0.0, high=1.0)
        num_arms = np.full(shape=(config.num_eval_envs,), fill_value=i)
        eval_dists[f"num_{i}"] = (means, num_arms)

    # cut a part of the train bandits for evaluation
    eval_dists["train"] = (
        train_means[: config.num_eval_envs],
        train_num_arms[: config.num_eval_envs],
    )
    return (train_means, train_num_arms), eval_dists
