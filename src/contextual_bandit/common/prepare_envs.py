from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Config:
    train_seed: int
    eval_seed: int

    num_train_envs: int
    num_eval_envs: int

    train_max_arms: int
    train_min_arms: int

    bandit_context_dim: int
    train_on_mixed: bool
    eval_more_arms_list: List[int]


def get_arm_embeds(
    rng: np.random.Generator, num_envs: int, num_arms: int, dim: int
) -> np.ndarray:
    """
    Returns arm embeddings for all bandits. They are generated from a normal distribution
    with mean 0 and std 1/sqrt(dim)

    :param rng: a random number generator
    :param num_envs: number of bandits.
    :param num_arms: the number of arms in each bandit.
    :param dim: the dimensionality of arm embeddings
    """

    arm_embeds = rng.normal(size=(num_envs, num_arms, dim)) / np.sqrt(dim)

    return arm_embeds


def gen_num_arms(config: Config, rng: np.random.Generator, num_envs: int):
    """
    This function generates the amount of arms for each bandit.
    If 'train_on_mixed' is True, then the amount of arms will be random in the specified range.
    Otherwise, the amount of arms is fixed
    """

    if config.train_on_mixed:
        num_arms = rng.integers(
            low=config.train_min_arms,
            high=config.train_max_arms,
            size=num_envs,
            endpoint=True,
        )
    else:
        num_arms = np.full(fill_value=config.train_max_arms, shape=(num_envs,))

    return num_arms


def make_envs(config: Config):
    """
    This function returns several sets of bandits parameterized
    by the arm ambeddings and the amount of arms in each bandit.

    Instead of putting this logic in the environment itself,
    I do it in the beginning of the program for a better control over
    the kinds of distributions and the relationships between train and test distributions
    I want to get. Also for reproducibility.
    """
    rng = np.random.default_rng(config.train_seed)

    # Train bandits
    train_arm_embeds = get_arm_embeds(
        rng=rng,
        num_envs=config.num_train_envs,
        num_arms=config.train_max_arms,
        dim=config.bandit_context_dim,
    )

    train_num_arms = gen_num_arms(
        config=config, rng=rng, num_envs=config.num_train_envs
    )

    eval_envs = {}
    eval_envs["train"] = (
        train_arm_embeds[: config.num_eval_envs],
        train_num_arms[: config.num_eval_envs],
    )

    # Eval bandits - generated from the same distribution as the train,
    # but containing unseen embeddings
    rng = np.random.default_rng(config.eval_seed)
    eval_arm_embeds = get_arm_embeds(
        rng=rng,
        num_envs=config.num_eval_envs,
        num_arms=config.train_max_arms,
        dim=config.bandit_context_dim,
    )
    eval_num_arms = gen_num_arms(config=config, rng=rng, num_envs=config.num_eval_envs)

    eval_envs["eval"] = (eval_arm_embeds, eval_num_arms)

    # Eval bandits - generated from a scaled distribution compared to the train,
    # but containing unseen embeddings
    eval_arm_embeds = 2 * get_arm_embeds(
        rng=rng,
        num_envs=config.num_eval_envs,
        num_arms=config.train_max_arms,
        dim=config.bandit_context_dim,
    )
    eval_num_arms = gen_num_arms(config=config, rng=rng, num_envs=config.num_eval_envs)

    eval_envs["eval_scaled"] = (eval_arm_embeds, eval_num_arms)

    # Eval bandits - generated from the same distribution as train,
    # but containing a fixed and an increased amount of arms
    for i in config.eval_more_arms_list:
        eval_arm_embeds = get_arm_embeds(
            rng=rng,
            num_envs=config.num_eval_envs,
            num_arms=i,
            dim=config.bandit_context_dim,
        )
        eval_num_arms = np.full(shape=config.num_eval_envs, fill_value=i, dtype=int)

        eval_envs[f"num_{i}"] = (
            eval_arm_embeds,
            eval_num_arms,
        )

    return (train_arm_embeds, train_num_arms), eval_envs
