from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    train_seed: int

    action_seq_len: int
    train_frac_acts: float
    train_frac_goals: float
    grid_size: int
    num_train_envs: int
    num_eval_envs: int

    action_space_type: str


def goal_splits(
    rng: np.random.Generator,
    train_frac_goals: float,
    grid_size: int,
):
    """
    Splits all cells into two disjoint train and test sets.
    """

    # Goals split
    all_goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    all_goals = rng.permutation(all_goals)

    train_goal_size = int(len(all_goals) * train_frac_goals)
    train_goals, test_goals = all_goals[:train_goal_size], all_goals[train_goal_size:]

    return train_goals, test_goals


def action_splits(
    rng: np.random.Generator,
    action_seq_len: int,
    train_frac_acts: float,
):
    """
    Splits action sequences into two disjoint train and test sets.
    """
    # 5 is the number of atomic actions
    num_actions = 5**action_seq_len
    act_indices = rng.permutation(num_actions)

    train_act_size = int(len(act_indices) * train_frac_acts)
    train_act_indices, test_act_indices = (
        act_indices[:train_act_size],
        act_indices[train_act_size:],
    )

    return train_act_indices, test_act_indices


def sample_n(rng: np.random.Generator, arr: np.ndarray, n: int):
    """
    Samples n (possibly repeating) elements from arr.
    """

    indices = np.arange(len(arr))
    all_indices = rng.choice(indices, size=n, replace=True)
    result = arr[all_indices]

    return result


def get_train_acts(
    config: Config,
    rng: np.random.Generator,
    train_acts_split: np.ndarray,
    test_acts_split: np.ndarray,
):
    """
    Returns an action set that will be used for train.
    The type of the action set is controlled by 'config.action_space_type'.
    """

    if (
        config.action_space_type == "train"
        or config.action_space_type == "for_headless"
    ):
        train_acts = np.tile(train_acts_split, (config.num_train_envs, 1))
    elif config.action_space_type == "perm_train":
        train_acts = np.tile(train_acts_split, (config.num_train_envs, 1))
        train_acts = rng.permutation(train_acts.T).T
    elif config.action_space_type == "test":
        train_acts = np.tile(test_acts_split, (config.num_train_envs, 1))
    elif config.action_space_type == "all":
        all_acts = np.concatenate([train_acts_split, test_acts_split], axis=0)
        train_acts = np.tile(all_acts, (config.num_train_envs, 1))
    else:
        raise NotImplementedError

    return train_acts


def get_eval_envs(
    config: Config,
    rng: np.random.Generator,
    train_goals_split: np.ndarray,
    test_goals_split: np.ndarray,
    train_acts_split: np.ndarray,
    test_acts_split: np.ndarray,
):
    """
    Returns evaluation sets.
    The types of the evaluation sets are controlled by 'config.action_space_type'.
    """

    eval_envs = {}

    if (
        config.action_space_type == "train"
        or config.action_space_type == "perm_train"
        or config.action_space_type == "for_headless"
    ):
        # Train action split
        eval_train = np.tile(train_acts_split, (config.num_eval_envs, 1))
        eval_envs["train_train"] = (
            eval_train,
            sample_n(rng=rng, arr=train_goals_split, n=config.num_eval_envs),
        )
        eval_envs["train_test"] = (
            eval_train,
            sample_n(rng=rng, arr=test_goals_split, n=config.num_eval_envs),
        )

        # Permuted train action split
        eval_train = np.tile(train_acts_split, (config.num_eval_envs, 1))
        eval_perm_train = rng.permutation(eval_train.T).T
        eval_envs["perm_train_train"] = (
            eval_perm_train,
            sample_n(rng=rng, arr=train_goals_split, n=config.num_eval_envs),
        )
        eval_envs["perm_train_test"] = (
            eval_perm_train,
            sample_n(rng=rng, arr=test_goals_split, n=config.num_eval_envs),
        )

        # Test action split
        eval_test = np.tile(test_acts_split, (config.num_eval_envs, 1))
        eval_test = eval_test[:, : len(train_acts_split)]
        eval_envs["cut_test_train"] = (
            eval_test,
            sample_n(rng=rng, arr=train_goals_split, n=config.num_eval_envs),
        )
        eval_envs["cut_test_test"] = (
            eval_test,
            sample_n(rng=rng, arr=test_goals_split, n=config.num_eval_envs),
        )

    if config.action_space_type == "test" or config.action_space_type == "for_headless":
        # Test action split
        eval_test = np.tile(test_acts_split, (config.num_eval_envs, 1))
        eval_envs["test_train"] = (
            eval_test,
            sample_n(rng=rng, arr=train_goals_split, n=config.num_eval_envs),
        )
        eval_envs["test_test"] = (
            eval_test,
            sample_n(rng=rng, arr=test_goals_split, n=config.num_eval_envs),
        )

    if config.action_space_type == "all" or config.action_space_type == "for_headless":
        all_acts = np.concatenate([train_acts_split, test_acts_split], axis=0)
        # Test action split
        eval_all = np.tile(all_acts, (config.num_eval_envs, 1))
        eval_envs["all_train"] = (
            eval_all,
            sample_n(rng=rng, arr=train_goals_split, n=config.num_eval_envs),
        )
        eval_envs["all_test"] = (
            eval_all,
            sample_n(rng=rng, arr=test_goals_split, n=config.num_eval_envs),
        )

    # Check that the amount of environments in each evaluation set is correct.
    for value in eval_envs.values():
        assert value[0].shape[0] == config.num_eval_envs, (
            value[0].shape[0],
            config.num_eval_envs,
        )

    return eval_envs


def make_envs(config: Config):
    """
    Create train and evalution environments.
    """

    rng = np.random.default_rng(config.train_seed)

    # Create train and test splits of goals and actions
    train_goals_split, test_goals_split = goal_splits(
        rng=rng, train_frac_goals=config.train_frac_goals, grid_size=config.grid_size
    )
    train_acts_split, test_acts_split = action_splits(
        rng=rng,
        action_seq_len=config.action_seq_len,
        train_frac_acts=config.train_frac_acts,
    )

    # Create train environments
    train_acts = get_train_acts(
        config=config,
        rng=rng,
        train_acts_split=train_acts_split,
        test_acts_split=test_acts_split,
    )
    train_goals = sample_n(rng=rng, arr=train_goals_split, n=config.num_train_envs)

    # Create evaluation environments
    eval_envs = get_eval_envs(
        config=config,
        rng=rng,
        train_goals_split=train_goals_split,
        test_goals_split=test_goals_split,
        train_acts_split=train_acts_split,
        test_acts_split=test_acts_split,
    )

    # Check that the amount of environments in each set is correct
    assert train_acts.shape[0] == config.num_train_envs, (
        train_acts.shape[0],
        config.num_train_envs,
    )
    for _, (acts, goals) in eval_envs.items():
        assert acts.shape[0] == config.num_eval_envs, (
            acts.shape[0],
            config.num_eval_envs,
        )
        assert goals.shape[0] == config.num_eval_envs, (
            goals.shape[0],
            config.num_eval_envs,
        )

    assert train_goals.shape[0] == config.num_train_envs, (
        train_goals.shape[0],
        config.num_train_envs,
    )

    return (train_acts, train_goals), eval_envs
