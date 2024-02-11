import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import wandb
from src.utils.misc import norm_regret


def log_raw_regrets(regrets: Dict[str, np.ndarray], name: str, step: int) -> None:
    """
    Takes the last regret and logs it to wandb.
    """
    to_log1 = {f"{name}/{k}": v.mean(axis=0)[-1] for k, v in regrets.items()}
    # to_log2 = {
    #     f"{name}/{k}_hist": wandb.Histogram(v[:, -1]) for k, v in regrets.items()
    # }
    to_log = {
        **to_log1,
        # **to_log2,
        f"{name}/step": step,
    }
    wandb.log(to_log)


def log_in_context(values_dict: Dict[str, np.ndarray], name: str) -> None:
    """
    Creates plots with in-context performance and logs them to wandb as images.
    """
    for key, values in values_dict.items():
        lower, upper = np.percentile(values, q=[5, 95], axis=0)
        plt.plot(values.mean(axis=0))
        plt.fill_between(np.arange(len(lower)), lower, upper, color="b", alpha=0.1)
        plt.title(key)
        plt.ylabel(name)
        plt.xlabel("in-context step")
        wandb.log({f"{name}/{key}": wandb.Image(plt)})
        plt.close()


def log_hists(hists: Dict[str, np.ndarray], step: int) -> None:
    """
    Logs the histories of interactions with an environment during evaluation
    to wandb as tables.
    """
    for key, value in hists.items():
        columns = [[f"a_{i}", f"r_{i}", f"e_{i}"] for i in range(value.shape[1] // 3)]
        columns = sum(columns, [])
        table = wandb.Table(columns=columns, data=value)
        wandb.log({f"hists/{step}_{key}": table})


def log_normalized_regrets(
    raw_regrets: Dict[str, np.ndarray],
    lower_regrets: Dict[str, np.ndarray],
    upper_regrets: Dict[str, np.ndarray],
    step: int,
) -> None:
    """
    Takes the final regrets and normalizes them against random agent and some bandit-specific algorithm.
    Then logs to wandb.
    """
    to_log = {}
    for k in raw_regrets.keys():
        normalized = (lower_regrets[k][:, -1] - raw_regrets[k][:, -1]) / (
            lower_regrets[k][:, -1] - upper_regrets[k][:, -1]
        )
        to_log[f"normalized_regret/{k}"] = np.mean(normalized)

        normalized_means = norm_regret(
            regrets=raw_regrets[k],
            upper_regrets=upper_regrets[k],
            lower_regrets=lower_regrets[k],
        )
        to_log[f"normalized_regret_means/{k}"] = normalized_means

        # to_log[f"normalized_regret/{k}_denom_hist"] = wandb.Histogram(
        #     lower_regrets[k][:, -1] - upper_regrets[k][:, -1]
        # )
        # to_log[f"normalized_regret/{k}_num_hist"] = wandb.Histogram(
        #     lower_regrets[k][:, -1] - raw_regrets[k][:, -1]
        # )
        # to_log[f"normalized_regret/{k}_hist"] = wandb.Histogram(normalized)

    to_log["normalized_regret/step"] = step
    to_log["normalized_regret_means/step"] = step

    wandb.log(to_log)


def log_list(values_dict: Dict[str, np.ndarray], name: str, step: int) -> None:
    log_dict = {f"{name}/{key}": value for key, value in values_dict.items()}
    log_dict = {
        **log_dict,
        f"{name}/step": step,
    }
    wandb.log(log_dict)


def arrays_to_wandb(logs_dir: str, arrays: Dict[str, np.ndarray], name: str):
    """
    Log an array to wandb
    """
    filename = os.path.join(logs_dir, f"{name}.pickle")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "+wb") as f:
        pickle.dump(arrays, f)

    wandb.save(filename)
