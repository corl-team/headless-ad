import os
from typing import List

import pandas as pd
import yaml


def env_consistency(df):
    df.loc[
        "num_in_context_episodes", pd.isna(df.loc["num_in_context_episodes"])
    ] = df.loc["num_env_steps"][~pd.isna(df.loc["num_env_steps"])]
    df.drop("num_env_steps", inplace=True)

    return df


def type_cast(value):
    value = str(value)

    try:
        return int(value)
    except Exception:
        try:
            # return float(value)
            value = float(value)
            if value < 0.1:
                return f"{value:.1e}"
            else:
                return round(value, 2)
            # return "%.1e" % Decimal(value)  # float(value)
        except Exception:
            return value


def make_csvs(filenames: List[str], names: List[str], hypers: List[str], csv_name: str):
    df = pd.DataFrame(columns=names, index=hypers)

    for name, filename in zip(names, filenames):
        with open(filename, "r") as f:
            config = yaml.safe_load(f)

        # print(name)
        # print(config)

        for hyper in hypers:
            if hyper in config:
                df.loc[hyper, name] = type_cast(config[hyper])
                # print(config[hyper], type(config[hyper]))

    df = env_consistency(df)
    # print(df)

    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    folder = os.path.join("reports", "hypers")
    os.makedirs(folder, exist_ok=True)
    make_csvs(
        [
            os.path.join("configs", "bandit", "headless_ad.yaml"),
            os.path.join("configs", "gridworld", "headless_ad.yaml"),
            os.path.join("configs", "contextual_bandit", "headless_ad.yaml"),
        ],
        ["Bandit", "Gridworld", "Contextual Bandit"],
        [
            "num_layers",
            "num_heads",
            "d_model",
            "seq_len",
            "tau",
            "learning_rate",
            "weight_decay",
            "beta1",
            "attention_dropout",
            "dropout",
            "num_in_context_episodes",
            "num_env_steps",
            "get_action_type",
        ],
        os.path.join(folder, "headless_ad.csv"),
    )

    make_csvs(
        [
            os.path.join("configs", "bandit", "ad.yaml"),
            os.path.join("configs", "gridworld", "ad.yaml"),
            os.path.join("configs", "contextual_bandit", "ad.yaml"),
        ],
        ["Bandit", "Gridworld", "Contextual Bandit"],
        [
            "num_layers",
            "num_heads",
            "d_model",
            "seq_len",
            "label_smoothing",
            "learning_rate",
            "weight_decay",
            "beta1",
            "attention_dropout",
            "dropout",
            "num_in_context_episodes",
            "num_env_steps",
            "get_action_type",
        ],
        os.path.join(folder, "ad.csv"),
    )

    make_csvs(
        [
            os.path.join("configs", "bandit", "no_act_set_prompt.yaml"),
            os.path.join("configs", "gridworld", "no_act_set_prompt.yaml"),
        ],
        ["Bandit", "Gridworld"],
        [
            "num_layers",
            "num_heads",
            "d_model",
            "seq_len",
            "tau",
            "learning_rate",
            "weight_decay",
            "beta1",
            "attention_dropout",
            "dropout",
            "num_in_context_episodes",
            "num_env_steps",
            "get_action_type",
        ],
        os.path.join(folder, "action_pool_ablation.csv"),
    )

    make_csvs(
        [
            os.path.join("configs", "bandit", "mse_loss.yaml"),
            os.path.join("configs", "gridworld", "mse_loss.yaml"),
        ],
        ["Bandit", "Gridworld"],
        [
            "num_layers",
            "num_heads",
            "d_model",
            "seq_len",
            "learning_rate",
            "weight_decay",
            "beta1",
            "attention_dropout",
            "dropout",
            "num_in_context_episodes",
            "num_env_steps",
            "get_action_type",
        ],
        os.path.join(folder, "mse_ablation.csv"),
    )
