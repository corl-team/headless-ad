import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import wandb

NUM_SEEDS = 5
NUM_ENVS = 100
ENV_LEN = 300


def download_file(run: wandb.apis.public.Run, filename: str) -> np.ndarray:
    files = run.files(names=[filename])
    for f in files:
        f.download(root="wandb_downloads", replace=True)

        with open(f"wandb_downloads/{filename}", "rb") as f:
            data = pickle.load(f)

            return data


def download_regrets(run: wandb.apis.public.Run) -> List[Dict[str, np.ndarray]]:
    files = run.files(
        names=[
            "logs/random.pickle",
            "logs/ucb.pickle",
            "logs/ours_200.pickle",
            "logs/ts.pickle",
        ]
    )
    for f in files:
        f.download(root="wandb_downloads", replace=True)

    with open("wandb_downloads/logs/ucb.pickle", "rb") as f:
        ucb_regrets = pickle.load(f)

    with open("wandb_downloads/logs/random.pickle", "rb") as f:
        random_regrets = pickle.load(f)

    with open("wandb_downloads/logs/ts.pickle", "rb") as f:
        ts_regrets = pickle.load(f)

    with open("wandb_downloads/logs/ours_200.pickle", "rb") as f:
        headless_regrets = pickle.load(f)

    return ucb_regrets, random_regrets, ts_regrets, headless_regrets


def merge_seeds(regrets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    merged_regrets = {}
    for key in regrets[0].keys():
        key_data = [regret[key] for regret in regrets]
        merged_regrets[key] = np.vstack(key_data)

    return merged_regrets


def load_data(api: wandb.Api, sweep: str) -> List[Dict[str, np.ndarray]]:
    runs = api.sweep(sweep).runs
    seeds_data = [download_regrets(run) for run in runs]

    return list(map(merge_seeds, zip(*seeds_data)))


def iqm_in_seeds(regrets):
    seed_grouped_regrets = regrets.reshape(NUM_SEEDS, -1, ENV_LEN)
    # regrets_iqm = scipy.stats.trim_mean(
    #     seed_grouped_regrets, proportiontocut=0.25, axis=1
    # )

    regrets_iqm = np.mean(seed_grouped_regrets, axis=1)

    return regrets_iqm


def plot_curve(ax, regrets, color, name, **kwargs):
    regrets_iqm = iqm_in_seeds(regrets)
    regrets_mean = regrets_iqm.mean(axis=0)
    ax.plot(regrets_mean, label=name, color=color, **kwargs)
    std = regrets_iqm.std(axis=0)
    ax.fill_between(
        np.arange(len(std)),
        regrets_mean - std,
        regrets_mean + std,
        color=color,
        alpha=0.1,
    )
    ax.set_xlim(0, ENV_LEN)
    ax.set_xticks(np.arange(0, ENV_LEN + 1, 100))


def regret_curve_plot(regrets, name_mapper):
    ucb_regrets, random_regrets, ts_regrets, ours_regrets, ad_regrets = regrets

    for key in regrets[0].keys():
        fig = plt.figure(layout="constrained")
        ax = plt.subplot(111)
        # ax.grid()
        # ax.set_axisbelow(True)
        # plot_curve(ax, random_regrets[key], "r", name="Random")
        plot_curve(ax, ts_regrets[key], "#99cc66", name="Thompson")
        plot_curve(ax, ours_regrets[key], "#66cccc", name="Headless-AD (ours)")
        if key in ad_regrets:
            plot_curve(ax, ad_regrets[key], "#9966cc", name="AD")
        # plot_curve(ax, ts_regrets[key], "r", name="Thompson")
        ax.set_title(name_mapper[key])
        ax.legend(loc="upper left")
        ax.set_ylabel("Regret")
        ax.set_xlabel("Step")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.savefig(
            f"plots/regret_curves/{key}.pdf",
            bbox_inches="tight",
            format="pdf",
            transparent=True,
        )
        plt.close()


def regret_curve_together_plot(
    regrets: Dict[str, np.ndarray],
    arm_keys: List[str],
    name_mapper: Dict[str, str],
    algo_names: List[str],
    colors: List[str],
    filename: str,
):
    ucb_regrets, random_regrets, ts_regrets, ours_regrets, ad_regrets = regrets
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(arm_keys),
        layout="constrained",
        figsize=(20, 4),
    )
    # for key in ["num_10", "num_20", "num_25", "num_30", "num_40", "num_50"]:
    for i, key in enumerate(arm_keys):
        # ax[i].grid()
        plot_curve(
            ax[i],
            random_regrets[key],
            "grey",
            name=algo_names[0],
            linestyle="dashed",
            linewidth=2,
        )
        plot_curve(ax[i], ts_regrets[key], colors[1], name=algo_names[1], linewidth=2)
        plot_curve(ax[i], ours_regrets[key], colors[2], name=algo_names[2], linewidth=2)
        plot_curve(ax[i], ad_regrets[key], colors[3], name=algo_names[3], linewidth=2)
        ax[i].set_ylim(0, 100)
        ax[i].set_yticks(np.arange(0, 101, 25))
        ax[i].set_title(name_mapper[key], fontsize=28)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        # mode="expand",
        bbox_to_anchor=(0, 1.02, 1.0, 0.2),
        borderaxespad=0,
        fontsize=28,
        ncol=4,
    )
    fig.text(
        0.5, -0.05, "Step", ha="center", va="center", fontsize=plt.rcParams["font.size"]
    )
    fig.text(
        -0.02,
        0.5,
        "Regret",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=plt.rcParams["font.size"],
    )

    fig.savefig(filename, bbox_inches="tight", format="pdf", transparent=True)
    plt.close()


def load_ad_data(api: wandb.Api, ad_sweep_name: str):
    sweep = api.sweep(ad_sweep_name)
    ad_regrets = {}
    for run_ in tqdm(sweep.runs, total=sweep.runs.length):
        files = run_.files(["logs/ours_100.pickle"])
        for f in files:
            f.download(root=os.path.join("wandb_downloads", "ad"), replace=True)

        with open("wandb_downloads/ad/logs/ours_100.pickle", "rb") as f:
            ad_regret = pickle.load(f)

        num_arms = json.loads(run_.json_config)["train_max_arms"]["value"]
        num_arms = f"num_{num_arms}"

        if num_arms in ad_regrets:
            ad_regrets[num_arms] = np.vstack(
                [ad_regrets[num_arms], ad_regret["all_new"]]
            )
        else:
            ad_regrets[num_arms] = ad_regret["all_new"]

    return ad_regrets


def bandit_regret_plots(
    api: wandb.Api, headless_sweep: str, ad_sweep: str, colors: List[str], folder: str
):
    ad_regrets = load_ad_data(api, ad_sweep)

    regrets = load_data(api=api, sweep=headless_sweep)

    regrets = (*regrets, ad_regrets)

    name_mapper = {
        "train": "Odd (seen)",
        "inverse": "Even (unseen)",
        "all_new": "Uniform (unseen)",
        "num_5": "5 Arms",
        "num_10": "10 Arms",
        "num_15": "15 Arms",
        "num_20": "20 Arms",
        "num_25": "25 Arms",
        "num_30": "30 Arms",
        "num_40": "40 Arms",
        "num_50": "50 Arms",
        "num_100": "100 Arms",
    }

    regret_curve_together_plot(
        regrets,
        arm_keys=["num_20", "num_25", "num_30", "num_40", "num_50"],
        name_mapper=name_mapper,
        algo_names=["Random", "Thompson", "Headless-AD (ours)", "AD"],
        colors=colors,
        filename=os.path.join(folder, "bandit_together.pdf"),
    )


def data_from_runs(
    runs, algos: List[str], keys: List[str], is_ad: Optional[bool] = False
):
    metrics = defaultdict(lambda: defaultdict(list))

    for algo in algos:
        for key in keys:
            combined_key = f"{algo}/{key}"
            for run_ in runs:
                if combined_key in run_.summary:
                    hist = run_.history(keys=[combined_key])[combined_key].values
                    # "all" and "test" action sets have drops in performance,
                    # so I take the performance value from an intermediate step.
                    if is_ad and (key.startswith("all") or key.startswith("test")):
                        metrics[algo][key].append(hist[59])
                    else:
                        metrics[algo][key].append(hist[-1])

    for algo in algos:
        for key in keys:
            if key in metrics[algo]:
                metrics[algo][key] = {
                    "mean": np.mean(metrics[algo][key]),
                    "std": np.std(metrics[algo][key]),
                }

    return metrics


def data_from_contextual_ad_runs(runs):
    all_regrets = defaultdict(list)

    for run_ in runs:
        num_arms = json.loads(run_.json_config)["train_max_arms"]["value"]
        num_arms = f"num_{num_arms}"
        all_regrets[num_arms].append(run_.summary["raw_regrets/train"])

    for key, value in all_regrets.items():
        all_regrets[key] = {"mean": np.mean(value), "std": np.std(value)}

    return all_regrets


def take_key(
    data: Dict[str, Dict[str, float]],
    key_order: List[str],
    subkey: str,
):
    values = []
    for key in key_order:
        if subkey in data[key]:
            values.append(data[key][subkey])
        else:
            values.append(0)

    return values


def make_barplot(
    data: Dict[str, Dict[str, Dict[str, float]]],
    barwidth: float,
    algo_name_map: Dict[str, str],
    key_name_map: Dict[str, str],
    colors: List[str],
    filename: str,
    ylabel: str = "Regret",
    xlabel: str = "Eval Setting",
    figsize: Tuple[int, int] = (40, 8),
):
    tick_width = (len(algo_name_map) + 1) * barwidth
    X_axis = np.arange(len(key_name_map)) * tick_width
    corner = (len(algo_name_map) - 1) * (barwidth / 2)
    offsets = np.linspace(-corner, corner + barwidth // 2, num=len(algo_name_map))[::-1]
    eval_setting_keys = list(key_name_map.keys())
    fig, ax = plt.subplots(figsize=figsize)
    for i, key in enumerate(algo_name_map.keys()):
        ax.bar(
            x=X_axis - offsets[i],
            height=take_key(data[key], eval_setting_keys, "mean"),
            width=barwidth,
            color=colors[i],
            yerr=take_key(data[key], eval_setting_keys, "std"),
            label=algo_name_map[key],
            error_kw=dict(lw=3, color="#0c0c0c"),
        )
    ax.margins(x=0.005)
    # ax.grid()
    # ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(X_axis, [key_name_map[x] for x in eval_setting_keys], fontsize=34)
    ax.set_xlabel(xlabel, fontsize=34, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(left=True, bottom=False, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        # bbox_to_anchor=(0, 1.02, 1.0, 0.2),
        fontsize=28,
        ncol=len(data),
    )
    fig.savefig(filename, bbox_inches="tight", format="pdf", transparent=True)


def make_2row_barplot(
    data: Dict[str, Dict[str, Dict[str, float]]],
    barwidth: float,
    algo_name_map: Dict[str, str],
    key_name_map: Dict[str, str],
    colors: List[str],
    filename: str,
    common_xlabel: str,
    **kwargs,
):
    tick_width = (len(algo_name_map) + 1) * barwidth
    X_axis = np.arange(len(key_name_map) // 2) * tick_width
    corner = (len(algo_name_map) - 1) * (barwidth / 2)
    offsets = np.linspace(-corner, corner + barwidth // 2, num=len(algo_name_map))[::-1]
    eval_setting_keys = list(key_name_map.keys())
    # fig, axs = plt.subplots(figsize=(40, 8), nrows=2)
    fig, axs = plt.subplots(nrows=2, **kwargs)
    fig.tight_layout()
    for row in range(2):
        for i, key in enumerate(data.keys()):
            height = take_key(data[key], eval_setting_keys, "mean")[
                row * len(X_axis) : (row + 1) * len(X_axis)
            ]
            yerr = take_key(data[key], eval_setting_keys, "std")[
                row * len(X_axis) : (row + 1) * len(X_axis)
            ]
            axs[row].bar(
                x=X_axis - offsets[i],
                height=height,
                width=barwidth,
                color=colors[i],
                yerr=yerr,
                label=algo_name_map[key],
                error_kw=dict(lw=3, color="#0c0c0c"),
            )
        axs[row].set_xticks(
            X_axis,
            [key_name_map[x] for x in eval_setting_keys][
                row * len(X_axis) : (row + 1) * len(X_axis)
            ],
            fontsize=38,
        )
        axs[row].margins(x=0.005)
        # axs[row].grid()
        # axs[r].set_axisbelow(True)
        axs[row].spines[["right", "top"]].set_visible(False)
        axs[row].tick_params(left=True, bottom=False, pad=10)

    axs[0].set_ylabel("Train Goals", fontsize=38)
    axs[1].set_ylabel("Test Goals", fontsize=38)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        # loc="outside upper center",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.96),  # , 1.0, 0.2),
        fontsize=28,
        ncol=len(algo_name_map),
    )
    if common_xlabel is not None:
        fig.text(
            0.5,
            -0.05,
            common_xlabel,
            ha="center",
            va="center",
            fontsize=38,
        )
    fig.savefig(filename, bbox_inches="tight", format="pdf", transparent=True)


def bandit_plot(
    api: wandb.Api, headless_sweep: str, ad_sweep: str, colors: list[str], folder: str
):
    exp_keys = [
        "train",
        "inverse",
        "all_new",
        "num_20",
        "num_25",
        "num_30",
        "num_40",
        "num_50",
        "num_100",
    ]
    headless_runs = api.sweep(headless_sweep).runs
    headless_data = data_from_runs(
        runs=headless_runs,
        algos=[
            "random_regrets",
            "ts_regrets",
            "raw_regrets",
            "normalized_regret_means",
        ],
        keys=exp_keys,
    )

    make_barplot(
        data=headless_data,
        barwidth=0.4,
        algo_name_map={
            "random_regrets": "Random",
            "ts_regrets": "Thompson Sampling",
            "raw_regrets": "Headless-AD (ours)",
        },
        key_name_map={
            "train": "Odd (seen)",
            "inverse": "Even (unseen)",
            "all_new": "Uniform (unseen)",
        },
        colors=colors,
        filename=os.path.join(folder, "bandit_new_regrets.pdf"),
        figsize=(20, 8),
        ylabel="Regret",
        xlabel="Reward Distribution",
    )

    bandit_regret_plots(
        api=api,
        headless_sweep=headless_sweep,
        ad_sweep=ad_sweep,
        colors=colors,
        folder=folder,
    )


def contextual_plot(
    api: wandb.Api, headless_sweep: str, ad_sweep: str, colors: list[str], folder: str
):
    exp_keys = [
        "train",
        # "eval_scaled",
        "num_20",
        "num_25",
        "num_30",
        "num_40",
        "num_50",
        "num_100",
    ]
    headless_runs = api.sweep(headless_sweep).runs
    headless_data = data_from_runs(
        runs=headless_runs,
        algos=["random_regrets", "ucb_regrets", "raw_regrets"],
        keys=exp_keys,
    )

    ad_runs = api.sweep(ad_sweep).runs

    ad_data = data_from_contextual_ad_runs(runs=ad_runs)
    data = {**headless_data}
    data["ad"] = ad_data

    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "random_regrets": "Random",
            "ucb_regrets": "LinUCB",
            "raw_regrets": "Headless-AD (ours)",
            "ad": "AD",
        },
        key_name_map={
            "train": "Train",
            # "eval_scaled": "Eval",
            "num_20": "20 Arms",
            "num_25": "25 Arms",
            "num_30": "30 Arms",
            "num_40": "40 Arms",
            "num_50": "50 Arms",
            "num_100": "100 Arms",
        },
        colors=colors,
        filename=os.path.join(folder, "contextual_bars.pdf"),
        xlabel="Evaluation Setting",
        ylabel="Regret",
    )


def gridworld_plot(
    api: wandb.Api, headless_sweep: str, ad_sweep: str, colors: list[str], folder: str
):
    exp_keys = [
        "train_train",
        "perm_train_train",
        "cut_test_train",
        "test_train",
        "all_train",
        "train_test",
        "perm_train_test",
        "cut_test_test",
        "test_test",
        "all_test",
    ]
    headless_runs = api.sweep(headless_sweep).runs
    headless_data = data_from_runs(
        runs=headless_runs,
        algos=["random_model", "q_learning", "returns"],
        keys=exp_keys,
    )

    ad_runs = api.sweep(ad_sweep).runs
    ad_runs = [
        run_ for run_ in ad_runs if run_.config["action_space_type"] != "perm_train"
    ]

    ad_data = data_from_runs(runs=ad_runs, algos=["returns"], keys=exp_keys, is_ad=True)
    data = {**headless_data}
    data["ad"] = ad_data["returns"]

    make_2row_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "random_model": "Random",
            "q_learning": "Q-Learning",
            "returns": "Headless-AD (ours)",
            "ad": "AD",
        },
        key_name_map={
            "train_train": "Train",
            "perm_train_train": "Permuted Train",
            "cut_test_train": "Sliced Test",
            "test_train": "Test",
            "all_train": "All",
            "train_test": "Train",
            "perm_train_test": "Permuted Train",
            "cut_test_test": "Sliced Test",
            "test_test": "Test",
            "all_test": "All",
        },
        colors=colors,
        filename=os.path.join(folder, "gridworld_bars.pdf"),
        figsize=(40, 10),
        common_xlabel="Action Set",
    )


def gridworld_compare_ad_trainings(
    api: wandb.Api, ad_sweep: str, colors: list[str], folder: str
):
    ad_runs = api.sweep(ad_sweep).runs
    runs_vanilla = [
        run_ for run_ in ad_runs if run_.config["action_space_type"] == "train"
    ]
    runs_perm = [
        run_ for run_ in ad_runs if run_.config["action_space_type"] == "perm_train"
    ]

    keys = [
        "perm_train_train",
        "cut_test_train",
        "perm_train_test",
        "cut_test_test",
    ]

    data_vanilla = data_from_runs(
        runs=runs_vanilla, algos=["returns"], keys=keys, is_ad=True
    )
    data_perm = data_from_runs(runs=runs_perm, algos=["returns"], keys=keys, is_ad=True)

    data = {}
    data["vanilla"] = data_vanilla["returns"]
    data["perm"] = data_perm["returns"]

    make_2row_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla": "AD",
            "perm": "AD-permuted",
        },
        key_name_map={
            "perm_train_train": "Permuted Train",
            "cut_test_train": "Sliced Test",
            "perm_train_test": "Permuted Train",
            "cut_test_test": "Sliced Test",
        },
        colors=colors,
        filename=os.path.join(folder, "gridworld_vanilla_perm.pdf"),
        figsize=(12, 10),
        common_xlabel=None,
    )


def gridworld_actions_used_plot(
    api: wandb.Api, sweep_name: str, colors: List[str], folder: str
):
    runs = api.sweep(sweep_name).runs

    train_tried = [
        download_file(run_, "logs/train_tried_10.pickle")["train_tried"]
        for run_ in runs
    ]
    test_tried = [
        download_file(run_, "logs/test_tried_10.pickle")["test_tried"] for run_ in runs
    ]

    train_tried = np.stack([x.mean(0) for x in train_tried], axis=0)
    test_tried = np.stack([x.mean(0) for x in test_tried], axis=0)

    train_tried_mean = train_tried.mean(0)
    test_tried_mean = test_tried.mean(0)
    train_tried_std = train_tried.std(0)
    test_tried_std = test_tried.std(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.grid()
    ax.spines[["right", "top"]].set_visible(False)
    ax.plot(train_tried_mean, label="Train", color=colors[0], linewidth=2)
    ax.fill_between(
        np.arange(len(train_tried_mean)),
        train_tried_mean - train_tried_std,
        train_tried_mean + train_tried_std,
        color=colors[0],
        alpha=0.1,
    )

    ax.plot(test_tried_mean, label="Test", color=colors[1], linewidth=2)
    ax.fill_between(
        np.arange(len(test_tried_mean)),
        test_tried_mean - test_tried_std,
        test_tried_mean + test_tried_std,
        color=colors[1],
        alpha=0.1,
    )

    ax.set_xlabel("In-Context Episode", fontsize=28)
    ax.set_ylabel("Fraction of Actions", fontsize=28)
    ax.legend(loc="best", fontsize=28)
    fig.savefig(
        os.path.join(folder, "gridworld_actions_used.pdf"),
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )


def gridworld_ablation_plot(
    api: wandb.Api,
    normal_sweep: str,
    ablated_sweep: str,
    colors: List[str],
    folder: str,
):
    exp_keys = [
        "train_train",
        "perm_train_train",
        "cut_test_train",
        "test_train",
        "all_train",
        "train_test",
        "perm_train_test",
        "cut_test_test",
        "test_test",
        "all_test",
    ]
    normal_runs = api.sweep(normal_sweep).runs
    normal_data = data_from_runs(
        runs=normal_runs,
        algos=["returns"],
        keys=exp_keys,
    )
    to_compare_runs = api.sweep(ablated_sweep).runs
    to_compare_data = data_from_runs(
        runs=to_compare_runs,
        algos=["returns"],
        keys=exp_keys,
    )

    data = {}
    data["vanilla"] = normal_data["returns"]
    data["ablated"] = to_compare_data["returns"]

    make_2row_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla": "w/ Action Set Prompt",
            "ablated": "wo/ Action Set Prompt",
        },
        key_name_map={
            "train_train": "Train",
            "perm_train_train": "Permuted Train",
            "cut_test_train": "Sliced Test",
            "test_train": "Test",
            "all_train": "All",
            "train_test": "Train",
            "perm_train_test": "Permuted Train",
            "cut_test_test": "Sliced Test",
            "test_test": "Test",
            "all_test": "All",
        },
        colors=colors,
        filename=os.path.join(folder, "gridworld_act_pool_ablation.pdf"),
        figsize=(40, 10),
        common_xlabel="Action Set",
    )


def bandit_ablation_plot(
    api: wandb.Api,
    normal_sweep: str,
    ablated_sweep: str,
    colors: List[str],
    folder: str,
):
    exp_keys = [
        "train",
        "inverse",
        "all_new",
        "num_20",
        "num_25",
        "num_30",
        "num_40",
        "num_50",
    ]
    algo_key = "normalized_regret_means"
    normal_runs = api.sweep(normal_sweep).runs
    normal_data = data_from_runs(
        runs=normal_runs,
        algos=[algo_key],
        keys=exp_keys,
    )
    to_compare_runs = api.sweep(ablated_sweep).runs
    to_compare_data = data_from_runs(
        runs=to_compare_runs,
        algos=[algo_key],
        keys=exp_keys,
    )

    data = {}
    data["vanilla"] = normal_data[algo_key]
    data["ablated"] = to_compare_data[algo_key]

    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla": "w/ Action Set Prompt",
            "ablated": "wo/ Action Set Prompt",
        },
        key_name_map={
            "train": "Odd\n(seen)",
            "inverse": "Even\n(unseen)",
            "all_new": "Uniform\n(unseen)",
            "num_20": "20 Arms\n(seen)",
            "num_25": "25 Arms\n(unseen)",
            "num_30": "30 Arms\n(unseen)",
            "num_40": "40 Arms\n(unseen)",
            "num_50": "50 Arms\n(unseen)",
        },
        colors=colors,
        ylabel="Normalized Regret",
        filename=os.path.join(folder, "bandit_act_pool_ablation.pdf"),
    )


def data_for_mean_ablation_plot(
    api: wandb.Api,
    bandit_normal_sweep: str,
    gridworld_normal_sweep: str,
    bandit_ablated_sweep: str,
    gridworld_ablated_sweep: str,
):
    exp_keys = [
        "train_train",
        "perm_train_train",
        "cut_test_train",
        "test_train",
        "all_train",
        "train_test",
        "perm_train_test",
        "cut_test_test",
        "test_test",
        "all_test",
    ]
    normal_runs = api.sweep(gridworld_normal_sweep).runs
    normal_data = data_from_runs(
        runs=normal_runs,
        algos=["returns", "num_uniques"],
        keys=exp_keys,
    )
    to_compare_runs = api.sweep(gridworld_ablated_sweep).runs
    to_compare_data = data_from_runs(
        runs=to_compare_runs,
        algos=["returns", "num_uniques"],
        keys=exp_keys,
    )

    gridworld_data = {}
    gridworld_data["vanilla"] = normal_data["returns"]
    gridworld_data["vanilla_acts"] = normal_data["num_uniques"]
    gridworld_data["ablated"] = to_compare_data["returns"]
    gridworld_data["ablated_acts"] = to_compare_data["num_uniques"]

    exp_keys = [
        "train",
        "inverse",
        "all_new",
        "num_20",
        "num_25",
        "num_30",
        "num_40",
        "num_50",
    ]
    normal_runs = api.sweep(bandit_normal_sweep).runs
    normal_data = data_from_runs(
        runs=normal_runs,
        algos=["normalized_regret_means", "num_uniques"],
        keys=exp_keys,
    )
    to_compare_runs = api.sweep(bandit_ablated_sweep).runs
    to_compare_data = data_from_runs(
        runs=to_compare_runs,
        algos=["normalized_regret_means", "num_uniques"],
        keys=exp_keys,
    )

    bandit_data = {}
    bandit_data["vanilla"] = normal_data["normalized_regret_means"]
    bandit_data["vanilla_acts"] = normal_data["num_uniques"]
    bandit_data["ablated"] = to_compare_data["normalized_regret_means"]
    bandit_data["ablated_acts"] = to_compare_data["num_uniques"]

    data = {}

    def pp(x):
        return {
            "bandit": {
                "mean": np.mean([x["mean"] for x in bandit_data[x].values()]),
                "std": np.std([x["mean"] for x in bandit_data[x].values()]),
            },
            "gridworld": {
                "mean": np.mean([x["mean"] for x in gridworld_data[x].values()]),
                "std": np.std([x["mean"] for x in gridworld_data[x].values()]),
            },
        }

    data["vanilla"] = pp("vanilla")
    data["vanilla_acts"] = pp("vanilla_acts")
    data["ablated"] = pp("ablated")
    data["ablated_acts"] = pp("ablated_acts")

    return data


def mean_prompt_ablation_plot(
    data: Dict[str, Dict[str, Dict[str, float]]], colors: List[str], folder: str
):
    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla": "w/ Action Set Prompt",
            "ablated": "wo/ Action Set Prompt",
        },
        key_name_map={"bandit": "Bernoulli Bandit", "gridworld": "Darkroom"},
        colors=colors,
        filename=os.path.join(folder, "means_act_pool_ablation.pdf"),
        ylabel="Performance",
        xlabel="Environment",
        figsize=(20, 8),
    )

    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla_acts": "w/ Action Set Prompt",
            "ablated_acts": "wo/ Action Set Prompt",
        },
        key_name_map={"bandit": "Bernoulli Bandit", "gridworld": "Darkroom"},
        colors=colors,
        filename=os.path.join(folder, "means_act_pool_ablation_num_acts.pdf"),
        ylabel="#Actions",
        xlabel="Environment",
        figsize=(20, 8),
    )


def mean_loss_ablation_plot(
    data: Dict[str, Dict[str, Dict[str, float]]], colors: List[str], folder: str
):
    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla": "Contrastive Loss",
            "ablated": "MSE Loss",
        },
        key_name_map={"bandit": "Bernoulli Bandit", "gridworld": "Darkroom"},
        colors=colors,
        filename=os.path.join(folder, "means_mse_ablation.pdf"),
        ylabel="Performance",
        xlabel="Environment",
        figsize=(20, 8),
    )

    make_barplot(
        data=data,
        barwidth=0.4,
        algo_name_map={
            "vanilla_acts": "Contrastive Loss",
            "ablated_acts": "MSE Loss",
        },
        key_name_map={"bandit": "Bernoulli Bandit", "gridworld": "Darkroom"},
        colors=colors,
        filename=os.path.join(folder, "means_mse_ablation_num_acts.pdf"),
        ylabel="#Actions",
        xlabel="Environment",
        figsize=(20, 8),
    )


def bad_ad_plot(api: wandb.Api, ad_sweep: str, colors: List[str], folder: str):
    ad_runs = api.sweep(ad_sweep).runs
    ad_runs = [
        run_ for run_ in ad_runs if run_.config["action_space_type"] != "perm_train"
    ]

    train_keys = ["train_train", "train_test"]
    # 'sem' stands for semantic change
    sem_keys = [
        "perm_train_train",
        "cut_test_train",
        "perm_train_test",
        "cut_test_test",
    ]
    # Load AD data
    ad_train = data_from_runs(
        runs=ad_runs, algos=["returns"], keys=train_keys, is_ad=True
    )
    ad_sem = data_from_runs(
        runs=ad_runs,
        algos=["returns"],
        keys=sem_keys,
        is_ad=True,
    )

    ad_train_mean = np.mean([ad_train["returns"][key]["mean"] for key in train_keys])
    ad_train_std = np.std([ad_train["returns"][key]["mean"] for key in train_keys])
    ad_sem_mean = np.mean([ad_sem["returns"][key]["mean"] for key in sem_keys])
    ad_sem_std = np.std([ad_sem["returns"][key]["mean"] for key in sem_keys])

    # Load AD-permuted data
    ad_runs = api.sweep(ad_sweep).runs
    ad_perm_runs = [
        run_ for run_ in ad_runs if run_.config["action_space_type"] == "perm_train"
    ]
    ad_perm_train = data_from_runs(
        runs=ad_perm_runs, algos=["returns"], keys=train_keys, is_ad=True
    )
    ad_perm_sem = data_from_runs(
        runs=ad_perm_runs,
        algos=["returns"],
        keys=sem_keys,
        is_ad=True,
    )

    ad_perm_train_mean = np.mean(
        [ad_perm_train["returns"][key]["mean"] for key in train_keys]
    )
    ad_perm_train_std = np.std(
        [ad_perm_train["returns"][key]["mean"] for key in train_keys]
    )
    ad_perm_sem_mean = np.mean(
        [ad_perm_sem["returns"][key]["mean"] for key in sem_keys]
    )
    ad_perm_sem_std = np.std([ad_perm_sem["returns"][key]["mean"] for key in sem_keys])

    barwidth = 0.4
    X_axis = np.arange(3) * 0.4 * 3
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(
        x=X_axis[:-1] - 0.2,
        height=[ad_train_mean, ad_sem_mean],
        width=barwidth,
        color=colors[1],
        yerr=[ad_train_std, ad_sem_std],
        label="AD",
        error_kw=dict(lw=3, color="#0c0c0c"),
    )
    ax.bar(
        x=X_axis[:-1] + 0.2,
        height=[ad_perm_train_mean, ad_perm_sem_mean],
        width=barwidth,
        color=colors[2],
        yerr=[ad_perm_train_std, ad_perm_sem_std],
        label="AD-permuted",
        error_kw=dict(lw=3, color="#0c0c0c"),
    )
    ax.text(X_axis[-1], 0.5, "Impossible\nto evaluate", ha="center")
    ax.bar(
        x=X_axis[-1] - 0.2,
        height=[1.0],
        width=barwidth,
        color="#eeeeee",
        error_kw=dict(lw=3, color="#0c0c0c"),
    )
    ax.bar(
        x=X_axis[-1] + 0.2,
        height=[1.0],
        width=barwidth,
        color="#eeeeee",
        error_kw=dict(lw=3, color="#0c0c0c"),
    )

    ax.margins(x=0.005)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(X_axis, ["Train", "Altered Semantics", "Altered Size"], fontsize=34)
    ax.set_ylabel("Success Rate", fontsize=34)
    ax.tick_params(left=True, bottom=False, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        fontsize=28,
        ncol=2,
    )
    fig.savefig(
        os.path.join(folder, "bad_ad.pdf"),
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )


def mse_regret_curves(
    api: wandb.Api,
    normal_sweep: str,
    ablated_sweep: str,
    colors: List[str],
    folder: str,
):
    # Load data from sweeps
    normal_regrets = load_data(api=api, sweep=normal_sweep)
    ablated_regrets = load_data(api=api, sweep=ablated_sweep)

    regrets = (*normal_regrets, ablated_regrets[-1])

    # Draw a plot
    name_mapper = {
        "train": "Odd\n(seen)",
        "inverse": "Even\n(unseen)",
        "all_new": "Uniform\n(unseen)",
        "num_5": "5 Arms\n(seen)",
        "num_10": "10 Arms\n(seen)",
        "num_15": "15 Arms\n(seen)",
        "num_20": "20 Arms\n(seen)",
        "num_25": "25 Arms\n(unseen)",
        "num_30": "30 Arms\n(unseen)",
        "num_40": "40 Arms\n(unseen)",
        "num_50": "50 Arms\n(unseen)",
        "num_100": "100 Arms\n(unseen)",
    }

    regret_curve_together_plot(
        regrets,
        arm_keys=[
            "train",
            "inverse",
            "all_new",
            "num_25",
            "num_30",
        ],
        name_mapper=name_mapper,
        algo_names=["Random", "Thompson", "Headless-AD (ours)", "MSE-Headless-AD"],
        colors=colors,
        filename=os.path.join(folder, "bandit_vanilla_mse.pdf"),
    )


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.family": "serif",
            "font.serif": "Arial",
            "font.weight": "regular",
            "font.size": 28,
            "legend.fontsize": 38,
        }
    )
    folder = os.path.join("reports", "plots")
    os.makedirs(folder, exist_ok=True)

    colors = ["#f9e8d9", "#ee7214", "#527853", "#f7b787"]
    api = wandb.Api()

    ### CORRECT SEEDS
    # bandit_headless = "tlab/Headless-AD/0w3xgbfh"
    # bandit_ad = "tlab/Headless-AD/q6fusdoj"
    # gridworld_headless = "tlab/Headless-AD/1l5nncul"
    # gridworld_ad = "tlab/Headless-AD/zfy0whwl"
    # contextual_headless = "tlab/Headless-AD/2kyaedh8"
    # contextual_ad = "tlab/Headless-AD/zrai04wx"

    # bandit_prompt = "tlab/Headless-AD/707r51yu"
    # gridworld_prompt = "tlab/Headless-AD/ayp94gag"

    # bandit_loss = "tlab/Headless-AD/g61rb7wv"
    # gridworld_loss = "tlab/Headless-AD/e0jpbe22"

    bandit_headless = "ummagumm-a/Headless-AD/6xdf8rqy"
    bandit_ad = "ummagumm-a/Headless-AD/zc7b6c30"
    gridworld_headless = "ummagumm-a/Headless-AD/e72q8mvu"
    gridworld_ad = "ummagumm-a/Headless-AD/vrtcv3yj"
    contextual_headless = "ummagumm-a/Headless-AD/18futcag"
    contextual_ad = "ummagumm-a/Headless-AD/2xsra5fq"

    bandit_prompt = "ummagumm-a/Headless-AD/r94ectqb"
    gridworld_prompt = "ummagumm-a/Headless-AD/th7nfwil"

    bandit_loss = "ummagumm-a/Headless-AD/b81oydwe"
    gridworld_loss = "ummagumm-a/Headless-AD/j78g2468"

    bandit_plot(
        api,
        headless_sweep=bandit_headless,
        ad_sweep=bandit_ad,
        colors=colors,
        folder=folder,
    )
    contextual_plot(
        api=api,
        headless_sweep=contextual_headless,
        ad_sweep=contextual_ad,
        colors=colors,
        folder=folder,
    )
    gridworld_plot(
        api,
        headless_sweep=gridworld_headless,
        ad_sweep=gridworld_ad,
        colors=colors,
        folder=folder,
    )

    gridworld_compare_ad_trainings(
        api=api, ad_sweep=gridworld_ad, colors=colors[1:], folder=folder
    )

    gridworld_actions_used_plot(
        api=api, sweep_name=gridworld_headless, colors=colors[1:], folder=folder
    )

    gridworld_ablation_plot(
        api=api,
        normal_sweep=gridworld_headless,
        ablated_sweep=gridworld_prompt,
        colors=colors[1:],
        folder=folder,
    )

    bandit_ablation_plot(
        api=api,
        normal_sweep=bandit_headless,
        ablated_sweep=bandit_prompt,
        colors=colors[1:],
        folder=folder,
    )

    # action pool ablation
    mean_prompt_ablation_plot(
        data=data_for_mean_ablation_plot(
            api=api,
            bandit_normal_sweep=bandit_headless,
            bandit_ablated_sweep=bandit_prompt,
            gridworld_normal_sweep=gridworld_headless,
            gridworld_ablated_sweep=gridworld_prompt,
        ),
        colors=colors[1:],
        folder=folder,
    )

    # mse ablation
    mean_loss_ablation_plot(
        data=data_for_mean_ablation_plot(
            api=api,
            bandit_normal_sweep=bandit_headless,
            bandit_ablated_sweep=bandit_loss,
            gridworld_normal_sweep=gridworld_headless,
            gridworld_ablated_sweep=gridworld_loss,
        ),
        colors=colors[1:],
        folder=folder,
    )

    bad_ad_plot(api=api, ad_sweep=gridworld_ad, colors=colors, folder=folder)

    mse_regret_curves(
        api=api,
        normal_sweep=bandit_headless,
        ablated_sweep=bandit_loss,
        colors=colors,
        folder=folder,
    )
