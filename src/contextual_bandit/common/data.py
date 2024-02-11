import json
import os
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset


def load_learning_histories(runs_path: str) -> List[Dict[str, Any]]:
    learning_histories = []

    # There can be multiple runs of differnet algorithms (e.g. different seeds, training goals)
    for subdir, _, files in os.walk(runs_path):
        # Extract metadata for different learning histories
        for filename in files:
            if not filename.endswith(".metadata"):
                continue

            with open(os.path.join(subdir, filename)) as f:
                metadata = json.load(f)

            # Extract full learning history from chunks
            learning_history = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "num_actions": [],
            }
            for filename in metadata["ordered_trajectories"]:
                chunk = np.load(os.path.join(subdir, filename))
                learning_history["states"].append(chunk["states"])
                learning_history["actions"].append(chunk["actions"])
                learning_history["rewards"].append(chunk["rewards"])
                learning_history["dones"].append(chunk["dones"])
                learning_history["num_actions"].append(chunk["num_actions"])

            learning_history["states"] = np.vstack(learning_history["states"])
            learning_history["actions"] = np.vstack(learning_history["actions"])
            learning_history["rewards"] = np.vstack(learning_history["rewards"])
            learning_history["dones"] = np.vstack(learning_history["dones"])
            learning_history["num_actions"] = np.array(
                learning_history["num_actions"]
            ).squeeze(0)

            learning_histories.append(learning_history)

    return learning_histories


class SequenceDataset(Dataset):
    def __init__(self, runs_path: str, seq_len: int = 60):
        self.seq_len = seq_len

        print("Loading training histories...")
        histories = load_learning_histories(runs_path)

        self._states = np.stack([hist["states"] for hist in histories], axis=0)
        self._actions = np.hstack([hist["actions"] for hist in histories]).T
        self._rewards = np.hstack([hist["rewards"] for hist in histories]).T
        self._num_actions = np.hstack([hist["num_actions"] for hist in histories]).T

    def __prepare_sample(self, idx):
        num_hists = self._states.shape[0]
        i1, i2 = idx % num_hists, idx // num_hists

        states = self._states[i1, i2 : i2 + self.seq_len]
        actions = self._actions[i1, i2 : i2 + self.seq_len]
        rewards = self._rewards[i1, i2 : i2 + self.seq_len]
        num_actions = self._num_actions[i1]
        # Check that data does not contain samples
        # where algorithm uses more arms than is available in the environment
        assert np.all(actions < num_actions)

        return states, actions, rewards, num_actions

    def __len__(self):
        return self._rewards.shape[0] * (self._states.shape[1] - self.seq_len + 1)

    def __getitem__(self, idx):
        return self.__prepare_sample(idx)
