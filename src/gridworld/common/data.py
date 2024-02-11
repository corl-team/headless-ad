import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


def load_single_history(inp) -> Optional[Dict[str, np.ndarray]]:
    subdir, _unused, files = inp
    if len(files) == 0:
        return

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
        }
        for filename in metadata["ordered_trajectories"]:
            chunk = np.load(os.path.join(subdir, filename))
            learning_history["states"].append(chunk["states"])
            learning_history["actions"].append(chunk["actions"])
            learning_history["rewards"].append(chunk["rewards"])
            learning_history["dones"].append(chunk["dones"])

        learning_history["states"] = np.vstack(learning_history["states"])
        learning_history["actions"] = np.vstack(learning_history["actions"])
        learning_history["rewards"] = np.vstack(learning_history["rewards"])
        learning_history["dones"] = np.vstack(learning_history["dones"])

    return learning_history


def load_learning_histories(runs_path: str) -> List[Dict[str, Any]]:
    with mp.Pool(processes=os.cpu_count()) as pool:
        learning_histories = pool.map(load_single_history, os.walk(runs_path))

    learning_histories = [hist for hist in learning_histories if hist is not None]

    return learning_histories


class SequenceDataset(Dataset):
    def __init__(self, runs_path: str, seq_len: int = 60, subsample: int = 1):
        self.seq_len = seq_len

        print("Loading training histories...")
        histories = load_learning_histories(runs_path)

        self._states = [hist["states"].squeeze(-1) for hist in histories]
        self._actions = [hist["actions"].squeeze(-1) for hist in histories]
        self._rewards = [hist["rewards"].squeeze(-1) for hist in histories]

        self._prefix_sum = self._build_prefix_sum(self._actions)
        min_len = np.min([len(x) for x in self._states])

        assert self.seq_len < min_len, (self.seq_len, min_len)

    def _build_prefix_sum(self, lists):
        cumulative_lengths = []
        total_length = 0
        for lst in lists:
            assert len(lst) >= self.seq_len
            total_length += len(lst) - self.seq_len
            cumulative_lengths.append(total_length)

        return np.array(cumulative_lengths)

    def _convert_index(self, index):
        # Binary search to find the correct list
        low, high = 0, len(self._prefix_sum)
        while low < high:
            mid = (low + high) // 2
            if self._prefix_sum[mid] <= index:
                low = mid + 1
            else:
                high = mid

        # Adjust the index to the local index in the found list
        list_index = low
        local_index = index - (
            self._prefix_sum[list_index - 1] if list_index > 0 else 0
        )

        return list_index, local_index

    def __prepare_sample(self, idx):
        i1, i2 = self._convert_index(idx)
        assert i2 < len(self._states[i1]) and i2 + self.seq_len <= len(
            self._states[i1]
        ), (i1, i2, i2 + self.seq_len, len(self._states[i1]))

        states = self._states[i1][i2 : i2 + self.seq_len]
        actions = self._actions[i1][i2 : i2 + self.seq_len]
        rewards = self._rewards[i1][i2 : i2 + self.seq_len]

        lens = np.array([len(states), len(actions), len(rewards)])
        assert np.all(lens == self.seq_len), (lens, self.seq_len)

        return states, actions, rewards

    def __len__(self):
        return self._prefix_sum[-1]

    def __getitem__(self, idx):
        return self.__prepare_sample(idx)
