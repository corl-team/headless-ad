import gymnasium as gym
import numpy as np


# https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture10/lecture10.pdf
class ContextualBandit(gym.Env):
    def __init__(
        self,
        context_dim: int,
        arm_embeds: np.ndarray,
        num_arms: int,
    ):
        self.arm_embeds = arm_embeds
        self.num_arms = num_arms
        self.context_dim = context_dim

        self.action_space = gym.spaces.Discrete(len(arm_embeds))

        self.observation_space = gym.spaces.Box(
            low=-1e20, high=1e20, shape=(context_dim,), dtype=np.float32
        )
        self.rng = np.random.default_rng()

        self.regret = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.rng = np.random.default_rng(seed)
        self.context = self._get_new_context()
        # we also need to reset regret manually
        self.regret = 0

        return self.context, {}

    def _get_new_context(self):
        return self.rng.normal(size=(self.context_dim,)) / np.sqrt(self.context_dim)

    def step(self, action: int):
        assert action < self.num_arms, (action, self.num_arms)

        all_means = (self.arm_embeds @ self.context)[: self.num_arms]

        # calc reward
        mean = all_means[action]
        reward = self.rng.normal(loc=mean, scale=1)

        # info for calculation of the regret
        opt_mean = all_means.max()
        opt_act = all_means.argmax()

        self.regret += opt_mean - mean
        info = {"regret": self.regret, "opt_act": opt_act, "mean": mean}

        self.context = self._get_new_context()

        return self.context, reward, False, False, info
