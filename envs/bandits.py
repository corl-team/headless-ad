import gymnasium as gym
import numpy as np


class MultiArmedBanditBernoulli(gym.Env):
    def __init__(self, arms_mean: np.ndarray, num_arms: int):
        self.arms_mean = arms_mean
        self.num_arms = num_arms

        self.action_space = gym.spaces.Discrete(len(arms_mean))
        # the only obs is 0
        self.observation_space = gym.spaces.Discrete(1)
        self.rng = np.random.default_rng()

        self.regret = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.rng = np.random.default_rng(seed)
        # we also need to reset regret manually
        self.regret = 0

        return 0, {}

    def step(self, action: int):
        assert action < self.num_arms, (action, self.num_arms)

        # calc reward
        mean = self.arms_mean[action]
        reward = self.rng.binomial(n=1, p=mean)

        # info for calculation of the regret
        opt_mean = self.arms_mean[: self.num_arms].max()
        opt_act = self.arms_mean[: self.num_arms].argmax()

        self.regret += opt_mean - mean
        info = {"regret": self.regret, "opt_act": opt_act}

        return 0, reward, False, False, info
