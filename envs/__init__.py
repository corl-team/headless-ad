from functools import partial

from gymnasium.envs.registration import register

from envs.bandits import MultiArmedBanditBernoulli
from envs.contextual_bandits import ContextualBandit
from envs.gridworld import GridWorld

register(id="MultiArmedBanditBernoulli", entry_point=MultiArmedBanditBernoulli)

register(
    id="GridWorld",
    entry_point=partial(GridWorld, terminate_on_goal=True),
    max_episode_steps=20,
)

register(id="ContextualBandit", entry_point=ContextualBandit)
