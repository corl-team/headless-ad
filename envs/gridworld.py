import warnings
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

from src.utils.misc import get_action_sequences

# gym warnings are annoying
warnings.filterwarnings("ignore")


@njit()
def pos_to_state(pos: Tuple[int, int], size: int):
    return int(pos[0] * size + pos[1])


@njit()
def single_step(
    action: int,
    agent_pos: np.ndarray,
    action_to_direction: np.ndarray,
    size: int,
    goal_pos: np.ndarray,
    terminate_on_goal: bool,
) -> Tuple[np.ndarray, Tuple[int, float, bool, bool]]:
    """
    This function makes an atomic step in the environment. Returns a new agent position and
    a usual tuple from a gym environment.

    :param action: index of an atomic action.
    :param agent_pos: the current position of an agent.
    :param action_to_direction: a list of transitions corresponding to each atomic action.
    :param size: the size of the grid.
    :param goal_pos: the goal's coordinates.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    agent_pos = np.clip(agent_pos + action_to_direction[action], 0, size - 1)

    reward = 1.0 if np.array_equal(agent_pos, goal_pos) else 0.0
    terminated = True if reward and terminate_on_goal else False

    gym_output = pos_to_state(agent_pos, size), reward, terminated, False

    return agent_pos, gym_output


@njit()
def multi_step(
    action: int,
    action_sequences: np.ndarray,
    agent_pos: np.ndarray,
    action_to_direction: np.ndarray,
    size: int,
    goal_pos: np.ndarray,
    terminate_on_goal: bool,
) -> Tuple[np.ndarray, Tuple[int, float, bool, bool]]:
    """
    This function makes an sequential step in the environment. Returns a new agent position and
    a usual tuple from a gym environment.

    :param action: index of a sequential action.
    :param action_sequences: for each sequential action specifies the sequence of atomic actions' indices.
    :param agent_pos: the current position of an agent.
    :param action_to_direction: a list of transitions corresponding to each atomic action.
    :param size: the size of the grid.
    :param goal_pos: the goal's coordinates.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    # Choose a sequence of atomic actions
    action_seq = action_sequences[action]

    # Perf each atomic action one after another
    rewards = np.zeros(len(action_seq))
    terms = np.zeros(len(action_seq))
    for i, act in enumerate(action_seq):
        agent_pos, gym_output = single_step(
            act, agent_pos, action_to_direction, size, goal_pos, terminate_on_goal
        )
        obs, rew, term, _ = gym_output
        rewards[i] = rew
        terms[i] = term

    # The reward will equal to 1 if the sequence's trajectory has passed
    # through a goal cell
    reward = int(np.any(rewards == 1))
    # The episode is finished if the sequence's trajectory has passed
    # through a goal cell
    term = np.any(terms)

    gym_output = obs, reward, term, False

    return agent_pos, gym_output


class GridWorld(gym.Env):
    """
    This is a darkroom environment where an agent operates in a grid and must reach a goal cell.
    A single action is a sequence of atomic actions 'noop', 'up', 'down', 'left' and 'right'.

    :param available_actions: indices of action sequences that the environment will use.
    :param action_seq_len: the amount of atomic actions constituting the action sequence.
    :param size: the size of the grid.
    :param goal_pos: the goal position. If None, will be chosen randomly.
    :param render_mode: same as in openai gym.
    :param terminate_on_goal: whether the episode ends upon reaching the goal.
    """

    def __init__(
        self,
        available_actions: np.ndarray,
        action_seq_len: int = 1,
        size: int = 9,
        goal_pos: Optional[np.ndarray] = None,
        render_mode=None,
        terminate_on_goal: bool = False,
    ):
        self.action_seq_len = action_seq_len
        # 5 is amount of atomic actions
        self.action_sequences = get_action_sequences(5, self.action_seq_len)
        self.action_sequences = self.action_sequences[available_actions]

        self.size = size
        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(len(available_actions))

        self.action_to_direction = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        # the agent will start here
        self.center_pos = (self.size // 2, self.size // 2)

        # set the goal cell
        if goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal_pos()

        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode

    def generate_goal_pos(self):
        """
        Generates random coordinates for the goal.
        """
        return self.np_random.integers(0, self.size, size=2)

    def state_to_pos(self, state):
        """
        Converts an index of a cell into 2-component coordinates
        """
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = np.array(self.center_pos, dtype=np.float32)

        return pos_to_state(self.agent_pos, self.size), {}

    def _single_step(self, action):
        """
        An atomic step in the environment.

        :param action: index of atomic action.
        """
        self.agent_pos, gym_output = single_step(
            action,
            agent_pos=self.agent_pos,
            action_to_direction=self.action_to_direction,
            size=self.size,
            goal_pos=self.goal_pos,
            terminate_on_goal=self.terminate_on_goal,
        )

        return gym_output + ({},)

    def step(self, action):
        """
        A 'sequential' step in an environment.

        :param action: index of a sequential action.
        """
        self.agent_pos, gym_output = multi_step(
            action,
            action_sequences=self.action_sequences,
            agent_pos=self.agent_pos,
            action_to_direction=self.action_to_direction,
            size=self.size,
            goal_pos=self.goal_pos,
            terminate_on_goal=self.terminate_on_goal,
        )

        return gym_output + ({},)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full(
                (self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8
            )
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid
