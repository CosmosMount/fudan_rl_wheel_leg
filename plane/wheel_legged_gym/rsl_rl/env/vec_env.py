from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Interface expected by the on-policy runner for vectorized environments."""

    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_observations(self):
        raise NotImplementedError

    @abstractmethod
    def get_privileged_observations(self):
        raise NotImplementedError
