import numpy as np
from numpy import ndarray
from typing import Tuple, Optional
from abc import ABC

from environment.base import Base

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.observation_space = None
        self.agent_space = None
        self.action_space = None
        self.n_blobs = 0

    def setup(self, base: Base, trained_model: Optional[str] = None):

        self.observation_space = (base.surrounding_coords.shape[0], base.surrounding_coords.shape[1], len(base.perceived_objects))
        self.agent_space = [2]
        self.action_space = [2]
        self.n_blobs = base.blobs.n_blobs

    def initialize(self, base: Base):
        pass

    def train(self, done: bool, step: int) -> Tuple[float, float]:
        return 0, 0

    def update_replay_memory(self, states: ndarray, agent_state : ndarray, actions: ndarray, rewards: ndarray,
                             new_states: ndarray, new_agent_state: ndarray, done: bool):
        pass

    def get_action(self, state: ndarray, agent_state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
        return None, None, None

    def save_model(self, file_name: str):
        pass

    def load_model(self, file_name: str):
        pass
