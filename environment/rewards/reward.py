from abc import ABC
import numpy as np

from environment.blobs import Blobs

class Reward (ABC):
	def __init__(self):
		self.blobs = None
		self.environment = None
		self.rewards = None

	def setup(self, blobs: Blobs):
		self.blobs = blobs
		self.environment = blobs.environment
		self.rewards = np.zeros(self.blobs.n_blobs, dtype=float)

	def observation(self, obs_coords, surrounding, agent_state):
		pass

	def step(self, done, turn_index, open_close_mandibles, on_off_pheromones):
		return self.rewards

	def visualization(self):
		return None
