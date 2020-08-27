from abc import ABC
import numpy as np

from environment.blobs import Blobs

class Reward (ABC):
	def __init__(self):
		self.blobs = None
		self.environment = None
		self.rewards = None

	def setup(self, blobs: Blobs):
		"""
		Setups the reward on a new blobs group
		:param blobs: the new ant group
		"""
		self.blobs = blobs
		self.environment = blobs.environment
		self.rewards = np.zeros(self.blobs.n_blobs, dtype=float)

	def observation(self, obs_coords, surrounding, agent_state):
		"""
		Called by the RL Api when performing an observation.
		:param obs_coords: the coordinates observed by each individual ant
		:param surrounding: the surrounding states of the blobs
		"""
		pass

	def step(self, done, turn_index, open_close_mandibles, on_off_pheromones):
		"""
		Computes the reward of each ant at a certain step.
		:param done: is this the end of the game?
		:param turn_index: turning action of all blobs
		:param open_close_mandibles: boolean state of blobs' mandibles
		:param on_off_pheromones: boolean state of blobs' pheromone activations
		:return: a numpy array of shape (n_blobs) containing the per-blobs reward
		"""
		return self.rewards

	def visualization(self):
		"""
		Returns a visualization heatmap that will be visible in the GUI. Returning None for no heatmap (saves disk space and performance).
		:return: None or a numpy array of shape (env.w, env.h)
		"""
		return None
