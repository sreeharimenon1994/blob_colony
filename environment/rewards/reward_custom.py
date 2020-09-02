import numpy as np

from environment.blobs import Blobs
from environment.home import Home
from environment.rewards.reward import Reward


class ExplorationReward(Reward):
    def __init__(self, ):
        super(ExplorationReward, self).__init__()
        self.explored_map = None

    def setup(self, blobs: Blobs):
        super(ExplorationReward, self).setup(blobs)
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)

    def observation(self, obs_coords, surrounding, agent_state):
    
        self.rewards = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]], axis=(1, 2)) / 10
        self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True

    def visualization(self):
        return self.explored_map.copy()


class Food_Reward(Reward):
    def __init__(self, ):
        super(Food_Reward, self).__init__()

    def setup(self, blobs: Blobs):
        super(Food_Reward, self).setup(blobs)
        self.rewards = self.blobs.holding
        self.blobs_holding = self.blobs.holding

    def observation(self, obs_coords, surrounding, agent_state):
        self.rewards = agent_state[:, 0] - self.blobs_holding
        self.blobs_holding = agent_state[:, 0]


class Main_Rewards(Reward):
    def __init__(self, fct_explore=1, fct_food=1, fct_home=5, fct_explore_holding=0, fct_headinghome=1):
        super(Main_Rewards, self).__init__()
        self.explored_map = None
        self.fct_explore = fct_explore
        self.fct_food = fct_food
        self.fct_home = fct_home
        self.fct_explore_holding = fct_explore_holding
        self.fct_headinghome = fct_headinghome

        self.previous_dist = None
        self.home_x = 0
        self.home_y = 0

        self.blobs_holding = None

    def compute_distance(self, x, y):
        return ((x - self.home_x) ** 2 + (y - self.home_y) ** 2) ** 0.5


    def setup(self, blobs: Blobs):
        super(Main_Rewards, self).setup(blobs)
        self.rewards = self.blobs.holding
        self.blobs_holding = self.blobs.holding
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)
        self.rewards_homeheading = np.zeros(blobs.n_blobs)
        self.blobs = blobs

        for obj in blobs.environment.objects:
            if isinstance(obj, Home):
                self.home_x = obj.x
                self.home_y = obj.y

        self.previous_dist = self.compute_distance(blobs.x, blobs.y)

    def observation(self, obs_coords, surrounding, agent_state):
        self.rewards = np.zeros_like(self.rewards)

        rewards_food = agent_state[:, 0] - self.blobs_holding
        rewards_home = rewards_food.copy()
        rewards_food[rewards_food < 0] = 0
        self.blobs_holding = agent_state[:, 0]

        if self.fct_explore != 0 or self.fct_explore_holding != 0:
            rewards_explore = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]],
                                          axis=(1, 2)) / 10
            rewards_explore = np.array([r * self.fct_explore if h == 0 else r * self.fct_explore_holding for r, h in zip(rewards_explore, self.blobs_holding)])
            self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True
            self.rewards += rewards_explore

        rewards_home[rewards_home > 0] = 0
        rewards_home[rewards_home < 0] = 1

        new_dist = self.compute_distance(self.blobs.x, self.blobs.y)
        self.rewards_homeheading = (self.previous_dist > new_dist) * (self.blobs.holding > 0) * 0.1
        self.previous_dist = new_dist

        self.rewards += rewards_food * self.fct_food + rewards_home * self.fct_home + self.rewards_homeheading * self.fct_headinghome

    def visualization(self):
        return self.explored_map.copy()
