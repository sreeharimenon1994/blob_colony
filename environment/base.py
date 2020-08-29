import numpy as np
from numpy import ndarray
from typing import List, Optional
from environment.environment import Environment, EnvObject
from environment.pheromone import Pheromone
from environment.food import Food
from environment.blobs import Blobs
from environment.walls import Walls
from environment.circle_obstacles import CircleObstacles
from environment.anthill import Anthill
from environment.rewards.reward import Reward

axis = np.newaxis
DELTA = 1.1

class RLVisualization(EnvObject):
    def __init__(self, env: Environment, heatmap):
        super().__init__(env)
        self.heatmap = heatmap


class Base(EnvObject):
    def __init__(self, reward: Reward, reward_threshold: float, max_speed: float, max_rot_speed: float, carry_speed_reduction: float, backward_speed_reduction: float):
        """ Initializes an RL API. Call register_blobs to register this API to a group of blobs and its environment.
        :param max_speed: The maximum forward and backward speed at which blobs can move.
        :param max_rot_speed: The maximum number of radians blobs can turn at each step.
        :param carry_speed_reduction: How much one unit of carried food reduces the max speed (cumulative factor).
        :param backward_speed_reduction: How much moving backward reduces the max speed (factor).
        """
        super().__init__(None)
        self.reward = reward
        self.reward_threshold = reward_threshold

        self.blobs = None
        self.original_blobs_position = None

        self.surrounding_radius = 0
        self.surrounding_mask = None
        self.perceived_objects: List[EnvObject] = []
        self.surrounding_coords = None
        self.surrounding_fwd_delta = 0

        self.max_speed = max_speed
        self.max_rot_speed = max_rot_speed
        self.carry_speed_reduction = carry_speed_reduction
        self.backward_speed_reduction = backward_speed_reduction

        # If set to True, will save the perceptive field of each ant as an image to display over environment during visualization.
        self.save_perceptive_field = False
        self.perceptive_field = None


    def visualize_copy(self, newenv: Environment):
        return RLVisualization(newenv, self.reward.visualization())


    def register_blobs(self, new_blobs: Blobs):
        if self.environment is not None:
            self.environment.detach_object(self)
        self.blobs = new_blobs
        self.environment = new_blobs.environment
        self.environment.add_object(self)
        self.perceived_objects = []
        self.original_blobs_position = new_blobs.xy

        self.reward.setup(self.blobs)


    def setup_surrounding(self, radius: int, objects: List[EnvObject], mask=None, forward_delta=0):
        """Setups surrounding parameters for the group of blobs.
        :param radius: Number of grid units the ant can see around itself.
        :param objects: List of environment objects the ant can perceive (= surrounding channels).
        :param mask: Square boolean matrix of side 2*radius+1, to mask certain grid units around the ant.
        :param forward_delta: How much should the perceptive field be shifted in front of the ant."""
        self.surrounding_radius = radius
        self.surrounding_mask = mask
        self.perceived_objects = objects
        self.surrounding_fwd_delta = forward_delta

        # Constructing relative grid coordinates of perceived slots in grid (2*radius+1, 2*radius+1, 2)
        self.surrounding_coords = np.dstack([np.arange(-radius, radius+1)[axis, :].repeat(2*radius+1, 0), np.arange(-radius, radius+1)[:, axis].repeat(2*radius+1, 1)]).astype(float)
        self.surrounding_coords *= DELTA


    def observation(self):
        """Performs an observation on the environment, by each individual ant, looking in front of itself.
        :return (n_blobs, 2*radius+1, 2*radius+1, n_objects) numpy array, with -1 where the ant can't see because of the mask.
        :return (n_blobs, 2 + n_phero) numpy array describing the state of the ant, with [mandibles' state, held food, pheromone activation 1, phero act 2...]."""
        xy_f = self.blobs.xy.copy()
        t_f = self.blobs.theta + np.pi * 0.5

        if self.surrounding_fwd_delta != 0:
            xy_f += np.array([np.cos(self.blobs.theta), np.sin(self.blobs.theta)]).T * self.surrounding_fwd_delta

        # Rotating the surrounding grid based on each individual ant's theta orientation
        cos_t = np.cos(t_f)
        sin_t = np.sin(t_f)
        relative_pos = self.surrounding_coords[axis, :, :].repeat(len(self.blobs.blobs), 0)
        relative_pos[:, :, :, 0], relative_pos[:, :, :, 1] = cos_t[:, axis, axis] * relative_pos[:, :, :, 0] - sin_t[:, axis, axis] * relative_pos[:, :, :, 1], \
                                                                   sin_t[:, axis, axis] * relative_pos[:, :, :, 0] + cos_t[:, axis, axis] * relative_pos[:, :, :, 1]

        # Adding individual ant's position to relative coordinates
        abs_pos = relative_pos + xy_f[:, axis, axis, :]

        # Rounding to integer grid coordinates and warping to other side of the map if too big/too small
        abs_pos = np.round(abs_pos).astype(int)
        abs_pos[:, :, :, 0] = np.mod(abs_pos[:, :, :, 0], self.environment.w)
        abs_pos[:, :, :, 1] = np.mod(abs_pos[:, :, :, 1], self.environment.h)

        surrounding = np.zeros(list(abs_pos.shape[:-1]) + [len(self.perceived_objects)])
        # print(self.perceived_objects)
        for i, obj in enumerate(self.perceived_objects):
            if isinstance(obj, Pheromone):
                surrounding[:, :, :, i] = obj.phero[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]] / obj.max_val
            elif isinstance(obj, Food):
                surrounding[:, :, :, i] = obj.qte[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]]
            elif isinstance(obj, Walls):
                surrounding[:, :, :, i] = obj.map[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]]
            elif isinstance(obj, Anthill):
                surrounding[:, :, :, i] = obj.area[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]]
            elif isinstance(obj, CircleObstacles):
                vecs = abs_pos[:, :, :, axis, :] - obj.centers[axis, axis, axis, :, :]
                dists = np.sum(vecs**2, axis=4)**0.5
                surrounding[:, :, :, i] = np.max(dists < obj.radiuses, axis=3)
            elif isinstance(obj, Blobs):
                blobs_xy = np.round(obj.xy.astype(int))
                blobs_xy[:, 0] = np.mod(blobs_xy[:, 0], self.environment.w)
                blobs_xy[:, 1] = np.mod(blobs_xy[:, 1], self.environment.h)
                blobs_map = np.zeros((self.environment.w, self.environment.h), dtype=int)
                blobs_map[blobs_xy[:, 0], blobs_xy[:, 1]] += 1
                surrounding[:, :, :, i] = blobs_map[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]]

        if self.save_perceptive_field:
            self.perceptive_field = np.zeros((self.environment.w, self.environment.h), dtype=bool)

        if self.surrounding_mask is not None:
            surrounding = self.surrounding_mask[axis, :, :, axis] * (surrounding + 1) - 1

            if self.save_perceptive_field:
                self.perceptive_field[abs_pos[:, self.surrounding_mask, 0], abs_pos[:, self.surrounding_mask, 1]] = True
        elif self.save_perceptive_field:
            self.perceptive_field[abs_pos[:, :, :, 0], abs_pos[:, :, :, 1]] = True

        state = np.zeros((len(self.blobs.blobs), 2 + len(self.blobs.pheromones)))
        state[:, 0] = self.blobs.mandibles
        state[:, 1] = self.blobs.holding
        state[:, 2:] = self.blobs.phero_activation > 0

        agent_state = np.zeros((self.blobs.n_blobs, 2))
        agent_state[:, 0] = self.blobs.holding
        agent_state[:, 1] = self.blobs.seed

        self.reward.observation(abs_pos, surrounding, agent_state)
        return surrounding, agent_state, state


    def step(self, rotation: Optional[ndarray], on_off_pheromones: Optional[ndarray]):
        """ Applies the different ant actions to the ant group. A None action won't change the state of blobs.
        :param rotation: How much the ant should turn right, will be multiplied by max_rot_speed
        :param open_close_mandibles: Are the mandibles opened or closed
        :param on_off_pheromones: Are the pheromones activated or not
        """
        open_close_mandibles = None
        #if open_close_mandibles is not None:
            #self.blobs.update_mandibles(open_close_mandibles)

        xy = self.blobs.prev_blobs[:, 0:2].astype(int)
        open_close_mandibles = self.blobs.mandibles.copy()
        for i, obj in enumerate(self.perceived_objects):
            if isinstance(obj, Food):
                open_close_mandibles = np.bitwise_or(obj.qte[xy[:, 0], xy[:, 1]] > 0, open_close_mandibles)
            elif isinstance(obj, Anthill):
                open_close_mandibles = np.bitwise_and(1 - obj.area[self.blobs.x.astype(int), self.blobs.y.astype(int)], open_close_mandibles)
        self.blobs.update_mandibles(open_close_mandibles)

        if on_off_pheromones is not None:
            self.blobs.activate_pheromone(on_off_pheromones)

        if rotation is not None:
            self.blobs.rotate_blobs(rotation * self.max_rot_speed)

        # Moves the blobs forward
        fwd = np.ones(self.blobs.n_blobs) * self.max_speed * (1 - self.blobs.holding * self.carry_speed_reduction)
        fwd[fwd < 0] *= self.backward_speed_reduction
        self.blobs.forward_blobs(fwd)

        surrounding, agent_state, state = self.observation()

        done = self.environment.max_time == self.environment.timestep

        reward = self.reward.step(done, rotation, open_close_mandibles, on_off_pheromones)
        self.blobs.give_reward(reward - self.reward_threshold)
        return surrounding, agent_state, reward, done
