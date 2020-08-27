import numpy as np
from .environment import Environment, EnvObject
from .blobs import Blobs
from typing import List, Tuple
axis = np.newaxis


class CircleObstaclesVisualization (EnvObject):
    def __init__(self, env, centers, rad, weights):
        super().__init__(env)
        self.centers = centers.copy()
        self.rad = rad.copy()
        self.weights = weights.copy()


class CircleObstacles (EnvObject):
    def __init__(self, environment: Environment, centers, rad, weights):
        super().__init__(environment)
        self.w = environment.w
        self.h = environment.h

        self.n_obst = len(rad)
        self.centers = centers
        self.rad = rad
        self.weights = weights

        self.crossed_rad = self.rad[axis, :] + self.rad[:, axis]
        self.crossed_weights = self.weights[axis, :] / (self.weights[:, axis] + self.weights[axis, :])

    def visualize_copy(self, newenv):
        return CircleObstaclesVisualization(newenv, self.centers, self.rad, self.weights)

    def update(self):
        for obj in self.environment.objects:
            if isinstance(obj, Blobs):
                vecs_from_centers = self.centers[axis, :, :] - obj.xy[:, axis, :]
                dist_to_centers = np.sum(vecs_from_centers**2, axis=2)**0.5
                vecs_from_rad = vecs_from_centers * (1 - self.rad[axis, :] / (dist_to_centers + 0.001))[:, :, axis]
                vecs_from_rad[dist_to_centers > self.rad[axis, :], :] = 0

                self.centers -= np.sum(vecs_from_rad, axis=0) / self.weights[:, axis]

        for obj in self.environment.objects:
            if isinstance(obj, Blobs):
                vecs_from_centers = self.centers[axis, :, :] - obj.xy[:, axis, :]
                dist_to_centers = np.sum(vecs_from_centers ** 2, axis=2) ** 0.5
                vecs_from_rad = vecs_from_centers * (1 - self.rad[axis, :] / (dist_to_centers + 0.001))[
                                                         :, :, axis]
                vecs_from_rad[dist_to_centers > self.rad[axis, :], :] = 0
                obj.translate_blobs(np.sum(vecs_from_rad, axis=1))

    def update_step(self):
        return 0
