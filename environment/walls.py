import numpy as np

from .environment import Environment, EnvObject
from .blobs import Blobs
from .pheromone import Pheromone

class Walls(EnvObject):
    def __init__(self, environment: Environment, map_in):
        super().__init__(environment)

        self.w = environment.w
        self.h = environment.h

        self.map = map_in.astype(bool)

    def visualize_copy(self, newenv):
        return self

    def update_step(self):
        return -1

    def update(self):
        for obj in self.environment.objects:
            if isinstance(obj, Blobs):
                xy = obj.xy.astype(int)
                colliding_blobs = self.map[xy[:, 0], xy[:, 1]]
                obj.blobs[colliding_blobs, 0:2] = obj.prev_blobs[colliding_blobs, 0:2]
                obj.blobs[colliding_blobs, 2] += np.random.random(np.sum(colliding_blobs)) - 0.5
            elif isinstance(obj, Pheromone):
                obj.phero[self.map] = 0
