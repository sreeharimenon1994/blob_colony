import random
from utils import *

from environment.environment import Environment
from environment.anthill import Anthill
from environment.blobs import Blobs
from environment.pheromone import Pheromone
from environment.circle_obstacles import CircleObstacles
from environment.walls import Walls
from environment.food import Food
from environment.base import Base

PHERO_COLORS = [
    (255, 64, 0),
    (64, 64, 255),
    (100, 255, 100)
]

class EnvironmentGenerator:
    def __init__(self, w, h, n_blobs, n_pheromones, n_rocks, food_generator, walls_generator, max_steps, seed=None):
        self.w = w
        self.h = h
        self.n_blobs = n_blobs
        self.n_pheromones = n_pheromones
        self.n_rocks = n_rocks
        self.food_generator = food_generator
        self.walls_generator = walls_generator

        self.surrounding_mask = np.array([[0, 0, 1, 1, 1, 0, 0],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 1, 0, 0]], dtype=bool)

        self.surrounding_shift = 4

        self.max_steps = max_steps
        self.seed = seed

    def setup_surrounding(self, new_mask, new_shift):
        self.surrounding_mask = new_mask
        self.surrounding_shift = new_shift

    def generate(self, base: Base):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed * 5)

        env = Environment(self.w, self.h, self.max_steps)
        perceived_objects = []

        anthill = Anthill(env,
                          int(random.random() * self.w * 0.5 + self.w * 0.25),
                          int(random.random() * self.h * 0.5 + self.h * 0.25),
                          int(random.random() * min(self.w, self.h) * 0.05 + min(self.w, self.h) * 0.05))
        perceived_objects.append(anthill)

        world_walls = self.walls_generator.generate(self.w, self.h)
        world_walls[anthill.area] = False
        walls = Walls(env, world_walls)
        perceived_objects.append(walls)

        food = Food(env, self.food_generator.generate(self.w, self.h))
        food.qte *= (1 - walls.map)
        perceived_objects.append(food)


        if self.n_rocks > 0:
            rock_centers = np.random.random((self.n_rocks, 2))
            rock_centers[:, 0] *= self.w * 0.75
            rock_centers[:, 1] *= self.h * 0.25
            rock_centers[:, 0] += self.w * 0.25
            rock_centers[:, 1] += self.h * 0.25
            rocks = CircleObstacles(env, centers=rock_centers,
                                    radiuses=np.random.random(n_rocks) * 5 + 5,
                                    weights=np.random.random(n_rocks) * 50 + 50)
            perceived_objects.append(rocks)

        blobs_angle = np.random.random(self.n_blobs) * 2 * np.pi
        blobs_dist = np.random.random(self.n_blobs) * anthill.radius * 0.8
        blobs_x = np.cos(blobs_angle) * blobs_dist + anthill.x
        blobs_y = np.sin(blobs_angle) * blobs_dist + anthill.y
        blobs_t = np.random.random(self.n_blobs) * 2 * np.pi

        blobs = Blobs(env, self.n_blobs, 5, xyt=np.array([blobs_x, blobs_y, blobs_t]).T)
        perceived_objects.insert(0, blobs)

        for p in range(self.n_pheromones):
            phero = Pheromone(env, color=PHERO_COLORS[p % len(PHERO_COLORS)], max_val=255)
            blobs.register_pheromone(phero)
            perceived_objects.insert(p + 1, phero)

        base.register_blobs(blobs)
        base.setup_surrounding(self.surrounding_mask.shape[0] // 2,
                                perceived_objects,
                                self.surrounding_mask,
                                self.surrounding_shift)
        return env
