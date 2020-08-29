import random
import numpy as np
import noise


def png(w, h, offset_x, offset_y, scale=22.0, octaves=2, persistence=0.5, lacunarity=2.0):
    shape = (w, h)
    gen = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            gen[i][j] = noise.pnoise2((i + offset_x) / scale, (j + offset_y) / scale, octaves=octaves,
                                      persistence=persistence, lacunarity=lacunarity, base=0)
    return gen


class PerlinGen:
    def __init__(self, scale=22.0, density=0.05, octaves=2, persistence=0.5, lacunarity=2.0):
        self.scale = scale
        self.density = density
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def generate(self, w, h):
        return png(w, h, offset_x=random.randint(-10000, 10000), offset_y=random.randint(-10000, 10000),
                   scale=self.scale, octaves=self.octaves, persistence=self.persistence,
                   lacunarity=self.lacunarity) > self.density


class CirclesGen:
    def __init__(self, n_circles, min_radius, max_radius):
        self.n_circles = n_circles
        self.min_radius = min_radius
        self.max_radius = max_radius

    def generate(self, w, h):
        gen = np.zeros((w, h), dtype=bool)
        for i in range(self.n_circles):
            radius = int(random.random() * (self.max_radius - self.min_radius) + self.min_radius)
            xc = int(random.random() * (w - 2 * radius) + radius)
            yc = int(random.random() * (h - 2 * radius) + radius)

            for x in range(xc - radius, xc + radius + 1):
                for y in range(yc - radius, yc + radius + 1):
                    dist = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                    if dist <= radius:
                        gen[x, y] = True
        return gen
