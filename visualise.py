import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from environment.base import Base
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from environment.rewards.reward_custom import Main_Rewards
from agents.collect_agent import CollectAgent
import gc
import time

delay = 0.05


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, ):
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1)
        self.ax.axis([0, 200, 0, 200])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        model_path = '28-8-2.pt'

        n_blobs = 15

        reward_funct = Main_Rewards(fct_explore=1, fct_food=2, fct_anthill=10, fct_explore_holding=1, fct_headinganthill=3)

        api = Base(reward=reward_funct, reward_threshold=1, max_speed=1, max_rot_speed=40 / 180 * np.pi,
                    carry_speed_reduction=0.05, backward_speed_reduction=0.5)

        api.save_perceptive_field = False

        agent = CollectAgent(epsilon=0.01, discount=0.99, rotations=3, pheromones=3, learning_rate=0.00001)

        generator = EnvironmentGenerator(w=200, h=200, n_blobs=n_blobs, n_pheromones=2, n_rocks=0,
                                         food_generator=CirclesGenerator(20, 5, 10),
                                         walls_generator=PerlinGenerator(scale=22.0, density=0.3),
                                         max_steps=2000, seed=None)

        env = generator.generate(api)
        agent.setup(api, model_path)
        agent.initialize(api)

        obs, agent_state, state = api.observation()

        while True:
            action = agent.get_action(obs, agent_state, True)
            new_state, new_agent_state, reward, done = api.step(*action[:2])
            obs = new_state
            agent_state = new_agent_state
            env.update()

            xy = np.dstack(np.where(api.perceived_objects[4].map > 0))[0] # wall
            s = np.ones(xy.shape[0]) * 100.0
            c = np.ones(xy.shape[0]) * .01

            tmp = np.dstack(np.where(api.perceived_objects[3].area > 0))[0] # hill
            # print('tmp', tmp.shape, 'xy', xy.tmp)
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
            c = np.append(c, np.ones(tmp.shape[0]) * .2, axis=0)

            tmp = np.dstack(np.where(api.perceived_objects[5].qte > 0))[0] # food
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
            c = np.append(c, np.ones(tmp.shape[0]) * .8, axis=0)

            tmp = np.dstack(np.where(api.perceived_objects[1].phero > 0))[0] # phero
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 10.0, axis=0)
            c = np.append(c, np.ones(tmp.shape[0]) * .3, axis=0)            

            tmp = np.dstack(np.where(api.perceived_objects[2].phero > 0))[0] # phero
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 10.0, axis=0)
            c = np.append(c, np.ones(tmp.shape[0]) * .5, axis=0)

            tmp = api.perceived_objects[0].xy # blobs
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
            c = np.append(c, np.ones(tmp.shape[0]) * .99, axis=0)

            # print(xy.shape, s.shape, c.shape)
            # time.sleep(.01)
            gc.collect() 

            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        # print(data.shape)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(abs(data[:, 2]))
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,



a = AnimatedScatter()
plt.show()