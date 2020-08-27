# import bpy
# b = bpy.data.collections['blob'].all_objects.items()
# for i, x in enumerate(b):
#     tmp = bpy.data.objects[x[0]].location
#     tmp[0] = i



import numpy as np
from tqdm import tqdm
from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from environment.rewards.reward_custom import *
from agents.collect_agent_memory import CollectAgentMemory



class AniPointCollect(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, ):
        self.stream = self.data_collection()

    def data_collection(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        model_path = '25-8-7.pt'
        # model_path = '23-8-9.pt'

        n_blobs = 15

        reward_funct = All_Rewards(fct_explore=1, fct_food=2, fct_anthill=10, fct_explore_holding=1, fct_headinganthill=3)

        api = RLApi(reward=reward_funct, reward_threshold=1, max_speed=1, max_rot_speed=40 / 180 * np.pi,
                    carry_speed_reduction=0.05, backward_speed_reduction=0.5)

        api.save_perceptive_field = False

        agent = CollectAgentMemory(epsilon=0.01, discount=0.99, rotations=3, pheromones=3, learning_rate=0.00001)

        generator = EnvironmentGenerator(w=200, h=200, n_blobs=n_blobs, n_pheromones=2, n_rocks=0,
                                         food_generator=CirclesGenerator(20, 5, 10),
                                         walls_generator=PerlinGenerator(scale=22.0, density=0.3),
                                         max_steps=2000, seed=None)

        env = generator.generate(api)
        agent.setup(api, model_path)
        agent.initialize(api)

        obs, agent_state, state = api.observation()

        epoch = 3000

        final = []
        wall = np.dstack(np.where(api.perceived_objects[4].map > 0))[0] # wall
        hill = np.dstack(np.where(api.perceived_objects[3].area > 0))[0] # hill

        for itr in tqdm(range(epoch)):
            t = []
            action = agent.get_action(obs, agent_state, True)
            new_state, new_agent_state, reward, done = api.step(*action[:2])
            obs = new_state
            agent_state = new_agent_state
            env.update()
          
            xy = np.dstack(np.where(api.perceived_objects[5].qte > 0))[0] # food
            t.append(xy)
            
            xy = np.dstack(np.where(api.perceived_objects[1].phero > 0))[0] # phero
            t.append(xy)
            
            xy = np.dstack(np.where(api.perceived_objects[2].phero > 0))[0] # phero
            t.append(xy)
            
            xy = api.perceived_objects[0].xy.copy()
            t.append(xy)

            final.append(t)

        final = np.array(final)
        return final, wall, hill

obj = AniPointCollect()
data, wall, hill = obj.stream

np.save('visualise_itr/data.npy', data)
np.save('visualise_itr/hill.npy', hill)
np.save('visualise_itr/wall.npy', wall)
