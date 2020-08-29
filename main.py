
from environment.base import Base
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from environment.rewards.reward_custom import Main_Rewards
from utils import plot_training
from agents.collect_agent import CollectAgent
from tqdm import tqdm
import datetime
import time
import math
import gc
save_model = True

epochs = 100
steps = 2000
epsilon_min = 0.01
epsilon_max = 1

def main():
    states = []
    n_blobs = 40

    reward_funct = Main_Rewards(fct_explore=1, fct_food=2, fct_anthill=10, fct_explore_holding=1, fct_headinganthill=3)

    api = Base(reward=reward_funct, reward_threshold=1, max_speed=1, max_rot_speed=40 / 180 * np.pi,
                carry_speed_reduction=0.05, backward_speed_reduction=0.5)

    api.save_perceptive_field = True

    agent = CollectAgent(epsilon=0.9, discount=0.99, rotations=3, pheromones=3, learning_rate=0.00001)

    agent_setup_once = True

    avg_loss = 0
    avg_time = None

    all_loss = []
    all_reward = []

    print("Simulating...")
    for epoch in range(epochs):

        generator = EnvironmentGenerator(w=200, h=200, n_blobs=n_blobs, n_pheromones=2, n_rocks=0,
                                         food_generator=CirclesGenerator(20, 5, 10),
                                         walls_generator=PerlinGenerator(scale=22.0, density=0.3),
                                         max_steps=steps, seed=None)

        env = generator.generate(api)
        print('\nEpoch: {}/{}'.format(epoch + 1, epochs))
        # Setups the agents only once
        if agent_setup_once:
            agent.setup(api)
            agent_setup_once = False
        # Initializes the agents on the new environment
        agent.initialize(api)

        obs, agent_state, state = api.observation()
        epoch_reward = np.zeros(n_blobs)
        mean_reward = 0

        for s in tqdm(range(steps)):

            action = agent.get_action(obs, agent_state, True)
            new_state, new_agent_state, reward, done = api.step(*action[:2])

            epoch_reward += reward
            agent.update_replay_memory(obs, agent_state, action, reward, new_state, new_agent_state, done)

            loss = agent.train(done, s)
            if avg_loss == 0:
                avg_loss = loss
            else:
                avg_loss = 0.99 * avg_loss + 0.01 * loss

            mean_reward = epoch_reward.mean(axis=0)
            # Set obs to the new state
            obs = new_state
            agent_state = new_agent_state

            env.update()
        

        gc.collect()
        agent.epsilon = max(epsilon_min,  min(epsilon_max, 1.0 - math.log10((epoch+1)/2)))
        all_loss.append(avg_loss)
        all_reward.append(mean_reward)

    if save_model:
        date = datetime.datetime.now()
        model_name = str(date.day) + '-' + str(date.month) + '-' + str(date.hour) + '.pt'
        agent.save_model(model_name)
        
    plot_training(all_reward, all_loss)




if __name__ == '__main__':
    main()

# visualiser = Visualizer()
# visualiser.big_dim = 900
# visualiser.visualize(save_file_name)