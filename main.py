from environment.base import Base
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from environment.rewards.reward_custom import Main_Rewards
from agents.collect_agent import CollectAgent
from tqdm import tqdm
import numpy as np
import datetime
import math
import gc
import matplotlib.pyplot as plt


save_model = True
epochs = 100
steps = 2000
epsilon_min = 0.01
epsilon_max = 1


def plot_graph(reward, loss):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(100, 300))
    ax = fig.add_subplot(211)
    ax.plot(reward, color='blue')
    ax.set(title="Mean reward/episode",
           ylabel="Reward",
           xlabel="Epoch")

    bx = fig.add_subplot(212)
    bx.plot(loss, color='red')
    bx.set(title="Mean loss/episode",
           ylabel="Loss",
           xlabel="Epoch")
    plt.show()


def main():
    states = []
    n_blobs = 40

    reward_funct = Main_Rewards(fct_explore=1, fct_food=2, fct_home=10, fct_explore_holding=1, fct_headinghome=3)

    api = Base(reward=reward_funct, reward_threshold=1, max_speed=1, max_rot_speed=40 / 180 * np.pi,
                carry_speed_reduction=0.05, backward_speed_reduction=0.5)

    api.perceptive_field_save = True

    agent = CollectAgent(epsilon=0.9, dis=0.99, rotations=3, pheromones=3, lr=0.00001)
    agent_setup_once = True
    loss_avg = 0
    all_loss = []
    all_reward = []

    for epoch in range(epochs):

        generator = EnvironmentGenerator(w=200, h=200, n_blobs=n_blobs, n_pheromones=2, n_rocks=0,
                                         food_gen=CirclesGen(20, 5, 10),
                                         walls_generator=PerlinGen(scale=22.0, density=0.3),
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
            if loss_avg == 0:
                loss_avg = loss
            else:
                loss_avg = 0.99 * loss_avg + 0.01 * loss

            mean_reward = epoch_reward.mean(axis=0)
            # Set obs to the new state
            obs = new_state
            agent_state = new_agent_state
            env.update()

        all_loss.append(loss_avg)
        all_reward.append(mean_reward)
        agent.epsilon = max(epsilon_min,  min(epsilon_max, 1.0 - math.log10((epoch+1)/2)))
        gc.collect()

    if save_model:
        date = datetime.datetime.now()
        model_name = str(date.day) + '-' + str(date.month) + '-' + str(date.hour) + '.pt'
        agent.save_model(model_name)
        
    plot_graph(all_reward, all_loss)


if __name__ == '__main__':
    main()
