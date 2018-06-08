# Solve CartPole using DQN
#
# https://gym.openai.com/envs/CartPole-v0/
#
# Some useful resources:
# * https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#

import time
import matplotlib.pyplot as plt
import gym
from qnetwork import QNetwork
from agent import DQNAgent, solve_env


def main():
    start_time = time.time()
    env = gym.make("CartPole-v0").env
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(network, env.action_space.n)
    rewards = solve_env(env, agent, max_sessions=2000, t_max=200, solved=190)
    end_time = time.time()
    print('Finished in {} seconds'.format(int(end_time-start_time)))
    plt.plot(rewards)
    plt.show()


main()
