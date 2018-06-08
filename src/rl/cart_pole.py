# Solved CartPole using DQN (Policy Gradient)
#
# This is minimal introduction to the DQN learning.
# The agent should solve the environment in ~1K episodes.
# But still it is not perfect. Sometimes it fails.
#
# https://gym.openai.com/envs/CartPole-v0/
#
# Some useful resources:
# * https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#

import time
import matplotlib.pyplot as plt
import gym
from dqn.qnetwork import QNetwork
from dqn.agent import DQNAgent, solve_env


MAX_SESSIONS = 2000
MAX_STEPS = 200
SOLVED_SCORE = 200  # 190


if __name__ == '__main__':
    start_time = time.time()
    env = gym.make("CartPole-v0").env
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(network, env.action_space.n)
    rewards = solve_env(env, agent, max_sessions=MAX_SESSIONS, t_max=MAX_STEPS, solved=SOLVED_SCORE)
    end_time = time.time()
    print('Finished in {} seconds'.format(int(end_time-start_time)))
    plt.plot(rewards)
    plt.show()
