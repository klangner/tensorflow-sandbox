# Solve CartPole using DQN
#
# https://gym.openai.com/envs/CartPole-v0/
#
# Some useful resources:
# * https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#

import numpy as np
import gym
from dqn.qnetwork import QNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, network, n_actions):
        self.n_actions = n_actions
        self.memory_capacity = 100000
        self.epsilon_min = 0.01
        self.epsilon_max = 0.01
        self.epsilon_lambda = 0.001
        self.batch_size = 64
        self.epsilon = self.epsilon_max
        self.model = network
        self.replays = ReplayBuffer(self.memory_capacity)
        self.step = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            qvalues = self.model.predict(state)
            action = np.argmax(qvalues)
            self.step += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max-self.epsilon_min) * np.power(np.e, self.epsilon_lambda*self.step)
        return action

    def add_observation(self, state, action, reward, next_state, is_done):
        self.replays.add(state, action, reward, next_state, is_done)

    def train(self):
        states, actions, rewards, states_next, dones = self.replays.sample(self.batch_size)
        qvalues = self.model.predict_batch(states)
        qvalues_next = self.model.predict_batch(states_next)
        y = self._build_training_set(qvalues, qvalues_next, actions, rewards, dones)
        self.model.train(states, y)

    @staticmethod
    def _build_training_set(qvalues, qvalues_next, actions, rewards, dones, gamma=0.99):
        """
        Create training set for QNetwork.
        Params:
          qvalues           - Q values for the starting state
          qvalues_next      - Q values for the state the next state
          actions           - Actions taken
          rewards           - Rewards received after taking action
          dones             - Did this action end the episode?

        Returns:
          Expected qvalues
        """
        y = qvalues.copy()
        next_rewards = np.where(dones, np.zeros(rewards.shape), np.max(qvalues_next, axis=1))
        y[np.arange(y.shape[0]), actions] = rewards + gamma * next_rewards
        return y


def generate_session(env, agent, t_max=1000):
    """Generate single session using given environment and agent"""
    total_reward = 0
    state = env.reset()
    for t in range(t_max):

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.add_observation(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward

        state = next_state
        if done:
            break

    return total_reward


def solve_env(env, agent, max_sessions=2000, t_max=200, solved=190):
    """Solve given environment using given agent"""
    rewards = []
    for i in range(1, max_sessions+1):
        session_reward = generate_session(env, agent, t_max)
        rewards.append(session_reward)
        if i % 100 == 0:
            mean_score = np.mean(rewards[-100:])
            print('Step: {}, mean reward: {}'.format(i, mean_score))
            if mean_score > solved:
                print("Solved in {} steps".format(i))
                break
    return rewards


# ----------------------------------------------------------------------------------------------------------------------
# Tests

def test_agent():
    env = gym.make("CartPole-v0")
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(network, env.action_space.n)
    s1 = env.reset()
    a = agent.act(s1)
    s2, r, d, _ = env.step(a)
    agent.add_observation(s1, a, r, s2, d)
    agent.train()
    assert True


def test_build_training_set():
    env = gym.make("CartPole-v0")
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(network, 2)
    qvalues = np.zeros((5, 2))
    qvalues2 = np.ones((5, 2))
    actions = np.array([0, 1, 0, 1, 0])
    rewards = np.array([1, 2, 3, 4, 5])
    dones = np.array([False, False, False, False, True])
    expected_y = np.array([[2, 0], [0, 3], [4, 0], [0, 5], [5, 0]])
    y = agent._build_training_set(qvalues, qvalues2, actions, rewards, dones, 1.0)
    assert np.array_equal(y, expected_y), 'Wrong expected qvalue calculated'


def test_session_gen():
    env = gym.make("CartPole-v0")
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(network, env.action_space.n)
    generate_session(env, agent)
    assert True


if __name__ == "__main__":
    test_agent()
    test_build_training_set()
    test_session_gen()


