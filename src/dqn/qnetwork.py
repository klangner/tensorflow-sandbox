#
# Various QNetwork architectures used in different scenarios
#

import numpy as np
import gym
from keras import models, layers, optimizers


# ----------------------------------------------------------------------------------------------------------------------
# Define QNetwork
class QNetwork:

    def __init__(self, input_shape, n_actions, alpha=0.0003):
        self.input_shape = input_shape
        self.output_shape = n_actions
        self.alpha = alpha
        self.model = self._build_network()

    def _build_network(self):
        model = models.Sequential()
        model.add(layers.InputLayer(self.input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.output_shape, activation='linear'))
        opt = optimizers.RMSprop(lr=self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def predict(self, state):
        """Make prediction for single state and return q values for all actions"""
        s = np.expand_dims(state, axis=0)
        return self.model.predict(s)[0]

    def predict_batch(self, states):
        """Make prediction for list of states"""
        return self.model.predict(states)

    def train(self, x, y):
        self.model.fit(x, y, batch_size=64, verbose=0)


# ----------------------------------------------------------------------------------------------------------------------
# Tests
GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'


def tests_qnetwork():
    """Check if QNetwork will compile"""
    env = gym.make("CartPole-v0")
    network = QNetwork(env.observation_space.shape, env.action_space.n)
    network.predict(env.reset())
    print(GREEN + 'QNetwork ok!' + ENDC)


if __name__ == "__main__":
    tests_qnetwork()
