import gym
import numpy as np

class Environment:

    def __init__(self, name):

        self.env = gym.make(name)
        self.env.reset()


    def reset(self):

        self.env.reset()

        return self.env.state


    def step(self, action):

        state = self.env.state.copy()

        self.env.step(action)

        next_state = self.env.state.copy()

        return state, action, next_state
