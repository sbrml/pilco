import gym
import numpy as np

class Environment:

    def __init__(self, name, reward_loc, reward_scale):

        self.env = gym.make(name)
        self.env.reset()

        self.reward_loc = reward_loc
        self.reward_scale = reward_scale
        

    def reset(self):

        self.env.reset()
        

    def step(self, action):

        state = self.env.state

        self.env.step(action)

        next_state = self.env.state

        reward = self.reward(state, action)

        return state, action, next_state, reward


    def reward(self, state, action):
        
        quadratic = (state - self.reward_loc) ** 2 / self.reward_scale

        reward = 1 - np.exp(- 0.5 * np.sum(quadratic))
        
        return reward
