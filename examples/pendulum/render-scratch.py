import gym
import time
import gym.wrappers as wrappers
import matplotlib.pyplot as plt
import numpy as np
import imageio

env = gym.make('Pendulum-v0')
env.reset()

frames = []

for i in range(100):
    env.render()
    env.step(0 * env.action_space.sample()) # take a random action
    
    frames.append(env.render(mode='rgb_array'))

env.close()

imageio.mimwrite('../gifs/thank-fucking-fuck.gif',
                 frames,
                 fps=20)
