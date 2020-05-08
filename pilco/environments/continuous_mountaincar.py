"""
Our modification of the OpenAI Gym Continuous Mountain Car by Olivier Sigaud:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

which was (ultimately) based on Sutton's implementation:
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class Mountaincar(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 30}

    def __init__(self):

        # State and action bounds
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = - 3.0
        self.max_position = 3.0
        self.max_speed = 0.07
        self.goal_position = 0.5

        # Force per mass the car can output
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed],
                                  dtype=np.float32)

        self.high_state = np.array([self.max_position, self.max_speed],
                                   dtype=np.float32)

        self.viewer = None

        # Allowed action space
        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,),
                                       dtype=np.float32)

        self.seed()

        # Temporary hack to work with rest of library
        self.env = self


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        # Check if action is in permissible space
        if not self.action_space.contains(action):
            EnvironmentError(f'Expected action in the range of [-1., 1.] '
                             f'got action {action}.')

        # Unpack positiion and valocity
        position, velocity = self.state

        # Increment position by velocity
        position_ = position + velocity

        # Increment velocity by Euler rule and clip
        velocity_ = velocity + action * self.power - 0.0025 * math.cos(3 * position)
        velocity_ = np.clip(velocity_, - self.max_speed, self.max_speed)

        self.state = np.array([position_, velocity_])

        return self.state, None, False, {}


    def reset(self):
        self.state = np.array([-0.5, 0.])
        return np.array(self.state)


    def _height(self, xs):
        return 0.55 + 0.45 * np.sin(3 * xs)

    def render(self, mode='human'):

        # Set picture size
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width

        # Set car size
        carwidth = 40
        carheight = 20

        if self.viewer is None:

            from gym.envs.classic_control import rendering

            # Car constants
            clearance = 10

            # Overall viewer
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Track on which the car moves
            xs = np.linspace(self.min_position, self.max_position, 200)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            # Add car
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)
            self.cartrans = rendering.Transform()

            # Car chasis
            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            # Front wheel
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)

            # Back wheel
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            # Flagpole on mountain peak
            flagx = scale * (0.5 - self.min_position)
            flagy1 = scale * self._height(self.goal_position)
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1),
                                      (flagx, flagy2))
            self.viewer.add_geom(flagpole)

            # Flag on flagpole
            flag = rendering.FilledPolygon([(flagx, flagy2),
                                            (flagx, flagy2 - 10),
                                            (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)


        # Translate and rotate car
        self.cartrans.set_translation(scale * (self.state[0] - self.min_position),
                                      scale * self._height(self.state[0]))
        self.cartrans.set_rotation(math.cos(3 * self.state[0]))

        return self.viewer.render(return_rgb_array=mode=='rgb_array')


    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None
