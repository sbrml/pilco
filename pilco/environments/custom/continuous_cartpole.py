"""
Our modification of Ian Danforth's modification:
https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8

of the OpenAI Gym (Discrete) Cartpole:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

which was itself based on Sutton's implementation:
http://incompleteideas.net/sutton/book/code/pole.c
"""

from pilco.errors import EnvironmentError

import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np


class CartPole(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self):

        super().__init__()

        # Pendulum constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0

        # Time between state updates
        self.tau = 0.02

        # Minimum and maximum actions
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Action space is a continuous range
        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,))

        # Seed environment
        self.seed()

        # Set viewer, state and steps_beyond_done
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # Temporary hack to work with rest of library
        self.env = self


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step_physics(self, force):

        # Unpack current state
        x, x_dot, theta, theta_dot = self.state

        # Sine and cosine of theta for envolving dynamics
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Compute intermediate quantities
        temp1 = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        temp2 = (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))

        # Acceleration of position and angle depend on cartpole parameters and angle
        thetaacc = (self.gravity * sintheta - costheta * temp1) / temp2
        xacc = temp1 - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update position and angle by their rates of change
        x = x + self.tau * x_dot
        theta = theta + self.tau * theta_dot

        # Update rates of change of position and angle by their acceleration
        x_dot = x_dot + self.tau * xacc
        theta_dot = theta_dot + self.tau * thetaacc

        return np.array([x, x_dot, theta, theta_dot])


    def step(self, action):

        # Check that action is in the allowed range
        if not self.action_space.contains(action):
            raise EnvironmentError(f'{type(self)} received invalid action. '
                                   f'Expected action in range [-1., 1.], '
                                   f'instead got {action}.')

        # Cast action to float to strip np trappings
        self.state = self.step_physics(self.force_mag * float(action))

        return np.array(self.state), None, False, {}


    def reset(self):
        """
        Reset state randomly, close to the top.
        :return:
        """

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human'):
        """
        Render images of the environment.

        :param mode:
        :return:
        """

        # Image height and width
        screen_width = 600
        screen_height = 400

        # Cart size
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:

            from gym.envs.classic_control import rendering

            # Overall viewer
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Cart and pole translations
            self.carttrans = rendering.Transform()
            self.poletrans = rendering.Transform(translation=(0, cartheight / 4.0))

            # Cart polygon
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # Pole polygon
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            # Axle circle
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)

            # Axle and track
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)


        if self.state is None:
            return None

        # Set cart position and pole rotation
        cartx = self.state[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-self.state[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def close(self):

        """
        Close the environment.
        :return:
        """

        if self.viewer:
            self.viewer.close()
            self.viewer = None
