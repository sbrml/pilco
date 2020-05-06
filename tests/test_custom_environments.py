from pilco.environments import Environment
from pilco.errors import EnvironmentError

import numpy as np


def test_custom_environment_runs():
    """
    Test custom implementations of our environments initialise and run.
    :return:
    """

    env_names = ['Cartpole',
                 'Mountaincar']

    actions = [[0.],
               [0.]]

    for env_name, action in zip(env_names, actions):

        environment = Environment(env_name,
                                  sub_sampling_factor=2)

        for i in range(10):
            environment.step(np.array(action))


def test_custom_environment_failures():
    """
    Test custom implementations of our environments fail when disallowed
    actions are passed to them.
    :return:
    """

    env_names = ['Cartpole',
                 'Mountaincar']

    actions = [[5.],
               [5.]]

    for env_name, action in zip(env_names, actions):

        environment = Environment(env_name,
                                  sub_sampling_factor=2)

        try:
            environment.step(np.array(action))

        except EnvironmentError:
            return

        EnvironmentError(f'{type(environment)} should have failed'
                         f'with action {action} but it did not.')
