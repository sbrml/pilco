from pilco.environments import Environment
from pilco.errors import EnvironmentError

import pytest

import numpy as np


@pytest.mark.parametrize("env_name, sub_sampling_factor, action, num_steps",
                         [('CartPole', 2, [0.], 10),
                          ('MountainCar', 2, [0.], 10)])
def test_custom_environment_runs(env_name, sub_sampling_factor, action, num_steps):
    """
    Test custom implementations of our environments initialise and run.
    :return:
    """

    environment = Environment(name=env_name,
                              sub_sampling_factor=sub_sampling_factor)

    for i in range(num_steps):
        environment.step(np.array(action))



@pytest.mark.parametrize("env_name, sub_sampling_factor, action",
                         [('CartPole', 2, [-5.]),
                          ('CartPole', 2, [5.]),
                          ('MountainCar', 2, [-5.]),
                          ('MountainCar', 2, [5.])])
def test_custom_environment_failures(env_name, sub_sampling_factor, action):
    """
    Test custom implementations of our environments fail when disallowed
    actions are passed to them.
    :return:
    """

    environment = Environment(name=env_name,
                              sub_sampling_factor=sub_sampling_factor)

    with pytest.raises(EnvironmentError):
        environment.step(np.array(action))
