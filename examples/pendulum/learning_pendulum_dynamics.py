import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import gpflow as gpf

import datetime
import numpy as np

from pendulum_eq_policy import PendulumEQPolicy
from pilco.agents import GPExplorationAgent
from pilco.environments import Environment
from pilco.policies import BatchedEQPolicy
from pilco.transforms import SineTransform, AbsoluteValueTransform, \
    ChainedMomentMatchingTransfom, AffineTransform
from pilco.costs import EQCost
from pilco.agents import EQGPAgent

import not_tf_opt as ntfo

from tqdm import trange

from sacred import Experiment

import imageio

from pilco.utils import plot_pendulum_rollouts

experiment = Experiment('pendulum-experiment')


@experiment.config
def config():

    # Replay buffer limit
    agent_replay_buffer_limit = 240

    # Number of rollouts
    num_rollouts = 50

    # Number of eq features in policy
    num_eq_features = 50

    # Subsampling factor
    sub_sampling_factor = 2

    # Number of episodes of random sampling of data
    num_random_episodes = 2

    # Number of steps per random episode
    num_steps_per_random_episode = 40

    # Parameters for agent-environment loops
    horizon = 40
    num_optim_steps = 100

    num_episodes = 10
    num_steps_per_episode = 40

    # Number of optimisation steps for dynamics GPs
    num_dynamics_optim_steps = 50

    # Number of times to restart the dynamics optimizer
    dynamics_optimisation_restarts = 3

    # Optimsation exit criterion tolerance
    tolerance = 0.

    # Root directory for saving
    root_dir = "./pendulum/"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def evaluate_agent_dynamics(agent, env, num_episodes, num_steps, seed):

    test_data = sample_transitions_uniformly(env,
                                             num_episodes,
                                             num_steps,
                                             seed)

    test_inputs, test_outputs = test_data

    pred_means, pred_vars = agent.gp_posterior_predictive(test_inputs)
    pred_means = pred_means + test_inputs[:, :2]

    sq_diff = tf.math.squared_difference(pred_means,
                                         test_outputs)

    max_diff = tf.reduce_max(sq_diff ** 0.5, axis=0)
    min_diff = tf.reduce_min(sq_diff ** 0.5, axis=0)

    rmse = tf.reduce_mean(sq_diff, axis=0) ** 0.5
    smse = tf.reduce_mean(sq_diff / pred_vars, axis=0)

    rmse = [round(num, 3) for num in rmse.numpy()]
    smse = [round(num, 3) for num in smse.numpy()]
    max_diff = [round(num, 3) for num in max_diff.numpy()]
    min_diff = [round(num, 3) for num in min_diff.numpy()]

    print(f'RMSE: {rmse} SMSE {smse} Min {min_diff} Max {max_diff}')


def sample_transitions_uniformly(env, num_episodes, num_steps, seed):
    np.random.seed(seed)

    state_actions = []
    next_states = []

    for episode in range(num_episodes):

        state = env.reset()

        state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        env.env.env.state = state

        for step in range(num_steps):
            action = tf.random.uniform(shape=()) * 4. - 2
            state, action, next_state = env.step(action[None].numpy())

            state_action = np.concatenate([state, action], axis=0)

            state_actions.append(state_action)
            next_states.append(next_state)

    state_actions = np.stack(state_actions, axis=0)
    next_states = np.stack(next_states, axis=0)

    return state_actions, next_states


@experiment.automain
def experiment(num_random_episodes,
               num_steps_per_random_episode,
               sub_sampling_factor,
               num_rollouts,
               num_eq_features,
               horizon,
               dynamics_optimisation_restarts,
               num_episodes,
               num_steps_per_episode,
               agent_replay_buffer_limit,
               root_dir,
               current_time):

    dtype = tf.float64

    # Create pendulum environment and reset
    env = Environment(name='Pendulum-v0',
                      sub_sampling_factor=sub_sampling_factor)
    env.reset()

    eq_policy = BatchedEQPolicy(state_dim=2,
                                action_dim=1,
                                num_eq_features=num_eq_features,
                                dtype=dtype)

    gp_means = []
    gp_covs = [gpf.kernels.RBF(variance=1., lengthscales=[np.pi, 8., 1.]),
               gpf.kernels.RBF(variance=1., lengthscales=[np.pi, 8., 1.])]

    agent = GPExplorationAgent(state_dim=2,
                               action_dim=1,
                               gp_means=gp_means,
                               gp_covs=gp_covs,
                               policy=eq_policy,
                               dtype=dtype,
                               replay_buffer_limit=agent_replay_buffer_limit)

    for episode in trange(num_random_episodes):

        env.reset()
        agent.policy.reset()

        state = np.array([-np.pi, 0.])
        env.env.env.state = state

        for step in range(num_steps_per_random_episode):
            action = agent.policy(np.reshape(state, (1, -1)))
            state, action, next_state = env.step(action.numpy())

            agent.observe(state, action, next_state)

    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)
    init_cov = 1e-8 * tf.eye(2, dtype=tf.float64)

    init_state_dist = tfd.MultivariateNormalFullCovariance(loc=init_state[0],
                                                           covariance_matrix=init_cov)

    for episode in range(num_episodes):

        agent.policy.reset()

        def expected_kl_divergence():

            rollout_results = agent.rollout(state_dist=init_state_dist,
                                            num_rollouts=num_rollouts,
                                            horizon=horizon,
                                            recondition=True)

            inputs, outputs, states, kl_divergence = rollout_results

            env.reset()
            env.env.env.state = init_state.numpy()[0]
            state = init_state.numpy()[0]

            true_traj = []
            true_actions = []
            true_traj.append(init_state.numpy()[0])

            for step in range(num_steps_per_episode):

                action = agent.act(np.reshape(state, (1, -1)))
                state, action, next_state = env.step(action.numpy())

                true_traj.append(next_state)
                true_actions.append(action)

            # New plotter here!
            print(f"Current KL: {kl_divergence}")

            return -kl_divergence

        while True:

            loss, converged, diverged = ntfo.minimize(function=expected_kl_divergence,
                                                      vs=[agent.policy.eq_locs,
                                                          agent.policy.eq_log_scales,
                                                          agent.policy.eq_weights,
                                                          ],
                                                      explicit=False,
                                                      max_iterations=150)

            print(f"Policy loss: {loss},"
                  f"converged: {converged}, "
                  f"diverged: {diverged}")

            if not diverged:
                break

            else:
                print("Optimization diverged, restarting optimization!")

        print(f'Performing episode {episode + 1}:')

        env.reset()
        env.env.env.state = init_state.numpy()[0]
        state = init_state.numpy()[0]

        frames = []

        for step in trange(num_steps_per_episode):
            frames.append(env.env.render(mode='rgb_array'))

            action = agent.act(np.reshape(state, (1, -1)))
            state, action, next_state = env.step(action.numpy())
            agent.observe(state, action, next_state)

        env.env.close()

        imageio.mimwrite(f'{root_dir}/gifs/learning-dynamics/pendulum-{episode}.gif', frames)

        agent.save_weights(f"{root_dir}/saved_agents/learning_dynamics/{current_time}/episode_{episode}/model")
