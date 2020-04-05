import tensorflow as tf

import numpy as np

from pilco.environments import Environment
from pilco.policies import RBFPolicy, TransformedPolicy
from pilco.transforms import SineTransform, CosineTransform, SineTransformWithPhase, AbsoluteValueTransform
from pilco.costs import EQCost
from pilco.agents import EQGPAgent

import not_tf_opt as ntfo

from tqdm import trange

import matplotlib.pyplot as plt
from sacred import Experiment

import time

import imageio

from pilco.utils import plot_pendulum_rollouts

experiment = Experiment('pendulum-experiment')


@experiment.config
def config():
    # Lengthscale for gaussian cost
    target_scale = [[0.5]]

    agent_replay_buffer_limit = 100000

    # Number of rbf features in policy
    num_rbf_features = 50

    # Subsampling factor
    sub_sampling_factor = 4

    # Number of episodes of random sampling of data
    num_random_episodes = 2

    # Number of steps per random episode
    num_steps_per_random_episode = 30

    # Parameters for agent-environment loops
    optimisation_horizon = 40
    num_optim_steps = 100

    num_episodes = 10
    num_steps_per_episode = 40

    # Number of optimisation steps for dynamics GPs
    num_dynamics_optim_steps = 30

    # Policy learn rate
    policy_lr = 5e-2

    # Dynamics learn rate
    dynamics_lr = 1e-1

    # Number of times to restart the dynamics optimizer
    dynamics_optimisation_restarts = 3

    # Optimsation exit criterion tolerance
    tolerance = 1e-5


class PendulumAgent(EQGPAgent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy,
                 cost,
                 dtype=tf.float64,
                 name='pendulum_agent',
                 **kwargs):
        super().__init__(in_state_dim=state_dim + 2,
                         out_state_dim=state_dim,
                         action_dim=action_dim,
                         policy=policy,
                         cost=cost,
                         dtype=dtype,
                         name=name,
                         **kwargs)

        self.trig_space_transform = SineTransformWithPhase(lower=-1,
                                                           upper=1,
                                                           phase=tf.constant([[0., np.pi / 2.]],
                                                                             dtype=dtype))

    def preprocess_observations(self, state, action, next_state):
        state = self._validate_and_convert(state, last_dim=self.in_state_dim - 2)
        action = self._validate_and_convert(action, last_dim=self.action_dim)
        next_state = self._validate_and_convert(next_state, last_dim=self.out_state_dim)

        delta_state = next_state - state

        # Trig transform on the input state
        trig_state = tf.stack([tf.sin(state[:, 0]),
                               tf.cos(state[:, 0]),
                               state[:, 0],
                               state[:, 1]],
                              axis=1)

        return trig_state, action, delta_state

    def match_moments(self, mean_full, cov_full):
        mean_full = tf.reshape(mean_full, shape=(1, self.in_state_and_action_dim - 2))
        cov_full = tf.reshape(cov_full, shape=(self.in_state_and_action_dim - 2,
                                               self.in_state_and_action_dim - 2))

        # Replicate the theta components in the loc and cov
        mean_full_ = tf.concat([mean_full[:, :1], mean_full[:, :1], mean_full], axis=1)

        rep_cov_theta_theta = tf.tile(cov_full[:1, :1], [3, 3])
        rep_cov_theta_thetadot = tf.tile(cov_full[:1, 1:], [3, 1])
        rep_cov_thetadot_theta = tf.tile(cov_full[1:, :1], [1, 3])

        row_blocks = [
            tf.concat([rep_cov_theta_theta, rep_cov_theta_thetadot], axis=1),
            tf.concat([rep_cov_thetadot_theta, cov_full[1:, 1:]], axis=1)
        ]

        cov_full_ = tf.concat(row_blocks, axis=0)

        mean_full_, cov_full_ = self.trig_space_transform.match_moments(loc=mean_full_[0],
                                                                        cov=cov_full_,
                                                                        indices=[0, 1])

        mean, cov, cross_cov = super().match_delta_moments(mean_full_, cov_full_)

        # Calcuate successor mean and covariance
        mean = mean + mean_full[0, :self.out_state_dim]

        cov = cov + cov_full[:self.out_state_dim, :self.out_state_dim]

        cross_cov = cross_cov[2:, :]

        cov = cov + cross_cov + tf.transpose(cross_cov)

        return mean, cov

    def gp_posterior_predictive(self, x_star):
        x_star = self._validate_and_convert(x_star, self.in_state_and_action_dim - 2)

        x_star_ = tf.concat([x_star[:, :1], x_star[:, :1], x_star], axis=1)

        return super().gp_posterior_predictive(x_star_)


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
               target_scale,
               num_rbf_features,
               optimisation_horizon,
               dynamics_optimisation_restarts,
               num_dynamics_optim_steps,
               num_optim_steps,
               num_episodes,
               num_steps_per_episode,
               agent_replay_buffer_limit,
               policy_lr,
               dynamics_lr,
               tolerance):
    dtype = tf.float64

    # Create pendulum environment and reset
    env = Environment(name='Pendulum-v0',
                      sub_sampling_factor=sub_sampling_factor)
    env.reset()

    # Create stuff for our controller
    target_loc = tf.constant([[0.]], dtype=dtype)
    target_scale = tf.constant(target_scale, dtype=dtype)

    cost_transform = CosineTransform(lower=-1.,
                                     upper=1.,
                                     dtype=dtype)

    eq_cost = EQCost(target_loc=target_loc,
                     target_scale=target_scale,
                     target_dim=1,
                     transform=cost_transform,
                     dtype=dtype)

    # Create EQ policy
    eq_policy = RBFPolicy(state_dim=2,
                          action_dim=1,
                          num_rbf_features=num_rbf_features,
                          dtype=dtype)

    # We can bound the range of the policy by passing it through an appropriately
    # Shifted and scaled sine function.
    sine_transform = SineTransform(lower=-2,
                                   upper=2)

    fucking_work_already = SineTransform(lower=-1,
                                         upper=1)

    abs_transform = AbsoluteValueTransform()

    eq_policy = TransformedPolicy(policy=eq_policy,
                                  transform=sine_transform)

    # Create agent
    eq_agent = PendulumAgent(state_dim=2,
                             action_dim=1,
                             policy=eq_policy,
                             cost=eq_cost,
                             replay_buffer_limit=agent_replay_buffer_limit,
                             dtype=dtype)

    eq_agent.policy.reset()

    for episode in trange(num_random_episodes):

        state = env.reset()
        eq_agent.policy.reset()

        # state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        state = np.array([-np.pi + np.random.normal(0., 0.1), 0.])
        # state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        env.env.env.state = state

        for step in range(num_steps_per_random_episode):
            # action = (tf.random.uniform(shape=()) * 4. - 2)[None]
            action = eq_agent.policy(state)
            state, action, next_state = env.step(action.numpy())

            eq_agent.observe(state, action, next_state)

    print(eq_agent.dynamics_inputs)
    print(eq_agent.dynamics_outputs)

    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)
    init_cov = 1e-4 * tf.eye(2, dtype=tf.float64)

    policy_optimiser = tf.optimizers.Adam(policy_lr)
    dynamics_optimiser = tf.optimizers.Adam(dynamics_lr)

    for episode in range(num_episodes):

        print(eq_agent.num_datapoints)
        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')
        evaluate_agent_dynamics(eq_agent,
                                env,
                                num_episodes=100,
                                num_steps=1,
                                seed=0)

        print('Optimising dynamics')

        if episode > -1:
            # Create variables
            best_eq_scales = eq_agent.eq_scales()
            best_eq_coeff = eq_agent.eq_coeff()
            best_eq_noise_coeff = eq_agent.eq_noise_coeff()

            best_loss = np.inf

            for idx in range(dynamics_optimisation_restarts):

                prev_loss = np.inf

                # Randomly initialize dynamics hyperparameters
                eq_agent.set_eq_scales_from_data()
                eq_agent.eq_scales.assign(eq_agent.eq_scales() + tf.random.uniform(minval=-0.05, maxval=1.05,
                                                                                   dtype=dtype,
                                                                                   shape=best_eq_scales.shape))
                eq_agent.eq_coeff.assign(tf.random.uniform(minval=5e-1, maxval=2e0, shape=best_eq_coeff.shape, dtype=dtype))
                eq_agent.eq_noise_coeff.assign(
                    tf.random.uniform(minval=5e-2, maxval=2e-1, shape=best_eq_noise_coeff.shape, dtype=dtype))

                for n in trange(num_dynamics_optim_steps):

                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(eq_agent.parameters)

                        loss = -eq_agent.dynamics_log_marginal()
                        loss = loss / tf.cast(eq_agent.num_datapoints, dtype=eq_agent.dtype)

                    gradients = tape.gradient(loss, eq_agent.parameters)
                    dynamics_optimiser.apply_gradients(zip(gradients, eq_agent.parameters))

                    # clip_tensor = tf.constant([[2, 2, np.pi, 4, 1],
                    #                            [2, 2, np.pi, 4, 1]],
                    #                           dtype=eq_agent.dtype)
                    #
                    # clipped_eq_scales = tf.minimum(eq_agent.eq_scales(), clip_tensor)

                    clipped_eq_scales = eq_agent.eq_scales()
                    #TODO: Delete this ugly hack, burn it with fire
                    clip_tensor = tf.constant([[0, 0, 100, 0, 0],
                                               [0, 0, 100, 0, 0]],
                                              dtype=eq_agent.dtype)
                    clipped_eq_scales = tf.maximum(clipped_eq_scales, clip_tensor)

                    eq_agent.eq_scales.assign(clipped_eq_scales)

                    if tf.abs(loss - prev_loss) < tolerance:
                        print(f"Early convergence!")
                        break

                    prev_loss = loss

                print(f"Optimization round {idx + 1}/{dynamics_optimisation_restarts}, loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss

                    best_eq_scales = eq_agent.eq_scales()
                    best_eq_coeff = eq_agent.eq_coeff()
                    best_eq_noise_coeff = eq_agent.eq_noise_coeff()

            # Assign best parameters
            eq_agent.eq_coeff.assign(best_eq_coeff)
            eq_agent.eq_noise_coeff.assign(best_eq_noise_coeff)
            eq_agent.eq_scales.assign(best_eq_scales)

        else:
            eq_agent.set_eq_scales_from_data()
            # TODO: Delete this ugly hack, burn it with fire
            clip_tensor = tf.constant([[0, 0, 100, 0, 0],
                                       [0, 0, 100, 0, 0]],
                                      dtype=eq_agent.dtype)
            clipped_eq_scales = tf.maximum(eq_agent.eq_scales(), clip_tensor)
            eq_agent.eq_scales.assign(clipped_eq_scales)

        evaluate_agent_dynamics(eq_agent, env, 1000, 1, seed=0)
        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')
        evaluate_agent_dynamics(eq_agent,
                                env,
                                num_episodes=100,
                                num_steps=1,
                                seed=0)

        eq_agent.policy.reset()

        prev_loss = np.inf

        current_optimisation_horizon = optimisation_horizon
        with trange(num_optim_steps) as bar:

            for n in bar:

                cost = 0.
                loc = init_state
                cov = init_cov

                true_traj = []
                locs = [loc[0].numpy()]
                covs = [cov.numpy()]

                step_costs = []

                with tf.GradientTape(watch_accessed_variables=False) as tape:

                    tape.watch(eq_agent.policy.parameters)

                    for t in range(current_optimisation_horizon):
                        mean_full, cov_full = eq_agent.policy.match_moments(loc, cov)

                        loc, cov = eq_agent.match_moments(mean_full, cov_full)

                        locs.append(loc.numpy())
                        covs.append(cov.numpy())

                        # Moment match by 1/2
                        loc_ = 0.5 * loc[:1]
                        cov_ = 0.25 * cov[:1, :1]

                        # Moment match by sine
                        loc_, cov_ = fucking_work_already.match_moments(loc_, cov_, indices=tf.constant([0]))

                        # Moment match by absolute value
                        loc_, cov_ = abs_transform.match_moments(loc_, cov_, indices=tf.constant([0]))

                        # Moment match by 2
                        loc_ = 2 * loc_
                        cov_ = 4 * cov_

                        step_cost = eq_agent.cost.expected_cost(loc_, cov_)

                        cost = cost + step_cost

                        step_costs.append(step_cost)

                    cost = cost / tf.cast(optimisation_horizon, dtype=eq_cost.dtype)

                gradients = tape.gradient(cost, eq_agent.policy.parameters)

                policy_optimiser.apply_gradients(zip(gradients, eq_agent.policy.parameters))

                if tf.abs(cost - prev_loss) < tolerance:
                    print(f"Early convergence!")
                    break

                prev_loss = cost

                print(tf.stack(step_costs, axis=0).numpy())

                env.reset()
                env.env.env.state = init_state.numpy()[0]
                true_traj.append(init_state.numpy()[0])

                true_cost = 0.

                for step in range(num_steps_per_episode):
                    action = eq_agent.act(state)
                    state, action, next_state = env.step(action.numpy())
                    true_cost = true_cost + eq_cost(tf.convert_to_tensor(2 * tf.abs(tf.sin(next_state[None, :1] / 2))))

                    true_traj.append(next_state)

                true_traj = np.stack(true_traj, axis=0)

                # Run rollout and plot
                locs_all = np.stack(locs, axis=0)
                vars_all = np.stack(covs, axis=0)[:, [0, 1], [0, 1]]
                steps = np.arange(locs_all.shape[0])

                plot_pendulum_rollouts(steps,
                                       true_traj,
                                       locs_all,
                                       vars_all,
                                       plot_path='../plots/',
                                       plot_prefix=f'optim-{episode}-{n}')

                true_cost = true_cost / num_steps_per_episode

                bar.set_description(f'Cost: {cost.numpy():.5f} (pred) '
                                    f'{true_cost.numpy():.5f} (true)')

        print(f'Performing episode {episode + 1}:')

        env.reset()
        env.env.env.state = init_state.numpy()[0]

        frames = []

        for step in trange(num_steps_per_episode):
            frames.append(env.env.render(mode='rgb_array'))

            action = eq_agent.act(state)
            state, action, next_state = env.step(action.numpy())
            eq_agent.observe(state, action, next_state)

        env.env.close()

        imageio.mimwrite(f'../gifs/pendulum-{episode}.gif', frames)
