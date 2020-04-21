import tensorflow as tf

import datetime
import numpy as np

from pilco.environments import Environment
from pilco.policies import Policy, RBFPolicy, TransformedPolicy
from pilco.transforms import SineTransform, SineTransformWithPhase, AbsoluteValueTransform, \
    ChainedMomentMatchingTransfom, AffineTransform
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
    target_scale = [[1.]]

    use_lbfgs = True

    agent_replay_buffer_limit = 200

    # Number of rbf features in policy
    num_rbf_features = 50

    # Subsampling factor
    sub_sampling_factor = 2

    # Number of episodes of random sampling of data
    num_random_episodes = 2

    # Number of steps per random episode
    num_steps_per_random_episode = 40

    # Parameters for agent-environment loops
    optimisation_horizon = 40
    num_optim_steps = 100

    num_episodes = 10
    num_steps_per_episode = 40

    # Number of optimisation steps for dynamics GPs
    num_dynamics_optim_steps = 50

    # Policy learn rate
    policy_lr = 3e-2

    # Dynamics learn rate
    dynamics_lr = 1e-1

    # Number of times to restart the dynamics optimizer
    dynamics_optimisation_restarts = 3

    # Optimsation exit criterion tolerance
    tolerance = 0.

    root_dir = "/Users/gergelyflamich/Documents/sbrml/pilco/"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{root_dir}/saved_agents/{current_time}/"


class PendulumAgent(EQGPAgent):

    def __init__(self,
                 policy,
                 cost,
                 dtype=tf.float64,
                 name='pendulum_agent',
                 **kwargs):
        super().__init__(in_state_dim=4,
                         out_state_dim=2,
                         action_dim=1,
                         policy=policy,
                         cost=cost,
                         dtype=dtype,
                         name=name,
                         **kwargs)

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
        """
        :param mean_full:
        :param cov_full:
        :return:

        Assumes *mean_full* and *cov_full* are in [sin, cos, theta, theta_dot, torque] space.
        """

        mean, cov, cross_cov = super().match_delta_moments(mean_full, cov_full)

        # print(f'{type(self)}.match_moments mean.shape, cov.shape, cross_cov.shape {mean.shape, cov.shape, cross_cov.shape}')

        # Calcuate successor mean and covariance
        mean = mean + mean_full[0, 2:4]

        cov = cov + cov_full[2:4, 2:4]

        cross_cov = cross_cov[2:, :]

        cov = cov + cross_cov + tf.transpose(cross_cov)

        return mean, cov


class PendulumPolicy(Policy):

    def __init__(self,
                 num_rbf_features,
                 dtype,
                 name='pendulum_policy',
                 **kwargs):
        super().__init__(state_dim=4,
                         action_dim=1,
                         dtype=dtype,
                         name=name,
                         **kwargs)

        # Create EQ policy
        eq_policy = RBFPolicy(state_dim=4,
                              action_dim=1,
                              num_rbf_features=num_rbf_features,
                              dtype=dtype)

        # We can bound the range of the policy by passing it through an appropriately
        # Shifted and scaled sine function.
        sine_transform = SineTransform(lower=-2,
                                       upper=2)

        self.eq_policy = TransformedPolicy(policy=eq_policy,
                                           transform=sine_transform)

        phase = tf.constant([0., np.pi / 2.], dtype=dtype)
        self.trig_space_transform = SineTransformWithPhase(lower=-1,
                                                           upper=1,
                                                           phase=phase)

    def match_moments(self, mean, cov):
        """
        :param self:
        :param mean:
        :param cov:
        :return:

        Assumes *mean* and *cov* are in [theta, theta_dot] space.
        Output is in [sin, cos, theta, theta_dot, action] space.
        """

        # print(f'{type(self)}.forward before replicating mean.shape, cov.shape {mean.shape, cov.shape}')
        mean, cov = self.match_moments_to_trig_and_theta_space(mean, cov)
        # print(f'{type(self)}.forward after replicating mean.shape, cov.shape {mean.shape, cov.shape}')
        mean_full, cov_full = self.eq_policy.match_moments(mean, cov)
        # print(f'{type(self)}.forward after last moment matching mean_full.shape, cov_full.shape {mean.shape, cov.shape}')

        return mean_full, cov_full

    def call(self, state):
        """
        :param self:
        :param state:
        :return:

        Assumes *state* is in [theta, theta_dot] space.
        Output is in [sin, cos, theta, theta_dot, action] space.
        """

        state = tf.reshape(state, shape=(1, 2))

        # Replicate the theta components in the loc and cov
        state = tf.concat([state[:, :1], state[:, :1], state], axis=1)

        state = self.trig_space_transform(tensor=state[0],
                                          indices=[0, 1])

        action = self.eq_policy(state)

        return action

    def match_moments_to_trig_and_theta_space(self, mean, cov):
        mean = tf.reshape(mean, shape=(1, 2))
        cov = tf.reshape(cov, shape=(2, 2))

        # Replicate the theta components in the loc and cov
        mean_ = tf.concat([mean[:, :1], mean[:, :1], mean], axis=1)

        rep_cov_theta_theta = tf.tile(cov[:1, :1], [3, 3])
        rep_cov_theta_thetadot = tf.tile(cov[:1, 1:], [3, 1])
        rep_cov_thetadot_theta = tf.tile(cov[1:, :1], [1, 3])

        row_blocks = [
            tf.concat([rep_cov_theta_theta, rep_cov_theta_thetadot], axis=1),
            tf.concat([rep_cov_thetadot_theta, cov[1:, 1:]], axis=1)
        ]

        cov_ = tf.concat(row_blocks, axis=0)

        # Moment match (replicated) thetas across sine and cosine
        mean, cov = self.trig_space_transform.match_moments(loc=mean_[0],
                                                            cov=cov_,
                                                            indices=[0, 1])

        return mean, cov

    def reset(self):
        self.eq_policy.reset()

    def clip(self):
        rbf_locs = self.eq_policy.policy.rbf_locs().numpy()
        rbf_locs[:, 2] = 0.
        self.eq_policy.policy.rbf_locs.assign(rbf_locs)

        rbf_log_scales = self.eq_policy.policy.rbf_log_scales().numpy()
        rbf_log_scales[:, 2] = 10
        self.eq_policy.policy.rbf_log_scales.assign(rbf_log_scales)


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
               tolerance,
               use_lbfgs,
               root_dir,
               save_dir):
    dtype = tf.float64

    # Create pendulum environment and reset
    env = Environment(name='Pendulum-v0',
                      sub_sampling_factor=sub_sampling_factor)
    env.reset()

    # Create stuff for our controller
    target_loc = tf.constant([[0.]], dtype=dtype)
    target_scale = tf.constant(target_scale, dtype=dtype)

    cost_transform = ChainedMomentMatchingTransfom(
        transforms=[
            AffineTransform(scale=0.5),
            SineTransform(lower=-1., upper=1.),
            AbsoluteValueTransform(),
            AffineTransform(scale=2.)
        ]
    )
    # THESE ARE THE PARTS OF THE COST WE ARE CURRENTLY USING
    # cost_sine_transform = SineTransform(lower=-1,
    #                                     upper=1)
    #
    # cost_abs_transform = AbsoluteValueTransform()

    eq_cost = EQCost(target_loc=target_loc,
                     target_scale=target_scale,
                     target_dim=1,
                     dtype=dtype)

    # eq_policy = PendulumPolicy(num_rbf_features=num_rbf_features,
    #                            dtype=dtype)
    #
    # # Create agent
    # eq_agent = PendulumAgent(policy=eq_policy,
    #                          cost=eq_cost,
    #                          replay_buffer_limit=agent_replay_buffer_limit,
    #                          dtype=dtype)

    eq_policy = TransformedPolicy(policy=RBFPolicy(state_dim=2,
                                                   action_dim=1,
                                                   num_rbf_features=num_rbf_features,
                                                   dtype=dtype),
                                  transform=SineTransform(-2, 2))

    eq_agent = EQGPAgent(in_state_dim=2,
                         out_state_dim=2,
                         action_dim=1,
                         policy=eq_policy,
                         cost=eq_cost,
                         dtype=dtype,
                         replay_buffer_limit=agent_replay_buffer_limit)

    eq_agent.policy.reset()

    for episode in trange(num_random_episodes):

        state = env.reset()
        eq_agent.policy.reset()

        # state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        state = np.array([-np.pi, 0.])
        # state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        env.env.env.state = state

        for step in range(num_steps_per_random_episode):
            # action = (tf.random.uniform(shape=()) * 4. - 2)[None]
            action = eq_agent.policy(state)
            state, action, next_state = env.step(action.numpy())

            eq_agent.observe(state, action, next_state)

    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)
    init_cov = 0. * tf.eye(2, dtype=tf.float64)

    policy_optimiser = tf.optimizers.Adam(policy_lr)
    dynamics_optimiser = tf.optimizers.Adam(dynamics_lr)

    for episode in range(num_episodes):

        print(eq_agent.num_datapoints)
        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')
        # evaluate_agent_dynamics(eq_agent,
        #                         env,
        #                         num_episodes=100,
        #                         num_steps=1,
        #                         seed=0)

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
                eq_agent.eq_coeff.assign(
                    tf.random.uniform(minval=5e-1, maxval=2e0, shape=best_eq_coeff.shape, dtype=dtype))
                eq_agent.eq_noise_coeff.assign(
                    tf.random.uniform(minval=5e-2, maxval=2e-1, shape=best_eq_noise_coeff.shape, dtype=dtype))

                loss, converged, diverged = ntfo.minimize(
                    function=lambda: -eq_agent.dynamics_log_marginal() / tf.cast(eq_agent.num_datapoints,
                                                                                 dtype=eq_agent.dtype),
                    vs=[eq_agent.eq_scales, eq_agent.eq_coeff, eq_agent.eq_noise_coeff],
                    explicit=False)

                print(f"Optimization round {idx + 1}/{dynamics_optimisation_restarts}, "
                      f"loss: {loss:.4f}, "
                      f"converged: {converged}, "
                      f"diverged: {diverged}")

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

        # clip_tensor = tf.constant([[1000., 1000., 2.],
        #                            [1000., 1000., 2.]], dtype=dtype)
        # clipped_tensor = tf.minimum(clip_tensor, eq_agent.eq_scales())
        # eq_agent.eq_scales.assign(clipped_tensor)

        # evaluate_agent_dynamics(eq_agent, env, 1000, 1, seed=0)
        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')
        # evaluate_agent_dynamics(eq_agent,
        #                         env,
        #                         num_episodes=100,
        #                         num_steps=1,
        #                         seed=0)

        eq_agent.policy.reset()

        if use_lbfgs:

            def expected_total_cost():

                cost = 0.
                loc = init_state
                cov = init_cov

                for t in range(optimisation_horizon):
                    mean_full, cov_full = eq_agent.policy.match_moments(loc, cov)

                    loc, cov = eq_agent.match_moments(mean_full, cov_full)

                    loc_, cov_ = cost_transform.match_moments(loc[:1], cov[:1, :1], indices=tf.constant([0]))

                    step_cost = eq_agent.cost.expected_cost(loc_, cov_)

                    cost = cost + step_cost

                cost = cost / tf.cast(optimisation_horizon, dtype=eq_cost.dtype)

                print(f"Current cost: {cost}")

                return cost

            while True:
                loss, converged, diverged = ntfo.minimize(function=expected_total_cost,
                                                          vs=[eq_agent.policy.policy.rbf_locs,
                                                              eq_agent.policy.policy.rbf_log_scales,
                                                              eq_agent.policy.policy.rbf_weights,
                                                              ],
                                                          explicit=False,
                                                          max_iterations=100)

                print(f"Policy loss: {loss},"
                      f"converged: {converged}, "
                      f"diverged: {diverged}")

                if not diverged:
                    break

                else:
                    print("Optimization diverged, restarting optimization!")

        else:
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

                            loc_, cov_ = cost_transform.match_moments(loc[:1], cov[:1, :1], indices=tf.constant([0]))

                            step_cost = eq_agent.cost.expected_cost(loc_, cov_)

                            cost = cost + step_cost

                            step_costs.append(step_cost)

                        cost = cost / tf.cast(optimisation_horizon, dtype=eq_cost.dtype)

                    gradients = tape.gradient(cost, eq_agent.policy.parameters)

                    policy_optimiser.apply_gradients(zip(gradients, eq_agent.policy.parameters))

                    # eq_agent.policy.clip()

                    if tf.abs(cost - prev_loss) < tolerance:
                        print(f"Early convergence!")
                        break

                    prev_loss = cost

                    print(tf.stack(step_costs, axis=0).numpy())

                    env.reset()
                    env.env.env.state = init_state.numpy()[0]
                    true_actions = []
                    true_traj.append(init_state.numpy()[0])

                    true_cost = 0.

                    for step in range(num_steps_per_episode):
                        action = eq_agent.act(state)
                        state, action, next_state = env.step(action.numpy())
                        true_cost = true_cost + eq_cost(tf.convert_to_tensor(2 * tf.abs(tf.sin(next_state[None, :1] / 2))))

                        true_traj.append(next_state)
                        true_actions.append(action)

                    true_traj = np.stack(true_traj, axis=0)
                    true_actions = np.stack(true_actions, axis=0)[:, 0]

                    # Run rollout and plot
                    locs_all = np.stack(locs, axis=0)
                    vars_all = np.stack(covs, axis=0)[:, [0, 1], [0, 1]]
                    print(f'locs_all.shape, covs_all.shape, {locs_all.shape, np.stack(covs, axis=0).shape}')
                    steps = np.arange(locs_all.shape[0])

                    plot_pendulum_rollouts(steps,
                                           true_traj,
                                           locs_all,
                                           vars_all,
                                           policy=eq_agent.policy,
                                           true_actions=true_actions,
                                           plot_path=f'{root_dir}/plots/',
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

        imageio.mimwrite(f'{root_dir}/gifs/pendulum-{episode}.gif', frames)

        eq_agent.save_weights(f"{save_dir}/episode_{episode}/model")
