import tensorflow as tf

import datetime
import numpy as np

from pilco.environments import Environment
from pilco.policies import PendulumEQPolicy, TransformedPolicy
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

    # Lengthscale for gaussian cost
    target_scale = [[1.]]

    agent_replay_buffer_limit = 240

    # Number of eq features in policy
    num_eq_features = 50

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
               target_scale,
               num_eq_features,
               optimisation_horizon,
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

    eq_cost = EQCost(target_loc=target_loc,
                     target_scale=target_scale,
                     target_dim=1,
                     dtype=dtype)

    eq_policy = TransformedPolicy(policy=PendulumEQPolicy(num_eq_features=num_eq_features,
                                                          dtype=dtype),
                                  transform=SineTransform(-1.4, 1.4))

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

        state = np.array([-np.pi, 0.])
        env.env.env.state = state

        for step in range(num_steps_per_random_episode):
            # action = (tf.random.uniform(shape=()) * 4. - 2)[None]
            action = eq_agent.policy(state)
            state, action, next_state = env.step(action.numpy())

            eq_agent.observe(state, action, next_state)

    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)
    init_cov = 1e-8 * tf.eye(2, dtype=tf.float64)

    for episode in range(num_episodes):

        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')

        print('Optimising dynamics')

        if episode > -1:
            # Create variables
            best_eq_scales = eq_agent.eq_scales()
            best_eq_coeff = eq_agent.eq_coeff()
            best_eq_noise_coeff = eq_agent.eq_noise_coeff()

            best_loss = np.inf

            idx = 0
            while idx < dynamics_optimisation_restarts:

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

                if diverged:
                    print("Dynamics optimization diverged, restarting!")
                    continue

                else:
                    idx += 1

                if loss < best_loss:
                    best_loss = loss

                    best_eq_scales = eq_agent.eq_scales()
                    best_eq_coeff = eq_agent.eq_coeff()
                    best_eq_noise_coeff = eq_agent.eq_noise_coeff()

            # Assign best parameters
            eq_agent.eq_coeff.assign(best_eq_coeff)
            print(best_eq_noise_coeff)
            eq_agent.eq_noise_coeff.assign(best_eq_noise_coeff)
            eq_agent.eq_scales.assign(best_eq_scales)

        else:
            eq_agent.set_eq_scales_from_data()


        print('\nEvaluating dynamics before optimisation')
        print(f'Length scales {eq_agent.eq_scales().numpy()}')
        print(f'Signal amplitude {eq_agent.eq_coeff().numpy()}')
        print(f'Noise amplitude {eq_agent.eq_noise_coeff().numpy()}')

        eq_agent.policy.reset()

        def expected_total_cost():

            cost = 0.
            loc = init_state
            cov = init_cov

            locs = [loc[0].numpy()]
            covs = [cov.numpy()]

            step_costs = []

            for t in range(optimisation_horizon):
                mean_full, cov_full = eq_agent.policy.match_moments(loc, cov)

                loc, cov = eq_agent.match_moments(mean_full, cov_full)
                locs.append(loc.numpy())
                covs.append(cov.numpy())

                loc_, cov_ = cost_transform.match_moments(loc[:1], cov[:1, :1], indices=tf.constant([0]))
                step_cost = eq_agent.cost.expected_cost(loc_, cov_)

                step_costs.append(step_cost)

                cost = cost + step_cost

            cost = cost / tf.cast(optimisation_horizon, dtype=eq_cost.dtype)

            print(tf.stack(step_costs, axis=0).numpy())

            env.reset()
            env.env.env.state = init_state.numpy()[0]
            state = init_state.numpy()[0]

            true_traj = []
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

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_pendulum_rollouts(steps,
                                   true_traj,
                                   locs_all,
                                   vars_all,
                                   policy=eq_agent.policy,
                                   true_actions=true_actions,
                                   plot_path=f'{root_dir}/plots/',
                                   plot_prefix=f'optim-{episode}-{current_time}')

            true_cost = true_cost / num_steps_per_episode
            print(f"Current cost: {cost}, true cost: {true_cost}")

            return cost

        while True:
            loss, converged, diverged = ntfo.minimize(function=expected_total_cost,
                                                      vs=[eq_agent.policy.policy.eq_locs,
                                                          eq_agent.policy.policy.eq_log_scales,
                                                          eq_agent.policy.policy.eq_weights,
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

            action = eq_agent.act(state)
            state, action, next_state = env.step(action.numpy())
            eq_agent.observe(state, action, next_state)

        env.env.close()

        imageio.mimwrite(f'{root_dir}/gifs/pendulum-{episode}.gif', frames)

        eq_agent.save_weights(f"{root_dir}/saved_agents/{current_time}/episode_{episode}/model")
