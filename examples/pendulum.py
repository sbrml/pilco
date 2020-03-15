import tensorflow as tf

import numpy as np

from pilco.environments import Environment
from pilco.policies import RBFPolicy, SineBoundedActionPolicy
from pilco.costs import EQCost
from pilco.agents import EQGPAgent

import not_tf_opt as ntfo

from tqdm import trange

import matplotlib.pyplot as plt
from sacred import Experiment

import time

import imageio

experiment = Experiment('pendulum-experiment')

@experiment.config
def config():

    # Lengthscale for gaussian cost
    target_scale = np.pi

    # Number of rbf features in policy
    num_rbf_features = 50

    # Subsampling factor
    sub_sampling_factor = 4

    # Number of episodes of random sampling of data
    num_random_episodes = 50

    # Number of steps per random episode
    num_steps_per_random_episode = 1

    # Parameters for agent-environment loops
    optimisation_horizon = 50
    num_optim_steps = 50

    num_episodes = 10
    num_steps_per_episode = 30

    # Number of optimisation steps for dynamics GPs
    num_dynamics_optim_steps = 50

    # Policy learn rate
    policy_lr = 1e0
    scaled_policy_lr = policy_lr / optimisation_horizon

    # Dynamics learn rate
    dynamics_lr = 1e-1

    # Optimsation exit criterion tolerance
    tolerance = 1e-5

        
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
               num_dynamics_optim_steps,
               num_optim_steps,
               num_episodes,
               num_steps_per_episode,
               scaled_policy_lr,
               dynamics_lr,
               tolerance):
    

    dtype = tf.float64

    # Create pendulum environment and reset
    env = Environment(name='Pendulum-v0',
                      sub_sampling_factor=sub_sampling_factor)
    env.reset()

    # Create stuff for our controller
    target_loc = tf.zeros([1, 2])

    eq_cost = EQCost(target_loc=target_loc,
                     target_scale=target_scale,
                     dtype=dtype)

    # Create EQ policy
    eq_policy = RBFPolicy(state_dim=2,
                          action_dim=1,
                          num_rbf_features=num_rbf_features,
                          dtype=dtype)

    eq_policy = SineBoundedActionPolicy(eq_policy,
                                        lower=-2,
                                        upper=2)

    # Create agent
    eq_agent = EQGPAgent(state_dim=2,
                         action_dim=1,
                         policy=eq_policy,
                         cost=eq_cost,
                         dtype=dtype)

    for episode in trange(num_random_episodes):
        
        state = env.reset()
        
        # state = np.array([np.pi, 8]) * (2 * np.random.uniform(size=(2,)) - 1)
        state = np.array([-np.pi, 0.])
        env.env.env.state = state
        
        for step in range(num_steps_per_random_episode):
            
            action = tf.random.uniform(shape=()) * 4. - 2
            state, action, next_state = env.step(action[None].numpy())
            
            eq_agent.observe(state, action, next_state)


    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)
    init_cov = tf.eye(2, dtype=tf.float64)

    policy_optimiser = tf.optimizers.Adam(scaled_policy_lr)
    dynamics_optimiser = tf.optimizers.Adam(dynamics_lr)

    for episode in range(num_episodes):

        print('Optimising policy')
        
        prev_loss = np.inf
        
        eq_agent.set_eq_scales_from_data()
        
        for n in trange(num_dynamics_optim_steps):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(eq_agent.parameters)

                loss = -eq_agent.dynamics_log_marginal()

            gradients = tape.gradient(loss, eq_agent.parameters)
            dynamics_optimiser.apply_gradients(zip(gradients, eq_agent.parameters))

            eq_agent.eq_scales.assign(tf.clip_by_value(eq_agent.eq_scales(), 0., 3.))

            if tf.abs(loss - prev_loss) < tolerance:
                print(f"Early convergence!")
                break

            prev_loss = loss

        evaluate_agent_dynamics(eq_agent, env, 1000, 1, seed=0)
        print(f'{eq_agent.eq_scales().numpy()}')
        
        eq_agent.policy.reset()
        
        prev_loss = np.inf
        with trange(num_optim_steps) as bar:

            for n in bar:

                cost = 0.
                loc = init_state
                cov = init_cov

                with tf.GradientTape(watch_accessed_variables=False) as tape:

                    tape.watch(eq_agent.policy.parameters)

                    for t in range(optimisation_horizon):

                        mean_full, cov_full = eq_agent.policy.match_moments(loc, cov)

                        loc, cov = eq_agent.match_moments(mean_full, cov_full)

                        cost = cost + eq_agent.cost.expected_cost(loc[None, :], cov)

                gradients = tape.gradient(cost, eq_agent.policy.parameters)

                policy_optimiser.apply_gradients(zip(gradients, eq_agent.policy.parameters))

                if tf.abs(cost - prev_loss) < tolerance:
                    print(f"Early convergence!")
                    break
                    
                prev_loss = cost
                
                bar.set_description(f'Cost: {cost.numpy():.3f}')
        
        print(f'Performing episode {episode + 1}:')
        
        env.reset()
        env.env.env.state = np.array([-np.pi, 0.])

        frames = []
        
        for step in trange(num_steps_per_episode):
            
            frames.append(env.env.render(mode='rgb_array'))

            action = eq_agent.act(state)
            state, action, next_state = env.step(action[None].numpy())
            eq_agent.observe(state, action, next_state)

        env.env.close()

        imageio.mimwrite(f'../gifs/pendulum-{episode}.gif', frames)

