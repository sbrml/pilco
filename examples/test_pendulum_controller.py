import tensorflow as tf
import numpy as np

import datetime
from tqdm import trange

from pilco.environments import Environment
from pilco.policies import Policy, RBFPolicy, TransformedPolicy
from pilco.transforms import SineTransform, SineTransformWithPhase, AbsoluteValueTransform, \
    ChainedMomentMatchingTransfom, AffineTransform
from pilco.costs import EQCost
from pilco.agents import EQGPAgent

import matplotlib.pyplot as plt
from sacred import Experiment

import imageio

experiment = Experiment('test-pendulum-controller')


@experiment.config
def config():
    # Lengthscale for gaussian cost
    target_scale = [[1.]]

    use_lbfgs = True

    agent_replay_buffer_limit = 250

    # Number of rbf features in policy
    num_rbf_features = 50

    # Subsampling factor
    sub_sampling_factor = 2

    root_dir = "/Users/gergelyflamich/Documents/sbrml/pilco/"

    agent_path = f"{root_dir}/saved_agents/20200421-112529/episode_2"


@experiment.automain
def experiment(sub_sampling_factor,
               target_scale,
               num_rbf_features,
               agent_replay_buffer_limit,
               agent_path,
               root_dir):
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


    eq_agent.load_weights(f"{agent_path}/model")

    print("Agent loaded!")


    init_state = tf.constant([[-np.pi, 0.]], dtype=tf.float64)

    env.reset()
    env.env.env.state = init_state.numpy()[0]
    state = init_state.numpy()[0]

    frames = []

    for step in trange(100):
        frames.append(env.env.render(mode='rgb_array'))

        action = eq_agent.act(state)
        state, action, next_state = env.step(action.numpy())
        eq_agent.observe(state, action, next_state)

    env.env.close()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    imageio.mimwrite(f'{root_dir}/gifs/pendulum-test-{current_time}.gif', frames)

