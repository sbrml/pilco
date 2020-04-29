import itertools

from pilco.errors import *
from pilco.policies import EQPolicy

from pilco.utils import assert_near

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def test_mm_eq_policy():

    """
    Test RBFPolicy.moment_matching works correctly, by comparing it against
    Monte Carlo integration. Given an initial distribution of the state s,
    computes the mean and covariance of the vector [s^T, u^T]^T, where
    u = rbf_policy(s), using both methods and compares the results.

    :return:
    """

    # Set random seed
    tf.random.set_seed(0)

    # Number of MC samples
    num_mc_samples = int(5e4)

    # Tolerances for mean and covariance estimates
    mean_tolerance = 1e-1
    cov_tolerance = 2e-1

    state_dims = [1, 5, 10]
    action_dims = [1, 5, 10]
    nums_eq_features = [1, 10, 100]

    params = [state_dims,
              action_dims,
              nums_eq_features]

    # Test different state-action dims
    for state_dim, action_dim, num_eq_features in itertools.product(*params):

        # Distribution parameters
        loc = tf.zeros(state_dim, dtype=tf.float32)
        cov = tf.eye(state_dim, dtype=tf.float32)

        # Define RBF policy and reset
        rbf_policy = EQPolicy(state_dim=state_dim,
                              action_dim=action_dim,
                              num_eq_features=num_eq_features,
                              dtype=tf.float32)
        rbf_policy.reset()

        # Match moments across rbf policy
        mm_mean, mm_cov = rbf_policy.match_moments(loc, cov)

        # State distribution to sample from
        state_dist = tfd.MultivariateNormalFullCovariance(loc=loc,
                                                          covariance_matrix=cov)

        # Sample states from distribution over states and pass through policy
        s = state_dist.sample(num_mc_samples)
        u = tf.stack([rbf_policy(s_) for s_ in s], axis=0)

        # Concatenate state-action samples to compute the overall mean
        su_samples = tf.concat([s, u[..., None]], axis=-1)

        # Monte Carlo mean and covariance
        mc_mean = tf.reduce_mean(su_samples, axis=0)[None, ...]
        mc_cov = (tf.einsum('ij, ik -> jk', su_samples, su_samples) / su_samples.shape[0])
        mc_cov = mc_cov - (tf.einsum('ij, ik -> jk', mc_mean, mc_mean) / mc_mean.shape[0])

        print(state_dim, action_dim, num_eq_features)
        print(tf.reduce_max(tf.abs(mm_mean - mc_mean)))
        print(tf.reduce_max(tf.abs(mm_cov - mc_cov)))
        # Assert error if MM and MC answers are not close
        assert_near(mm_mean, mc_mean, atol=mean_tolerance, rtol=None)
        assert_near(mm_cov, mc_cov, atol=cov_tolerance, rtol=None)

test_mm_eq_policy()
