import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from pilco.policies import RBFPolicy


class MomentMatchingError(Exception):
    pass


def test_mm_rbf_policy():

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
    num_mc_samples = int(1e4)
    state_dim = 2
    action_dim = 1
    num_rbf_features = 10
    mean_tolerance = 2e-2
    cov_tolerance = 2e-2

    # Distribution parameters
    loc = tf.zeros(state_dim, dtype=tf.float32)
    cov = tf.eye(state_dim, dtype=tf.float32)

    # Define RBF policy and reset
    rbf_policy = RBFPolicy(state_dim=state_dim,
                           action_dim=action_dim,
                           num_rbf_features=num_rbf_features,
                           dtype=tf.float32)
    rbf_policy.reset()

    # Match moments across rbf policy
    mm_mean, mm_cov = rbf_policy.match_moments(loc, cov)

    # State distribution to sample from
    state_dist = tfd.MultivariateNormalFullCovariance(loc=loc,
                                                      covariance_matrix=cov)

    # Sample states from distribution over states and pass through policy
    s = [state_dist.sample() for i in range(num_mc_samples)]
    u = [rbf_policy(s_) for s_ in s]

    # Convert states and actions to tensors
    s = tf.convert_to_tensor(s)
    u = tf.convert_to_tensor(u)

    # Concatenate state-action samples to compute the overall mean
    su_samples = tf.concat([s, u[..., None]], axis=-1)

    # Monte Carlo mean and covariance
    mc_mean = tf.reduce_mean(su_samples, axis=0)[None, ...]
    mc_cov = (tf.einsum('ij, ik -> jk', su_samples, su_samples) / su_samples.shape[0])
    mc_cov = mc_cov - (tf.einsum('ij, ik -> jk', mc_mean, mc_mean) / mc_mean.shape[0])

    abs_diff_mean = tf.reduce_max(tf.abs(mm_mean - mc_mean))
    abs_diff_cov = tf.reduce_max(tf.abs(mm_cov - mc_cov))

    if abs_diff_mean > mean_tolerance or abs_diff_cov > cov_tolerance:

        raise MomentMatchingError(f'Moment matching and Monte Carlo answers do not'
                                  f' agree within tolerance for RBFPolicy.')
