from pilco.policies.policy import Policy

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class RBFPolicy(Policy):

    def __init__(self,
                 state_dim,
                 action_dim,
                 num_rbf_features,
                 name='rbf_policy',
                 **kwargs):

        super(RBFPolicy, self).__init__(state_dim=state_dim,
                                        action_dim=action_dim,
                                        name=name,
                                        **kwargs)

        # Number of radial basis functions
        self.num_rbf_features = num_rbf_features

        # Set RBF policy locations
        self.rbf_locs = tf.Variable(tf.zeros(num_rbf_features, state_dim),
                                    name='rbf_locs')

        # Set RBF policy lengthscales
        self.rbf_log_scales = tf.Variable(tf.zeros(1, state_dim),
                                          name='rbf_log_scales')

        # Set RBF policy weights
        self.rbf_weights = tf.Variable(tf.zeros(num_rbf_features),
                                       name='rbf_weights')


    def reset(self):

        # Sample policy parameters from standard normal
        for param in [self.rbf_locs, self.rbf_log_scales, self.rbf_weights]:

            param.assign(tf.random.normal(mean=0, stddev=1, shape=param.shape))


    def match_moments(self, loc, cov):
          
        # Convert state mean to tensor and reshape to be rank 2
        loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        loc = tf.reshape(loc, (1, self.state_dim))

        # Convert state covariance to tensor and ensure it's a square matrix
        cov = tf.convert_to_tensor(cov, dtype=tf.float32)
        cov = tf.reshape(cov, (self.state_dim, self.state_dim))

        # COMMON
        rbf_scales_inv = 1. / self.rbf_scales

        # Selecting rbf_scales[0] makes it 1d for diag to work properly
        cov_plus_scales_2 = cov + tf.linalg.diag(self.rbf_scales[0] ** 2)
        cov_plus_scales_2_inv = tf.linalg.inv(cov_plus_scales_2)

        # MEAN
        A = state_cov * rbf_scales_inv ** 2 + tf.eye(self.state_dim)
        A_det = tf.linalg.det(A)
        
        diff = tf.transpose(loc - self.rbf_locs)
        quad_form = diff * tf.matmul(cov_plus_scales_2_inv, diff)
        quad_form = tf.reduce_sum(quad_form, axis=-1)
        exp_quads = tf.math.exp(-0.5 * quad_form)

        rbf_components = A_det ** -0.5 * exp_quads
        mean = tf.matmul(self.rbf_weights, rbf_components)

        return mean


    @property
    def rbf_scales(self):
        return tf.math.exp(self.rbf_log_scales)


    def call(self, state):
        
        # Convert state to tensor and reshape to be rank 2
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, (1, -1))

        # Compute quadratic form and exponentiate for each component
        quad = ((state - self.rbf_locs) / self.rbf_scales) ** 2
        exp_quads = tf.math.exp(-0.5 * tf.reduce_sum(quad, axis=-1))

        # RBF output is the weighted sum of rbf components
        rbf = tf.matmul(self.rbf_weights, exp_quads)

        return rbf
