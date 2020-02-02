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

        # Compute mean
        mean_det_coeff = tf.eye(self.state_dim)
        mean_det_coeff = mean_det_coeff + cov * self.rbf_scales ** -1
        mean_det_coeff = tf.linalg.det(mean_det_coeff) ** -0.5

        scales_plus_cov = self.rbf_scales_mat + cov
        scales_plus_cov_inv = tf.linalg.inv(scales_plus_cov)

        # num_rbf_features x state_dim
        diff_mui_mus = self.rbf_locs - loc

        mean_u_quad = tf.einsum('ij, jk, ik -> i',
                                diff_mui_mus,
                                scales_plus_cov_inv,
                                diff_mui_mus)
        
        exp_mean_u_quad = tf.math.exp(-0.5 * mean_u_quad)
        rbf_comp_mean = mean_det_coeff * exp_mean_u_quad

        mean_u = tf.squeeze(tf.matmul(self.rbf_weights, rbf_comp_mean))

        # Compute cov_su
        Q = tf.einsum('ij, jk, kl -> il',
                      self.rbf_locs,
                      scales_plus_cov_inv,
                      cov)

        Q = Q + tf.einsum('ij, jk, kl -> il',
                          loc,
                          scales_plus_cov_inv,
                          self.rbf_scales_mat) 
        
        Q = Q * rbf_comp_mean[..., None]

        cov_su = tf.matmul(self.rbf_weights, Q) - mean_u * tf.squeeze(loc)

        # Compute cov_uu
        cov_det_coeff = tf.eye(self.state_dim)
        cov_det_coeff = cov_det_coeff + 2. * cov * self.rbf_scales ** -1
        cov_det_coeff = tf.linalg.det(cov_det_coeff) ** -0.5

        half_scales_plus_cov = self.rbf_scales_mat / 2 + cov
        half_scales_plus_cov_inv = tf.linalg.inv(half_scales_plus_cov)

        muij = (self.rbf_locs[:, None, :] + self.rbf_locs[None, :, :]) / 2
        diff_muij_mus = muij - loc[None, ...]
        diff_mui_muj = (self.rbf_locs[:, None, :] - self.rbf_locs[None, :, :])

        cov_uu_quad = tf.einsum('ijk, kl, ijl -> ij',
                                diff_muij_mus,
                                half_scales_plus_cov_inv,
                                diff_muij_mus)

        cov_uu_quad = cov_uu_quad + 0.5 * tf.einsum('ijk, kl, ijl -> ij',
                                                    diff_mui_muj,
                                                    self.rbf_scales_mat,
                                                    diff_mu_muj)

        exp_cov_uu_quad = tf.math.exp(-0.5 * cov_uu_quad)

        S = cov_det_coeff * exp_cov_uu_quad

        cov_uu = tf.einsum('i, ij, j ->',
                           self.rbf_weights,
                           S,
                           self.rbf_weights)

        cov_uu = cov_uu - mean_u ** 2

        return mean_u, cov_su, cov_uu


    @property
    def rbf_scales(self):
        """
        Returns 1 x state_dim tensor of RBF squared lengthscales.
        """

        return tf.math.exp(self.rbf_log_scales) ** 2


    @property
    def rbf_scales_mat(self):
        """
        Returns diagonal state_dim x state_dim tensor
        of RBF squared lengthscales.
        """

        rbf_scales = self.rbf_scales

        return tf.diag(rbf_scales)


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
