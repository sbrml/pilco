from pilco.policies.policy import Policy

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from not_tf_opt import UnconstrainedVariable

tfd = tfp.distributions


class RBFPolicy(Policy):

    def __init__(self,
                 state_dim,
                 action_dim,
                 num_rbf_features,
                 dtype,
                 name='rbf_policy',
                 **kwargs):
        super().__init__(state_dim=state_dim,
                         action_dim=action_dim,
                         name=name,
                         dtype=dtype,
                         **kwargs)

        # Number of radial basis functions
        self.num_rbf_features = num_rbf_features

        # Set RBF policy locations
        rbf_locs_init = tf.zeros((num_rbf_features, state_dim), dtype=dtype)
        self.rbf_locs = UnconstrainedVariable(rbf_locs_init,
                                              name='rbf_locs',
                                              dtype=dtype)

        # Set RBF policy lengthscales
        rbf_log_scales_init = tf.zeros((1, state_dim), dtype=dtype)
        self.rbf_log_scales = UnconstrainedVariable(rbf_log_scales_init,
                                                    name='rbf_log_scales',
                                                    dtype=dtype)

        # Set RBF policy weights
        rbf_weights_init = tf.zeros((num_rbf_features,), dtype=dtype)
        self.rbf_weights = UnconstrainedVariable(rbf_weights_init,
                                                 name='rbf_weights',
                                                 dtype=dtype)

    @property
    def parameters(self):
        return self.rbf_locs.var, self.rbf_log_scales.var, self.rbf_weights.var

    def reset(self):
        # Sample policy parameters from standard normal
        rbf_theta_locs = tf.random.uniform(shape=(self.num_rbf_features,),
                                           minval=-np.pi,
                                           maxval=np.pi,
                                           dtype=self.dtype)

        rbf_theta_dot_locs = tf.random.uniform(shape=(self.num_rbf_features,),
                                               minval=-8.,
                                               maxval=8.,
                                               dtype=self.dtype)

        self.rbf_locs.assign(tf.stack([rbf_theta_locs, rbf_theta_dot_locs], axis=-1))

        self.rbf_log_scales.assign(self.num_rbf_features ** -0.5 * tf.ones(shape=(1, self.state_dim), dtype=self.dtype))

        rbf_weights_init = tf.random.normal(shape=(self.num_rbf_features,),
                                            mean=0.,
                                            stddev=1.,
                                            dtype=self.dtype)
        self.rbf_weights.assign(rbf_weights_init)

        # for param in [self.rbf_locs, self.rbf_log_scales, self.rbf_weights]:
        #     # TODO: update NTFO
        #     param.assign(tf.random.normal(mean=0,
        #                                   stddev=1,
        #                                   shape=param.var.shape,
        #                                   dtype=self.dtype))

    def match_moments(self, loc, cov):
        # Convert state mean to tensor and reshape to be rank 2
        loc = tf.convert_to_tensor(loc)
        loc = tf.cast(loc, self.dtype)
        loc = tf.reshape(loc, (1, self.state_dim))

        # Convert state covariance to tensor and ensure it's a square matrix
        cov = tf.convert_to_tensor(cov)
        cov = tf.cast(cov, self.dtype)
        cov = tf.reshape(cov, (self.state_dim, self.state_dim))

        # Compute mean_u
        mean_det_coeff = tf.eye(self.state_dim, dtype=self.dtype)
        mean_det_coeff = mean_det_coeff + tf.matmul(cov,
                                                    tf.linalg.diag(1. / self.rbf_scales))
        mean_det_coeff = tf.linalg.det(mean_det_coeff) ** -0.5

        scales_plus_cov = self.rbf_scales_mat + cov
        scales_plus_cov_inv = tf.linalg.inv(scales_plus_cov)

        diff_mui_mus = self.rbf_locs() - loc

        mean_u_quad = tf.einsum('ij, jk, ik -> i',
                                diff_mui_mus,
                                scales_plus_cov_inv,
                                diff_mui_mus)

        exp_mean_u_quad = tf.math.exp(-0.5 * mean_u_quad)
        rbf_comp_mean = mean_det_coeff * exp_mean_u_quad

        mean_u = tf.einsum('i, i ->',
                           self.rbf_weights(),
                           rbf_comp_mean)

        # Compute cov_su
        Q = tf.einsum('ij, jk, lk -> li',
                      cov,
                      scales_plus_cov_inv,
                      self.rbf_locs())

        Q = Q + tf.einsum('ij, jk, lk -> li',
                          self.rbf_scales_mat,
                          scales_plus_cov_inv,
                          loc)

        Q = Q * rbf_comp_mean[:, None]

        cov_su = tf.einsum('i, ij -> j', self.rbf_weights(), Q)

        cov_su = cov_su - mean_u * tf.squeeze(loc)

        # Compute cov_uu
        cov_det_coeff = tf.eye(self.state_dim, dtype=self.dtype)
        cov_det_coeff = cov_det_coeff + 2. * cov / self.rbf_scales
        cov_det_coeff = tf.linalg.det(cov_det_coeff) ** -0.5

        half_scales_plus_cov = 0.5 * self.rbf_scales_mat + cov
        half_scales_plus_cov_inv = tf.linalg.inv(half_scales_plus_cov)

        muij = (self.rbf_locs()[None, :, :] + self.rbf_locs()[:, None, :]) / 2

        diff_muij_mus = muij - loc[None, :, :]

        diff_mui_muj = self.rbf_locs()[None, :, :] - self.rbf_locs()[:, None, :]

        cov_uu_quad = tf.einsum('ijk, kl, ijl -> ij',
                                diff_muij_mus,
                                half_scales_plus_cov_inv,
                                diff_muij_mus)

        cov_uu_quad = cov_uu_quad + 0.5 * tf.einsum('ijk, k, ijk -> ij',
                                                    diff_mui_muj,
                                                    1. / self.rbf_scales[0],
                                                    diff_mui_muj)

        exp_cov_uu_quad = tf.math.exp(-0.5 * cov_uu_quad)

        S = cov_det_coeff * exp_cov_uu_quad

        cov_uu = tf.einsum('i, ij, j ->',
                           self.rbf_weights(),
                           S,
                           self.rbf_weights())

        cov_uu = cov_uu - mean_u ** 2

        cov_full = self.join_covariance_matrices(cov,
                                                 cov_su[:, None],
                                                 cov_uu[None, None])

        mean_full = tf.concat([loc, mean_u[None, None]], axis=1)

        return mean_full, cov_full

    @property
    def rbf_scales(self):
        """
        Returns 1 x state_dim tensor of RBF squared lengthscales.
        """

        return tf.math.exp(self.rbf_log_scales()) ** 2

    @property
    def rbf_scales_mat(self):
        """
        Returns diagonal state_dim x state_dim tensor
        of RBF squared lengthscales.
        """

        rbf_scales = self.rbf_scales[0]

        return tf.linalg.diag(rbf_scales)

    def call(self, state):
        # Convert state to tensor and reshape to be rank 2
        state = tf.convert_to_tensor(state, dtype=self.dtype)
        state = tf.reshape(state, (1, -1))

        # Compute quadratic form and exponentiate for each component
        diff_state_mui = state - self.rbf_locs()
        quad = tf.einsum('ik, lk, ik -> i',
                         diff_state_mui,
                         self.rbf_scales ** -1,
                         diff_state_mui)

        exp_quad = tf.math.exp(-0.5 * quad)

        # RBF output is the weighted sum of rbf components
        rbf = tf.einsum('i, i ->',
                        self.rbf_weights(),
                        exp_quad)

        return rbf
