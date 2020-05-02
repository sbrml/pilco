from pilco.policies.policy import Policy

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from not_tf_opt import UnconstrainedVariable

tfd = tfp.distributions


class EQPolicy(Policy):

    def __init__(self,
                 state_dim,
                 action_dim,
                 num_eq_features,
                 dtype,
                 name='eq_policy',
                 **kwargs):

        super().__init__(state_dim=state_dim,
                         action_dim=action_dim,
                         name=name,
                         dtype=dtype,
                         **kwargs)

        # Number of radial basis functions
        self.num_eq_features = num_eq_features

        # Set RBF policy locations
        eq_locs_init = tf.zeros((num_eq_features, state_dim), dtype=dtype)
        self.eq_locs = UnconstrainedVariable(eq_locs_init,
                                             name='eq_locs',
                                             dtype=dtype)

        # Set RBF policy lengthscales
        eq_log_scales_init = tf.zeros((1, state_dim), dtype=dtype)
        self.eq_log_scales = UnconstrainedVariable(eq_log_scales_init,
                                                    name='eq_log_scales',
                                                    dtype=dtype)

        # Set RBF policy weights
        eq_weights_init = tf.zeros((num_eq_features,), dtype=dtype)
        self.eq_weights = UnconstrainedVariable(eq_weights_init,
                                                name='eq_weights',
                                                dtype=dtype)

    @property
    def parameters(self):
        return self.eq_locs.var, self.eq_log_scales.var, self.eq_weights.var


    def reset(self):

        for param in [self.eq_locs, self.eq_log_scales, self.eq_weights]:

            tensor = tf.random.normal(mean=0.,
                                      stddev=1.,
                                      shape=param.var.shape,
                                      dtype=self.dtype)
            param.assign(tensor)


    def match_moments(self, loc, cov):

        # Convert state mean to tensor and reshape to be rank 2
        loc = tf.convert_to_tensor(loc)
        loc = tf.cast(loc, self.dtype)
        loc = tf.reshape(loc, (1, self.state_dim))

        # Convert state covariance to tensor and ensure it's square
        cov = tf.convert_to_tensor(cov)
        cov = tf.cast(cov, self.dtype)
        cov = tf.reshape(cov, (self.state_dim, self.state_dim))

        # Compute mean_u
        mean_det_coeff = tf.eye(self.state_dim, dtype=self.dtype)
        mean_det_coeff = mean_det_coeff + tf.matmul(cov,
                                                    tf.linalg.diag(1. / self.eq_scales))
        mean_det_coeff = tf.linalg.det(mean_det_coeff) ** -0.5

        scales_plus_cov = self.eq_scales_mat + cov
        scales_plus_cov_inv = tf.linalg.inv(scales_plus_cov)

        diff_mui_mus = self.eq_locs() - loc

        mean_u_quad = tf.einsum('ij, jk, ik -> i',
                                diff_mui_mus,
                                scales_plus_cov_inv,
                                diff_mui_mus)

        exp_mean_u_quad = tf.math.exp(-0.5 * mean_u_quad)
        eq_comp_mean = mean_det_coeff * exp_mean_u_quad

        mean_u = tf.einsum('i, i ->',
                           self.eq_weights(),
                           eq_comp_mean)

        # Compute cov_su
        Q = tf.einsum('ij, jk, lk -> li',
                      cov,
                      scales_plus_cov_inv,
                      self.eq_locs())

        Q = Q + tf.einsum('ij, jk, lk -> li',
                          self.eq_scales_mat,
                          scales_plus_cov_inv,
                          loc)

        Q = Q * eq_comp_mean[:, None]

        cov_su = tf.einsum('i, ij -> j', self.eq_weights(), Q)

        cov_su = cov_su - mean_u * tf.squeeze(loc)

        # Compute cov_uu
        cov_det_coeff = tf.eye(self.state_dim, dtype=self.dtype)
        cov_det_coeff = cov_det_coeff + 2. * cov / self.eq_scales
        cov_det_coeff = tf.linalg.det(cov_det_coeff) ** -0.5

        half_scales_plus_cov = 0.5 * self.eq_scales_mat + cov
        half_scales_plus_cov_inv = tf.linalg.inv(half_scales_plus_cov)

        muij = (self.eq_locs()[None, :, :] + self.eq_locs()[:, None, :]) / 2

        diff_muij_mus = muij - loc[None, :, :]

        diff_mui_muj = self.eq_locs()[None, :, :] - self.eq_locs()[:, None, :]

        cov_uu_quad = tf.einsum('ijk, kl, ijl -> ij',
                                diff_muij_mus,
                                half_scales_plus_cov_inv,
                                diff_muij_mus)

        cov_uu_quad = cov_uu_quad + 0.5 * tf.einsum('ijk, k, ijk -> ij',
                                                    diff_mui_muj,
                                                    1. / self.eq_scales[0],
                                                    diff_mui_muj)

        exp_cov_uu_quad = tf.math.exp(-0.5 * cov_uu_quad)

        S = cov_det_coeff * exp_cov_uu_quad

        cov_uu = tf.einsum('i, ij, j ->',
                           self.eq_weights(),
                           S,
                           self.eq_weights())

        cov_uu = cov_uu - mean_u ** 2

        cov_full = self.join_covariance_matrices(cov,
                                                 cov_su[:, None],
                                                 cov_uu[None, None])

        mean_full = tf.concat([loc, mean_u[None, None]], axis=1)

        return mean_full, cov_full


    @property
    def eq_scales(self):
        """
        Returns 1 x state_dim tensor of RBF squared lengthscales.
        """

        return tf.math.exp(self.eq_log_scales()) ** 2


    @property
    def eq_scales_mat(self):
        """
        Returns diagonal state_dim x state_dim tensor
        of RBF squared lengthscales.
        """

        eq_scales = self.eq_scales[0]

        return tf.linalg.diag(eq_scales)


    def call(self, state):

        # Convert state to tensor and reshape to be rank 2
        state = tf.convert_to_tensor(state, dtype=self.dtype)
        state = tf.reshape(state, (1, -1))

        # Compute quadratic form and exponentiate for each component
        diff_state_mui = state - self.eq_locs()
        quad = tf.einsum('ik, lk, ik -> i',
                         diff_state_mui,
                         self.eq_scales ** -1,
                         diff_state_mui)

        exp_quad = tf.math.exp(-0.5 * quad)

        # RBF output is the weighted sum of eq components
        eq = tf.einsum('i, i ->',
                        self.eq_weights(),
                        exp_quad)

        return eq



class PendulumEQPolicy(EQPolicy):


    def __init__(self,
                 num_eq_features,
                 dtype,
                 name='pendulum_eq_policy',
                 **kwargs):

        super().__init__(state_dim=2,
                         action_dim=1,
                         num_eq_features=num_eq_features,
                         dtype=dtype,
                         **kwargs)


    def reset(self):

        """
        Resets policy parameters: location, scale and weights of EQ features.
        :return:
        """

        # Locations of EQ scales along the angle dimension
        eq_theta_locs = tf.random.normal(shape=(self.num_eq_features,),
                                         mean=-np.pi,
                                         stddev=2 * np.pi,
                                         dtype=self.dtype)

        # Locations of EQ scales along the angular velocity dimension
        eq_theta_dot_locs = tf.random.normal(shape=(self.num_eq_features,),
                                             mean=0.,
                                             stddev=4.,
                                             dtype=self.dtype)

        # Concatenate components of EQ locations and assign
        eq_locs = tf.stack([eq_theta_locs,
                            eq_theta_dot_locs], axis=-1)
        self.eq_locs.assign(eq_locs)

        # Log-scales of EQ components (shared across all EQ components)
        eq_log_scales = self.num_eq_features ** -0.5 * tf.ones(shape=(1, self.state_dim), dtype=self.dtype)
        self.eq_log_scales.assign(eq_log_scales)

        # Weight coefficients for each EQ component
        eq_weights_init = tf.random.normal(shape=(self.num_eq_features,),
                                           mean=0.,
                                           stddev=0.3,
                                           dtype=self.dtype)
        self.eq_weights.assign(eq_weights_init)


class BatchedEQPolicy(EQPolicy):

    def __init__(self,
                 state_dim,
                 action_dim,
                 num_eq_features,
                 dtype,
                 name='eq_policy',
                 **kwargs):


        super().__init__(state_dim=state_dim,
                         action_dim=action_dim,
                         num_eq_features=num_eq_features,
                         dtype=dtype)


    def call(self, state):

        """
        :param state: (M, S)
        :return:
        """

        # Convert state to tensor and reshape to be rank 2
        state = tf.convert_to_tensor(state, dtype=self.dtype)

        # Compute quadratic form and exponentiate for each component
        diff_state_mui = state[:, None, :] - self.eq_locs()[None, :, :]
        quad = tf.einsum('mik, lk, mik -> mi',
                         diff_state_mui,
                         self.eq_scales ** -1,
                         diff_state_mui)

        exp_quad = tf.math.exp(-0.5 * quad)

        # RBF output is the weighted sum of eq components
        eq = tf.einsum('i, mi -> m',
                        self.eq_weights(),
                        exp_quad)

        eq = eq[:, None]

        return eq