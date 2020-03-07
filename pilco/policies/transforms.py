from .policy import Policy

import tensorflow as tf


class SineBoundedActionPolicy(Policy):

    def __init__(self,
                 policy,
                 lower=0.,
                 upper=1.,
                 name="sine_bounded_action_policy",
                 **kwargs):

        super().__init__(state_dim=policy.state_dim,
                         action_dim=policy.action_dim,
                         name=name,
                         dtype=policy.dtype,
                         **kwargs)

        self.policy = policy
        self.lower = tf.cast(lower, dtype=self.policy.dtype)
        self.upper = tf.cast(upper, dtype=self.policy.dtype)

    @property
    def shift(self):
        return (self.upper + self.lower) / 2.

    @property
    def scale(self):
        return (self.upper - self.lower) / 2.

    @property
    def parameters(self):
        return self.policy.parameters

    def reset(self):
        self.policy.reset()

    def match_moments(self, state_loc, state_cov, joint_result=True):
        # We first match the moments through the base policy
        mean, cov = self.policy.match_moments(state_loc, state_cov)

        # Slice base policy statistics
        mean_s = mean[:, :self.state_dim]
        mean_u = mean[:, self.state_dim:]

        cov_ss = cov[:self.state_dim, :self.state_dim]
        cov_uu = cov[self.state_dim:, self.state_dim:]
        cov_su = cov[:self.state_dim, self.state_dim:]

        # Moment match the mean through the sine
        mean_coeff = tf.exp(-0.5 * tf.linalg.diag_part(cov_uu))[None, :]
        mean_u_bounded = tf.sin(mean_u) * mean_coeff

        mean_u_bounded_rescaled = self.shift + self.scale * mean_u_bounded

        # 1 x D
        mean_full_bounded = tf.concat([mean_s, mean_u_bounded_rescaled], axis=1)

        # Calculate the cross-covariance term
        cov_su_bounded = tf.transpose(mean_s) * tf.sin(mean_u) + cov_su * tf.cos(mean_u)
        cov_su_bounded = mean_coeff * cov_su_bounded
        cov_su_bounded = cov_su_bounded - tf.transpose(mean_s) * mean_u_bounded
        cov_su_bounded_rescaled = self.scale * cov_su_bounded

        # Calculate covariance term

        # A x A x A
        c = tf.eye(self.action_dim, dtype=self.dtype)[:, None, :] + tf.eye(self.action_dim, dtype=self.dtype)[None, :, :]
        d = tf.eye(self.action_dim, dtype=self.dtype)[:, None, :] - tf.eye(self.action_dim, dtype=self.dtype)[None, :, :]

        def calculate_cov_terms(v):
            mean_cross = tf.einsum('iu, abu -> ab', mean_u, v)
            cov_cross = tf.einsum('ij, abi, abj -> ab', cov_uu, v, v)

            return tf.cos(mean_cross) * tf.exp(-0.5 * cov_cross)

        cov_uu_bounded = calculate_cov_terms(d) - calculate_cov_terms(c)
        cov_uu_bounded = 0.5 * cov_uu_bounded - tf.transpose(mean_u_bounded) * mean_u_bounded
        cov_uu_bounded_rescaled = cov_uu_bounded * self.scale**2

        cov_full_bounded = self.join_covariance_matrices(cov_ss,
                                                         cov_su_bounded_rescaled,
                                                         cov_uu_bounded_rescaled)

        return mean_full_bounded, cov_full_bounded

    def call(self, state):
        return self.shift + self.scale * tf.sin(self.policy(state))

