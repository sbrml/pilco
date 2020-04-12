from pilco.policies.policy import Policy

import tensorflow as tf


class TransformedPolicy(Policy):

    def __init__(self,
                 policy,
                 transform,
                 name="sine_bounded_action_policy",
                 **kwargs):

        super().__init__(state_dim=policy.state_dim,
                         action_dim=policy.action_dim,
                         name=name,
                         dtype=policy.dtype,
                         **kwargs)

        self.policy = policy
        self.transform = transform

    @property
    def parameters(self):
        return self.policy.parameters

    @property
    def action_indices(self):
        return tf.range(self.state_dim, self.state_dim + self.action_dim)

    def reset(self):
        self.policy.reset()

    def match_moments(self, state_loc, state_cov, joint_result=True):

        # We first match the moments through the base policy
        loc, cov = self.policy.match_moments(state_loc, state_cov)
        loc, cov = self.transform.match_moments(loc=loc,
                                                      cov=cov,
                                                      indices=self.action_indices)

        return loc, cov

    def call(self, state):
        # Convert state to tensor and reshape to be rank 2
        state = tf.convert_to_tensor(state)
        state = tf.cast(state, self.dtype)

        # N x D
        state = tf.reshape(state, (-1, self.state_dim))

        # N x (D + A)
        full_vec = tf.concat([state, self.policy(state)], axis=1)

        return self.transform(full_vec, indices=self.action_indices)[:, self.state_dim:]
