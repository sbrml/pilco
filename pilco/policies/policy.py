import abc
import tensorflow as tf


class Policy(tf.Module):

    def __init__(self, 
                 state_dim,
                 action_dim,
                 name='policy',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim


    @abc.abstractmethod
    def reset(self):
        pass


    @abc.abstractmethod
    def match_moments(self, state_loc, state_cov, joint_result=True):
        pass


    @abc.abstractmethod
    def call(self, state):
        pass

    def join_covariance_matrices(self, cov_ss, cov_su, cov_uu):

        cov_upper = tf.concat([cov_ss, cov_su], axis=1)
        cov_lower = tf.concat([tf.transpose(cov_su), cov_uu], axis=1)

        cov = tf.concat([cov_upper, cov_lower], axis=0)

        return cov


    def __call__(self, state):
        return self.call(state)
