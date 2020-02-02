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
    def match_moments(self, state_loc, state_cov):
        pass


    @abc.abstractmethod
    def call(self, state):
        pass


    def __call__(self, state):
        return self.call(state)
