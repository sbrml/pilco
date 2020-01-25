import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Agent(tf.keras.Model):

    def __init__(self,
                 state_dim,
                 action_dim,
                 num_rbf_features,
                 name='agent',
                 **kwargs):

        super(Agent, self).__init__(name=name, **kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_rbf_features = num_rbf_features

        # Set RBF policy locations
        self.rbf_locs = tf.Variable(tf.zeros(num_rbf_features, state_dim),
                                    name='rbf_locs')

        # Set RBF policy lengthscales
        self.rbf_log_scales = tf.Variable(tf.zeros(num_rbf_features, state_dim),
                                          name='rbf_log_scales')

        # Set RBF policy weights
        self.rbf_weights = tf.Variable(tf.zeros(num_rbf_features),
                                       name='rbf_weights')

    def sample_policy_parameters(self):

        # Locations, scales and weights have same shapes
        shape = (num_rbf_features,))
          
        # Assign rbf locations from standard normal
        self.rbf_locs.assign(tf.random.normal(mean=0.,
                                              stddev=1.,
                                              shape=shape))

        # Assign rbf log_scales from standard normal
        self.rbf_log_scales.assign(tf.random.normal(mean=0.,
                                                    stddev=1.,
                                                    shape=shape))

        # Assign rbf weights from standard normal
        self.rbf_weights.assign(tf.random.normal(mean=0.,
                                                 stddev=1.,
                                                 shape=shape))


    def act(self, state):

        state = tf.conver_to_tensor(state, dtype=tf.float32)
        
        quad = state - self.rbf_locs)


    def rbf(self, state):
