from abc import abstractmethod, ABC

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Agent(tf.keras.Model, ABC):

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy,
                 dtype,
                 name='agent',
                 **kwargs):

        super().__init__(name=name, dtype=dtype, **kwargs)

        assert dtype == policy.dtype,                            \
            f'Agent and policy dtypes expected to be the same, ' \
            f'found {dtype} and {policy.dtype}'

        # Set state and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_action_dim = state_dim + action_dim

        # Set instantiated policy
        self.policy = policy

        # Initialise variables to hold observed dynamics data
        inputs_init = tf.zeros((0, state_dim + action_dim), dtype=dtype)
        self.dynamics_inputs = tf.Variable(inputs_init,
                                           shape=(None, state_dim + action_dim),
                                           dtype=dtype,
                                           name='dynamics_inputs')

        outputs_init = tf.zeros((0, state_dim), dtype=dtype)
        self.dynamics_outputs = tf.Variable(outputs_init,
                                            shape=(None, state_dim),
                                            dtype=dtype,
                                            name='dynamics_outputs')


    def act(self, state):
        return self.policy(state)


    def observe(self, state, action, next_state):

        # Convert state, action and next state to tensors with correct dtype
        state = tf.convert_to_tensor(state, dtype=self.dtype)
        state = tf.reshape(state, shape=(1, -1))

        action = tf.convert_to_tensor(action, dtype=self.dtype)
        action = tf.reshape(action, shape=(1, -1))

        next_state = tf.convert_to_tensor(next_state, dtype=self.dtype)
        next_state = tf.reshape(next_state, shape=(1, -1))
        
        # Add observed state and action to the training data
        observed_input = tf.concat([state, action], axis=-1)
        observed_inputs = tf.concat([self.dynamics_inputs, observed_input],
                                    axis=0)
        self.dynamics_inputs.assign(observed_inputs)

        # Add observed next state to the training data
        observed_outputs = tf.concat([self.dynamics_outputs, next_state],
                                     axis=0)
        self.dynamics_outputs.assign(observed_outputs)


    @abstractmethod
    def train_dynamics_model(self, **kwargs):
        pass


    @abstractmethod
    def match_moments(self, mu_su, cov_su):
        pass



class EQGPAgent(Agent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy,
                 dtype,
                 name='agent',
                 **kwargs):

        super().__init__(state_dim=state_dim,
                         action_dim=action_dim,
                         policy=policy,
                         dtype=dtype,
                         name=name,
                         **kwargs)

        # Set EQ covariance parameters: coefficient, scales and noise level
        eq_coeff_init = tf.ones((state_dim,), dtype=dtype)
        self.eq_coeff = tf.Variable(eq_coeff_init, name='eq_coeff', dtype=dtype)

        eq_scales_init = tf.ones((state_dim, state_dim + action_dim),
                                 dtype=dtype)
        self.eq_scales = tf.Variable(eq_scales_init,
                                     name='eq_scales',
                                     dtype=dtype)
        
        eq_noise_coeff_init = tf.ones((state_dim,), dtype=dtype)
        self.eq_noise_coeff = tf.Variable(eq_noise_coeff_init,
                                          name='eq_noise_coeff',
                                          dtype=dtype)


    def match_moments(self, mu_su, cov_su):

        # Reshape mean and covariance
        mu_su = tf.reshape(mu_su, shape=(1, self.state_action_dim))
        cov_su = tf.reshape(cov_su, shape=(self.state_action_dim,
                                           self.state_action_dim))

        # Compute mean
        cov = self.exponentiated_quadratic(self.dynamics_inputs)
        cov = cov + tf.eye(cov.shape[-1])[None, :, :] \
                    * self.eq_noise_coeff[:, None, None]

        cov_inv_output = tf.linalg.solve(cov, tf.transpose(self.dynamics_outputs)[:, :, None])[:, :, 0]

        mean_det_coeff = tf.einsum('kl, glm -> gkm',
                                   cov_su,
                                   tf.linalg.diag(self.eq_scales ** -1))

        mean_det_coeff = mean_det_coeff + \
                         tf.eye(mean_det_coeff.shape[-1])[None, ...]

        mean_det_coeff = tf.linalg.det(mean_det_coeff)

        cov_su_plus_scales = cov_su[None, :, :] + tf.linalg.diag(self.eq_scales)

        nu = self.dynamics_inputs - mu_su

        mean_quad = tf.einsum('is, gsi -> gi', nu, tf.linalg.solve(cov_su_plus_scales, tf.tile(tf.transpose(nu)[None, :, :], (self.state_dim, 1, 1))))

        mean_exp_quad = tf.math.exp(-0.5 * mean_quad)

        print(mean_det_coeff.shape, mean_exp_quad.shape)
        mean_exp_quad = mean_det_coeff[:, None] * mean_exp_quad

        mean = tf.einsum('gi, gi -> g', cov_inv_output, mean_exp_quad)

        return mean


    def exponentiated_quadratic(self, inputs):
        
        diffs = inputs[None, :, :] - inputs[:, None, :]

        quads = tf.einsum('ijd, ijd, gd -> gij',
                          diffs,
                          diffs,
                          self.eq_scales ** -1)

        exp_quad = self.eq_coeff[:, None, None] * tf.math.exp(-0.5 * quads)

        return exp_quad

    
    def train_dynamics_model(self):
        pass





