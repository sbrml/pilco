from abc import abstractmethod, ABC

import tensorflow as tf
import tensorflow_probability as tfp

from pilco.policies.policy import Policy
from pilco.costs.costs import Cost

tfd = tfp.distributions


class AgentError(Exception):
    pass


class Agent(tf.keras.Model, ABC):

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy,
                 cost,
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
        if not issubclass(policy.__class__, Policy):
            raise AgentError("Policy must be a subclass of pilco.policies.Policy!")
        self.policy = policy

        # Set cost
        if not issubclass(cost.__class__, Cost):
            raise AgentError("Cost must be a subclass of pilco.costs.Cost!")
        self.cost = cost


        # Initialise variables to hold observed dynamics data
        inputs_init = tf.zeros((0, state_dim + action_dim), dtype=dtype)
        self._dynamics_inputs = tf.Variable(inputs_init,
                                            shape=(None, state_dim + action_dim),
                                            dtype=dtype,
                                            name='dynamics_inputs')

        outputs_init = tf.zeros((0, state_dim), dtype=dtype)
        self._dynamics_outputs = tf.Variable(outputs_init,
                                             shape=(None, state_dim),
                                             dtype=dtype,
                                             name='dynamics_outputs')

        # We will store the data statistics in these variables so that they can be easily
        # transformed back and forth
        self.inputs_mean = tf.Variable(tf.zeros((1, state_dim + action_dim), dtype=dtype),
                                       dtype=dtype,
                                       name="inputs_mean")
        self.inputs_std = tf.Variable(tf.ones((1, state_dim + action_dim), dtype=dtype),
                                      dtype=dtype,
                                      name="inputs_std")

        self.outputs_mean = tf.Variable(tf.zeros((1, state_dim), dtype=dtype),
                                        dtype=dtype,
                                        name="outputs_mean")
        self.outputs_std = tf.Variable(tf.ones((1, state_dim), dtype=dtype),
                                       dtype=dtype,
                                       name="outputs_std")

    @property
    def dynamics_inputs(self):
        return self._dynamics_inputs #(self._dynamics_inputs - self.inputs_mean) / self.inputs_std

    @property
    def dynamics_outputs(self):
        return self._dynamics_outputs #(self._dynamics_outputs - self.outputs_mean) / self.outputs_std


    def act(self, state):
        return self.policy(state)


    def observe(self, state, action, next_state, eps=1e-10):

        # Convert state, action and next state to tensors with correct dtype
        state = self._validate_and_convert(state, last_dim=self.state_dim)
        action = self._validate_and_convert(action, last_dim=self.action_dim)
        next_state = self._validate_and_convert(next_state, last_dim=self.state_dim)

        # Add observed state and action to the training data
        observed_input = tf.concat([state, action], axis=-1)
        observed_inputs = tf.concat([self._dynamics_inputs, observed_input],
                                    axis=0)
        self._dynamics_inputs.assign(observed_inputs)

        # Add observed next state to the training data
        observed_outputs = tf.concat([self._dynamics_outputs, next_state],
                                     axis=0)
        self._dynamics_outputs.assign(observed_outputs)


        # Update the observations means and standard deviations
        mean, variance = tf.nn.moments(self._dynamics_inputs, axes=[0], keepdims=True)
        variance = tf.maximum(variance, eps)
        std = tf.math.sqrt(variance)

        self.inputs_mean.assign(mean)
        self.inputs_std.assign(std)

        mean, variance = tf.nn.moments(self._dynamics_outputs, axes=[0], keepdims=True)
        variance = tf.maximum(variance, eps)
        std = tf.math.sqrt(variance)

        self.outputs_mean.assign(mean)
        self.outputs_std.assign(std)



    @abstractmethod
    def train_dynamics_model(self, **kwargs):
        pass


    @abstractmethod
    def match_moments(self, mu_su, cov_su):
        pass


    def _validate_and_convert(self, xs, last_dim):
        """
        Convert xs into rank-2 tensor of the right datatype

        TODO: add validation code
        """

        xs = tf.convert_to_tensor(xs, dtype=self.dtype)
        xs = tf.reshape(xs, [-1, last_dim])

        return xs


class EQGPAgent(Agent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy,
                 cost,
                 dtype,
                 name='agent',
                 **kwargs):

        super().__init__(state_dim=state_dim,
                         action_dim=action_dim,
                         policy=policy,
                         cost=cost,
                         dtype=dtype,
                         name=name,
                         **kwargs)

        # Set EQ covariance parameters: coefficient, scales and noise level
        eq_coeff_init = tf.ones((state_dim,), dtype=dtype)
        self.eq_coeff = tf.Variable(eq_coeff_init, name='eq_coeff', dtype=dtype)

        eq_scales_init = 1e-2 * tf.ones((state_dim, state_dim + action_dim),
                                        dtype=dtype)

        self.eq_scales = tf.Variable(eq_scales_init,
                                     name='eq_scales',
                                     dtype=dtype)

        eq_noise_coeff_init = 1e-4 * tf.ones((state_dim,), dtype=dtype)
        self.eq_noise_coeff = tf.Variable(eq_noise_coeff_init,
                                          name='eq_noise_coeff',
                                          dtype=dtype)


    @property
    def parameters(self):
        return (self.eq_coeff, self.eq_scales, self.eq_noise_coeff)


    def match_moments(self, mean_full, cov_full):

        # Reshape mean and covariance
        mean_full = tf.reshape(mean_full, shape=(1, self.state_action_dim))
        cov_full = tf.reshape(cov_full, shape=(self.state_action_dim,
                                               self.state_action_dim))

        # ----------------------------------------------------------------------
        # Compute mean
        # ----------------------------------------------------------------------

        # S x D x D
        eq_scales_inv = tf.linalg.diag(1. / self.eq_scales)

        # S x D x D
        mean_det_coeff = tf.einsum('kl, glm -> gkm',
                                   cov_full,
                                   eq_scales_inv)

        mean_det_coeff = mean_det_coeff + \
                         tf.eye(mean_det_coeff.shape[-1], dtype=self.dtype)[None, :, :]

        mean_det_coeff = tf.linalg.det(mean_det_coeff) ** -0.5

        mean_det_coeff = self.eq_coeff * mean_det_coeff

        cov_full_plus_scales = cov_full[None, :, :] + tf.linalg.diag(self.eq_scales)

        # N x D
        nu = self.dynamics_inputs - mean_full

        # TODO: Two separate solves for the nu, reuse the solves for cross-cov
        mean_quad = tf.einsum('id, gdi -> gi',
                              nu,
                              tf.linalg.solve(cov_full_plus_scales,
                                              tf.tile(tf.transpose(nu)[None, :, :],
                                                      (self.state_dim, 1, 1))))

        mean_exp_quad = tf.math.exp(-0.5 * mean_quad)

        mean_exp_quad = mean_det_coeff[:, None] * mean_exp_quad

        mean = tf.einsum('gi, gi -> g', self.beta, mean_exp_quad)

        # ----------------------------------------------------------------------
        # Compute covariance
        # ----------------------------------------------------------------------

        # Calculate denominator for EQ coefficient

        # S x S x D x D
        eq_scales_cross_sum = eq_scales_inv[None, :, :, :] + eq_scales_inv[:, None, :, :]


        # S x S x D x D
        R = tf.einsum('ij, abjk -> abik',
                      cov_full,
                      eq_scales_cross_sum)

        # S x S x D x D
        R = R + tf.eye(self.state_action_dim, dtype=self.dtype)[None, None, :, :]

        # S x S
        #R_det_inv_sqrt = tf.linalg.det(R) ** -0.5
        reshaped_R = tf.reshape(R, [-1, self.state_action_dim, self.state_action_dim])

        # Ignore sign, because R will always be positive definite
        _, log_R_det = tf.linalg.slogdet(reshaped_R)
        log_R_det_inv_sqrt = -0.5 * tf.reshape(log_R_det,
                                               [self.state_dim, self.state_dim])

        # Calculate numerator for EQ coeffieicent

        # S x 1 x N
        log_k_data_mu = self.exponentiated_quadratic(mean_full,
                                                     self.dynamics_inputs,
                                                     log=True)

        # S x S x N x N
        log_k_ab = log_k_data_mu[None, :, :, :] + log_k_data_mu[:, :, :, None]

        # Calculate exponentiated quadratic

        # S x N x D
        eq_scale_times_nu = tf.einsum('sij, nj -> sni',
                                      eq_scales_inv,
                                      nu)

        # S x S x N x N x D
        z = eq_scale_times_nu[:, None, :, None, :] + eq_scale_times_nu[None, :, None, :, :]

        # S x S x N x N x D
        cov_full_times_z = tf.einsum('ij, abnmj -> abnmi',
                                   cov_full,
                                   z)

        # S x S x N x N x D x D
        R_tiled = tf.tile(R[:, :, None, None, :, :],
                          [1, 1, self.num_datapoints, self.num_datapoints, 1, 1])

        # S x S x N x N x D
        R_inv_cov_full_z = tf.linalg.solve(R_tiled,
                                           cov_full_times_z[:, :, :, :, :, None])[:, :, :, :, :, 0]

        # S x S x N x N
        cov_quad = tf.einsum('abnmi, abnmi -> abnm',
                             z,
                             R_inv_cov_full_z)
        cov_quad = 0.5 * cov_quad

        # Put coefficient and EQ together

        # S x S x N x N
        log_Q = log_k_ab + cov_quad + log_R_det_inv_sqrt[:, :, None, None]
        Q = tf.math.exp(log_Q)

        # Calculate diagonal covariance terms

        # Select diagonal entries of Q
        Q_diag_indices = tf.tile(tf.range(self.state_dim)[:, None], [1, 2])

        # S x N x N
        Q_diag = tf.gather_nd(Q, Q_diag_indices)

        # S x N x N
        data_cov_inv_times_Q_diag = tf.linalg.solve(self.data_covariance, Q_diag)

        # S
        expected_var = tf.linalg.trace(data_cov_inv_times_Q_diag)
        expected_var = self.eq_coeff - expected_var

        # Calculate general covariance terms
        # S x N
        beta = self.beta

        # S x S
        cov = tf.einsum('ai, bj, abij -> ab',
                        beta,
                        beta,
                        Q)

        mean_times_mean = mean[:, None] * mean[None, :]

        cov = cov - mean_times_mean

        cov = cov + tf.linalg.diag(expected_var)

        # Compute Cov[x, Δ]
        mean_full_tiled = tf.tile(tf.transpose(mean_full)[None, :, :],
                                  (self.state_dim, 1, 1))

        dynamics_inputs_tiled = tf.tile(tf.transpose(self.dynamics_inputs)[None, :, :],
                                        (self.state_dim, 1, 1))

        # A = cov_full + scales
        # G x D x 1
        A_inv_times_mean_full = tf.linalg.solve(cov_full_plus_scales,
                                                mean_full_tiled)

        # G x D x N
        A_inv_times_dynamics_inputs = tf.linalg.solve(cov_full_plus_scales,
                                                      dynamics_inputs_tiled)

        # G x D x 1
        cross_cov_mu = self.eq_scales[:, :, None] * A_inv_times_mean_full

        # G x D x N
        cross_cov_mu = cross_cov_mu + tf.einsum('ij, gjk -> gik',
                                                cov_full,
                                                A_inv_times_dynamics_inputs)

        # D x G
        cross_cov = tf.einsum('gk, gdk, gk -> dg',
                              mean_exp_quad,
                              cross_cov_mu,
                              beta)

        # S x G
        cross_cov_s = cross_cov[:self.state_dim, :]
        cross_cov_mean_prod = tf.transpose(mean_full[:, :self.state_dim]) * mean[None, :]
        cross_cov_s = cross_cov_s - cross_cov_mean_prod

        # Calcuate successor mean and covariance
        mean = mean + mean_full[0, :self.state_dim]

        cov = cov + cov_full[:self.state_dim, :self.state_dim]
        cov = cov + cross_cov_s + tf.transpose(cross_cov_s)

        return mean, cov


    @property
    def num_datapoints(self):
        return self._dynamics_inputs.value().shape[0]


    @property
    def beta(self):

        # S x N x N
        cov = self.data_covariance

        # S x N x 1
        dynamics_outputs_T = tf.transpose(self.dynamics_outputs)[:, :, None]

        # S x N
        cov_inv_output = tf.linalg.solve(cov, dynamics_outputs_T)[:, :, 0]

        return cov_inv_output


    @property
    def data_covariance(self):

        dynamics_inputs = self.dynamics_inputs

        K = self.exponentiated_quadratic(dynamics_inputs,
                                         dynamics_inputs)

        noise = self.eq_noise_coeff[:, None, None] * tf.eye(K.shape[-1],
                                                            dtype=self.dtype)[None, :, :]

        return K + noise

    def exponentiated_quadratic(self, x, x_, log=False):
        """
        x - N x D tensor
        x_ - M x D tensor

        where N, M are the batch dimensions and D is the dimensionality

        returns S x N x M
        """

        # N x M x D tensor
        diffs = x[:, None, :] - x_[None, :, :]

        quads = tf.einsum('nmd, nmd, gd -> gnm',
                          diffs,
                          diffs,
                          1. / self.eq_scales)

        if log:
            return tf.math.log(self.eq_coeff[:, None, None]) - 0.5 * quads
        else:
            return self.eq_coeff[:, None, None] * tf.math.exp(-0.5 * quads)


    def gp_posterior_predictive(self, x_star):
        """
        x_star - K x D tensor of K inputs of dimensionality D
        """

        x_star = self._validate_and_convert(x_star, self.state_action_dim)

        # S x N x K
        k_star = self.exponentiated_quadratic(self.dynamics_inputs,
                                              x_star)

        # S x K x K
        k_star_star = self.exponentiated_quadratic(x_star,
                                                   x_star)

        # S x N x N
        K_plus_noise = self.data_covariance

        # S x N
        pred_mean = tf.einsum('snk, sn -> ks',
                              k_star,
                              self.beta)

        # S x N x K
        cov_inv_k = tf.linalg.solve(K_plus_noise,
                                    k_star)

        # S x K x K
        k_cov_inv_k = tf.einsum('snk, snl -> skl',
                                 k_star,
                                 cov_inv_k)

        pred_cov = k_star_star - k_cov_inv_k

        # Put data back in original data domian
        # pred_mean = pred_mean * self.outputs_std + self.outputs_mean
        # pred_cov = pred_cov * self.outputs_std * self.outputs_std

        return pred_mean, pred_cov


    def get_cost(self, state_loc, state_cov, horizon):

        self.policy.match_moments()

    def optimize_policy(self):
        pass

    def train_dynamics_model(self):
        pass
