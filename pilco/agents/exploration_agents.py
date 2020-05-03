from pilco.agents import Agent
from pilco.errors import AgentError
from pilco.utils import chol_update_by_block_lu

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class GPExplorationAgent(Agent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 gp_means,
                 gp_covs,
                 policy,
                 dtype,
                 replay_buffer_limit=None,
                 name='gp_exploration_agent',
                 **kwargs):

        super().__init__(in_state_dim=state_dim,
                         out_state_dim=state_dim,
                         action_dim=action_dim,
                         policy=policy,
                         cost=None,
                         dtype=dtype,
                         replay_buffer_limit=replay_buffer_limit,
                         name=name,
                         **kwargs)

        # Check that means and covariances are GPFlow means and covariances

        # Set GP mean and covariance functions
        self._gp_means = gp_means
        self._gp_covs = gp_covs

        self.state_dim = state_dim
        self._log_noise = tf.Variable(tf.constant(-2., dtype=self.dtype))

        #TODO: ADD MEAN FUNCTIONALITY - CURRENTLY ZERO MEAN


    def act(self, state):
        return self.policy(state)


    def rollout(self, state_dist, num_rollouts, horizon, recondition):

        """
        Input initial state distribution.

        :return:
            Sampled rollouts
            Posterior GP mean and cov (of observed data)
            Posterior GP mean and cov (of true and observed data)
            KL divergence between posterior GPs
        """

        # Compute initial covariance matrices for each GP (S, N, N)
        init_covs = self.gp_covs(self.dynamics_inputs, noise=True)

        # Compute choleskies and tile up for each Monte Carlo rollout (M, S, N, N)
        init_covs_chol = tf.linalg.cholesky(init_covs)
        init_covs_chol = tf.tile(init_covs_chol[None, :, :, :], (num_rollouts, 1, 1, 1))
        covs_chol = init_covs_chol[:]

        # Tile inputs to roll out in parallel (M, S + A, N)
        inputs = tf.transpose(self.dynamics_inputs, (1, 0))
        inputs = tf.tile(inputs[None, :, :], (num_rollouts, 1, 1))

        # Tile outputs to roll out in parallel (M, S, N, 1)
        outputs = tf.transpose(self.dynamics_outputs, (1, 0))
        outputs = tf.tile(outputs[None, :, :, None], (num_rollouts, 1, 1, 1))

        # Sample initial state for every rollout (M, S, 1)
        state = state_dist.sample(num_rollouts)[:, :, None]

        states = [state]

        # Do rollouts
        for h in range(horizon):

            # Compute action from current state (M, A, 1)
            action = self.policy(state[:, :, 0])[:, :, None]

            # Concatenate current state and action (M, S + A, 1)
            state_action = tf.concat([state, action], axis=1)

            # Append state-action to inputs (M, S + A, N + 1)
            inputs = tf.concat([inputs, state_action], axis=2)

            # Get covariance matrix (M, S, N + 1 + H, N + 1 + H)
            K_full = self.gp_covs(x1=tf.transpose(inputs, (0, 2, 1)), noise=True)
            K_full = tf.transpose(K_full, (1, 0, 2, 3))

            # Split into submatrices
            # (M, S, N, N)
            K_data_data_noise = K_full[:, :, :-1, :-1]
            # (M, S, 1, N)
            K_star_data = K_full[:, :, -1:, :-1]
            # (M, S, N, 1)
            K_data_star = K_full[:, :, :-1, -1:]
            # (M, S, 1, 1)
            K_star_star_noise = K_full[:, :, -1:, -1:]

            # Compute mean of next datum (M, S, 1, 1)
            mean = tf.einsum('msij, msjk -> msik',
                             K_star_data,
                             tf.linalg.cholesky_solve(covs_chol, outputs))

            # Compute mean of next datum (M, S, 1, 1)
            var = K_star_star_noise - tf.einsum('msij, msjk -> msik',
                                                K_star_data,
                                                tf.linalg.cholesky_solve(covs_chol, K_data_star))

            # Sample from normal (M, S, 1, 1)
            state_delta = tfd.Normal(loc=mean, scale=(var + 1e-12) ** 0.5).sample()
            state = state + state_delta[:, :, 0]

            states.append(state)

            if recondition:

                # Append new datum to outputs (M, S, N + 1, 1)
                outputs = tf.concat([outputs, state_delta], axis=2)

                # Cholesky of new covariance matrix (M, S, N + 1, N + 1)
                covs_chol = chol_update_by_block_lu(L=covs_chol,
                                                    a=K_data_star,
                                                    b=K_star_star_noise)
            else:
                inputs = inputs[:, :, :-1]


        states = tf.stack(states, axis=-1)

        if recondition:

            K12 = self.gp_covs(x1=tf.transpose(inputs, (0, 2, 1))[:, :-horizon, :],
                               x2=tf.transpose(inputs, (0, 2, 1)), noise=False)
            K21 = tf.transpose(K12, (0, 1, 3, 2))
            K22 = self.gp_covs(x1=tf.transpose(inputs, (0, 2, 1)), noise=False)
            K22 = tf.transpose(K22, (1, 0, 2, 3))
            K22_noise = self.gp_covs(x1=tf.transpose(inputs, (0, 2, 1)), noise=True)
            K22_noise = tf.transpose(K22_noise, (1, 0, 2, 3))
            K11_noise = self.gp_covs(x1=tf.transpose(inputs, (0, 2, 1))[:, :-horizon, :], noise=True)
            K11_noise = tf.transpose(K11_noise, (1, 0, 2, 3))

            mean_1 = tf.einsum('msij, msjk -> msi',
                               K21,
                               tf.linalg.cholesky_solve(init_covs_chol, outputs[:, :, :-horizon, :]))

            Sigma_1 = K22_noise - 1e0 * tf.einsum('msij, msjk -> msik',
                                                   K21,
                                                   tf.linalg.solve(K11_noise, K12))

            mean_2 = tf.einsum('msij, msjk -> msi',
                               K22,
                               tf.linalg.cholesky_solve(covs_chol, outputs[:, :, :, :]))

            Sigma_2 = K22_noise - 1e0 * tf.einsum('msij, msjk -> msik',
                                          K22,
                                          tf.linalg.solve(K22_noise, K22))

            Sigma_1_chol = tf.linalg.cholesky(Sigma_1)
            Sigma_2_chol = tf.linalg.cholesky(Sigma_2)

            normal_1 = tfd.MultivariateNormalTriL(loc=mean_1,
                                                  scale_tril=Sigma_1_chol)

            normal_2 = tfd.MultivariateNormalTriL(loc=mean_2,
                                                  scale_tril=Sigma_2_chol)

            kl_divergence = normal_1.kl_divergence(normal_2)
            kl_divergence = tf.reduce_mean(kl_divergence, axis=0)
            kl_divergence = tf.reduce_sum(kl_divergence)
            kl_divergence = kl_divergence / Sigma_2.shape[-1]

            return inputs, outputs[..., 0], states, kl_divergence

        else:
            return inputs, outputs[..., 0], states



    def gp_covs(self, x1, x2=None, noise=True):

        """
        Computes the covariance between different input locations. If x2 is not
        passed, the covariance between x1 and itself is computed. Supports
        batch calculation of covariances over leading dimensions.

        :param x1: tf.tensor, first tensor of inputs (..., N1, D)
        :param x2: tf.tensor, second tensor of inputs (..., N2, D)
        :param noise: bool, whether to include the diagonal noise
        :return:
            If x1 with shape (..., N1, D) is passed, returns tensor of shape
            (..., N1, N1). If x2 with shape (..., N2, D) is passed in addition
            to x1, returns tf.tensor of shape (..., N1, N2)
        """

        # Check that x1 has 2 or more dimensions
        if tf.rank(x1) <= 1:
            raise AgentError(f'{type(self)}.gp_covs expected x1 to have more'
                             f' than two dimensions, found shape {x1.shape}.')

        # Check that x1 and x2 have the same rank and batch dimensions
        if (x2 is not None) and (x1.shape[:-2] != x2.shape[:-2] or tf.rank(x1) != tf.rank(x2)):
            raise AgentError(f'{type(self)}.gp_covs expected x1 and x2 to'
                             f' have the same number of dimensions, found'
                             f' shapes {x1.shape} and {x2.shape}.')

        batch_dims = len(x1.shape) - 2

        if x2 is None:
            covs = tf.stack([gp_cov(x1) for gp_cov in self._gp_covs])

        else:

            # Flatten over batch dimensions
            x1 = tf.reshape(x1, (-1,) + tuple(x1.shape[-2:]))
            x2 = tf.reshape(x2, (-1,) + tuple(x2.shape[-2:]))

            batch_size = x1.shape[0]

            covs = tf.stack([[gp_cov(x1[i], x2[i]) for gp_cov in self._gp_covs]
                             for i in range(batch_size)])


        if noise:
            eye_shape = batch_dims * (1,) + 2 * (covs.shape[-1],)
            eye = tf.reshape(tf.eye(covs.shape[-1], dtype=self.dtype), eye_shape)

            covs = covs + self.noise ** 2 * eye

        return covs


    @property
    def noise(self):
        return 10 ** self._log_noise


#    def gp_covs(self, x1, x2=none, noise=true):
#
#        if (x2 is not None) and (x1.rank != x2.rank or x1.rank != 2):
#            raise AgentError(f'{type(self)}.gp_covs expected x1 and x2 to'
#                             f' both have two dimensions, found'
#                             f' shapes {x1.shape} and {x2.shape}.')
#
#        elif x1.rank <= 1:
#            raise AgentError(f'{type(self)}.gp_covs expected x1 to have'
#                             f' two or more dimensions, found shape {x1.rank}.')
#
#        if x2 is None:
#            covs = tf.stack([gp_cov(x1) for gp_cov in self._gp_covs])
#
#        else:
#            covs = tf.stack([gp_cov(x1, x2) for gp_cov in self._gp_covs])
#
#
#        if noise:
#
#            eye_shape = (len(x1.shape) - 2) * (1,) + 2 * (covs.shape[-1],)
#            eye = tf.reshape(tf.eye(covs.shape[-1]), eye_shape)
#
#            covs = covs + self.noise ** 2 * eye
#
#        return covs


    def train_dynamics_model(self, **kwargs):
        pass


    def match_delta_moments(self, mean_full, cov_full):
        raise NotImplementedError

