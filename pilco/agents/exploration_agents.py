import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from pilco.agents import Agent

class GPExplorationAgent(Agent):

    def __init__(self,
                 state_dim,
                 action_dim,
                 gp_means,
                 gp_covs,
                 policy,
                 dtype,
                 replay_buffer_limit=None,
                 name='agent',
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


    def act(self, state):
        return self.policy(state)


    def rollout(self, state_dist, num_rollouts, horizon):
        """
        Input initial state distribution.
        :return:
            Sampled rollouts
            Posterior GP mean and cov (of observed data)
            Posterior GP mean and cov (of true and observed data)
            KL divergence between posterior GPs
        """

        # Compute initial covariance matrices for each GP
        init_covs = self.gp_covs(self.dynamics_inputs, self.dynamics_inputs)

        # Add the noise diagonal and compute choleskies
        init_covs_chol = tf.linalg.cholesky(init_covs)

        # Tile inputs to roll out in parallel (M, S + A, N)
        inputs = tf.transpose(self.dynamics_outputs, (1, 0))
        inputs = tf.tile(self.dynamics_inputs[None, :, :], (num_rollouts, 1, 1))

        # Tile outputs to roll out in parallel (M, S, N)
        outputs = tf.transpose(self.dynamics_outputs, (1, 0))
        outputs = tf.tile(self.dynamics_outputs[None, :, :], (num_rollouts, 1, 1))

        # Sample initial state for every rollout
        state = state_dist.sample(num_rollouts)[:, :, None]

        # Do rollouts
        for h in range(horizon):

            action = self.policy(state)
            state_action = tf.concat([state, action], axis=1)

            # Get mean and covariance matrices using covariance cholesky
            K_data_data = [gp_cov(inputs, inputs) for gp_cov in self.gp_covs]
            K_star_data = [gp_cov(state_action, inputs) for gp_cov in self.gp_covs]
            K_star_star = [gp_cov(state_action, state_action) for gp_cov in self.gp_covs]

            # Sample next state
            state_delta = tfd.MultivariateNormalTriL(loc=None,
                                                     scale_tril=None)

            # Update covariance cholesky

        # Compute KL divergence


    def gp_covs(self, x1, x2, noise=True):

        covs = tf.stack([gp_cov(x1, x2) for gp_cov in self._gp_covs])

        if noise:
            covs = covs + tf.eye(covs.shape[-1])


    @abstractmethod
    def train_dynamics_model(self, **kwargs):
        pass


    def match_delta_moments(self, mean_full, cov_full):
        raise NotImplementedError

