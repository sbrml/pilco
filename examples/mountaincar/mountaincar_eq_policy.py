import numpy as np
import tensorflow as tf

from pilco.policies import EQPolicy


class MountaincarEQPolicy(EQPolicy):

    def __init__(self,
                 num_eq_features,
                 dtype,
                 name='mountaincar_eq_policy',
                 **kwargs):

        super().__init__(state_dim=2,
                         action_dim=1,
                         num_eq_features=num_eq_features,
                         dtype=dtype,
                         name=name,
                         **kwargs)


    def reset(self):

        """
        Resets policy parameters: location, scale and weights of EQ features.
        :return:
        """

        # Locations of EQ scales along the angle dimension
        eq_pos_loc = tf.random.normal(shape=(self.num_eq_features,),
                                      mean=0.5,
                                      stddev=0.5,
                                      dtype=self.dtype)

        # Locations of EQ scales along the angular velocity dimension
        eq_vel_locs = tf.random.normal(shape=(self.num_eq_features,),
                                       mean=0.,
                                       stddev=0.035,
                                       dtype=self.dtype)

        # Concatenate components of EQ locations and assign
        eq_locs = tf.stack([eq_pos_loc, eq_vel_locs], axis=-1)

        self.eq_locs.assign(eq_locs)

        # Log-scales of EQ components (shared across all EQ components)
        eq_log_scales = tf.constant([[-1., -3.]], dtype=self.dtype)
        self.eq_log_scales.assign(eq_log_scales)

        # Weight coefficients for each EQ component
        eq_weights_init = tf.random.normal(shape=(self.num_eq_features,),
                                           mean=0.,
                                           stddev=0.3,
                                           dtype=self.dtype)
        self.eq_weights.assign(eq_weights_init)