from abc import abstractmethod
import tensorflow as tf

from pilco.transforms import MomentMatchingTransform, IdentityTransform


class CostError(Exception):
    pass


class Cost(tf.Module):

    def __init__(self, dtype, name="cost", **kwargs):

        super().__init__(name=name,
                         **kwargs)

        self.dtype = dtype

    @abstractmethod
    def call(self, x):
        pass

    def __call__(self, x):
        return self.call(x)


class EQCost(Cost):

    def __init__(self,
                 target_loc,
                 target_scale,
                 target_dim,
                 dtype,
                 transform = None,
                 name="eq_cost",
                 **kwargs):

        super().__init__(name=name,
                         dtype=dtype,
                         **kwargs)

        self.target_dim = target_dim

        self.target_loc = tf.convert_to_tensor(target_loc)
        self.target_loc = tf.cast(self.target_loc, dtype)

        if self.target_loc.shape != (1, target_dim):
            raise CostError(f"Expected target_loc.shape == (1, target_dim) "
                            f"found {self.target_loc.shape}, {(1, target_dim)}.")

        self.target_scale = tf.convert_to_tensor(target_scale)
        self.target_scale = tf.cast(self.target_scale, dtype)

        if self.target_loc.shape != self.target_scale.shape:
            raise CostError(f"Expected target_loc.shape == target_scale.shape "
                            f"found {self.target_loc.shape}, {self.target_scale.shape}.")

        if transform is None:
            transform = IdentityTransform()

        self.transform: MomentMatchingTransform = transform

    def expected_cost(self, loc, cov):

        loc = tf.convert_to_tensor(loc)
        loc = tf.cast(loc, self.dtype)

        if loc.shape != (1, self.target_dim):
            raise CostError(f"Location shape must be equal to (1, target_dim)"
                            f" (Found shape {loc.shape}, {(1, self.target_dim)}")

        cov = tf.convert_to_tensor(cov)
        cov = tf.cast(cov, self.dtype)

        if tf.rank(cov) != 2 or cov.shape[0] != cov.shape[1] or cov.shape[1] != loc.shape[1]:
            raise CostError(f"Incorrect dimensions for covariance!"
                            f" (Expected ({loc.shape[1], loc.shape[1]}), found {cov.shape})")

        loc, cov = self.transform.match_moments(loc=loc,
                                                cov=cov,
                                                indices=tf.constant([0], dtype=tf.int32))

        I = tf.eye(cov.shape[0], dtype=self.dtype)

        cov_plus_target_scale = cov + I * tf.linalg.diag(self.target_scale[0, :] ** 2)

        diffs = loc - self.target_loc

        quad = tf.linalg.solve(cov_plus_target_scale, tf.transpose(diffs))
        quad = tf.einsum('ij, jk ->',
                         diffs,
                         quad)
        quad = -0.5 * quad

        exp_quad = tf.math.exp(quad)

        det_coeff = tf.math.sqrt(tf.linalg.det(I + cov / self.target_scale**2))

        cost = 1. - exp_quad / det_coeff

        return cost

    def call(self, loc):
        return self.expected_cost(loc, tf.zeros([loc.shape[1], loc.shape[1]]))
