import abc
import tensorflow as tf
import numpy as np

from pilco.utils import get_complementary_indices


class TransformError(Exception):
    pass


class Transfrom(tf.Module, abc.ABC):

    def __init__(self,
                 dtype=tf.float64,
                 name="transform",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.dtype = dtype


class MomentMatchingTransform(Transfrom, abc.ABC):

    def __init__(self,
                 dtype=tf.float64,
                 name="moment_matching_transform",
                 **kwargs):
        super().__init__(name=name,
                         dtype=dtype,
                         **kwargs)

    @abc.abstractmethod
    def match_moments(self, loc, cov, indices):
        """
        Performs moment matching on the slice of the loc and the covariance specified by indices.
        :param loc:
        :param cov:
        :param indices:
        :return:
        """
        pass


class IdentityTransform(MomentMatchingTransform):

    def __init__(self,
                 dtype=tf.float64,
                 name="identity_mm_transform",
                 **kwargs):
        super().__init__(dtype=dtype,
                         name=name,
                         **kwargs)

    def match_moments(self, loc, cov, indices):
        return loc, cov


class SineTransformWithPhase(MomentMatchingTransform):

    def __init__(self,
                 lower,
                 upper,
                 phase,
                 dtype=tf.float64,
                 name="sine_transform_with_phase",
                 **kwargs):

        super().__init__(name=name,
                         dtype=dtype,
                         **kwargs)

        self.lower = tf.cast(lower, dtype=self.dtype)
        self.upper = tf.cast(upper, dtype=self.dtype)

        self.phase = tf.cast(phase, dtype=self.dtype)

    @property
    def shift(self):
        return (self.upper + self.lower) / 2.

    @property
    def scale(self):
        return (self.upper - self.lower) / 2.

    def match_moments(self, loc, cov, indices):
        """

        :param loc: K x 1 vector
        :param cov: K x K vector
        :param indices: integer list specifying the slices
        :return:
        """

        # Convert inputs
        loc = tf.convert_to_tensor(loc)
        loc = tf.cast(loc, self.dtype)

        cov = tf.convert_to_tensor(cov)
        cov = tf.cast(cov, self.dtype)

        indices = tf.convert_to_tensor(indices)
        indices = tf.cast(indices, tf.int32)

        if tf.rank(loc) == 2 and loc.shape[0] == 1:
            loc = loc[0]

        if tf.rank(loc) != 1:
            raise TransformError(f"Loc must be rank 1, but had shape {loc.shape}!")

        # Store the dimensionality of the whole vector for later
        full_dim = loc.shape[0]

        if tf.rank(cov) != 2 or cov.shape[0] != cov.shape[1]:
            raise TransformError(f"Cov must be a square matrix, but had shape {cov.shape}!")

        # Check that loc and cov are conformal
        if cov.shape[0] != full_dim:
            raise TransformError(f"Loc and Cov must have matching dimensions, "
                                 f"but had shapes {loc.shape}, {cov.shape}!")

        if not tf.reduce_all(tf.logical_and(indices >= 0,
                                            indices < full_dim)):
            raise TransformError(f"Indices must lie in the range [0, {full_dim}), "
                                 f"but were {indices.numpy()}!")

        # TODO: Check there are no duplicates in the index list

        # Create index sets for slicing.
        # Index set A: slice that we will pass through the trig function
        # Index set B: slice that will NOT be transformed
        # Note, that the indices need to be rank-2 in order to select a vector, not a scalar
        indices_a = indices[:, None]
        indices_b = get_complementary_indices(indices, full_dim)[:, None]

        # Get slices into the covariance matrix
        indices_aa = tf.stack(tf.meshgrid(indices_a, indices_a), axis=2)
        indices_bb = tf.stack(tf.meshgrid(indices_b, indices_b), axis=2)

        indices_ab = tf.stack(tf.meshgrid(indices_b, indices_a), axis=2)
        indices_ba = tf.stack(tf.meshgrid(indices_a, indices_b), axis=2)

        a_dim = indices_a.shape[0]

        # Perform slicing for the mean
        # The rest of the code is expecting the means as row vectors
        mean_a = tf.gather_nd(loc, indices_a)[None, :]
        #print("mean a ", mean_a)

        # Shift the selected mean by the phase
        mean_a = mean_a + self.phase
        #print("mean a plus pi/2", mean_a)

        # The rest of the code is expecting the means as row vectors
        mean_b = tf.gather_nd(loc, indices_b)[None, :]

        # Perform slicing for the covariance. Note that the selected covariances need no shift
        cov_aa = tf.gather_nd(cov, indices_aa)
        cov_bb = tf.gather_nd(cov, indices_bb)
        cov_ba = tf.gather_nd(cov, indices_ba)

        # Moment match the mean through the sine
        mean_coeff = tf.exp(-0.5 * tf.linalg.diag_part(cov_aa))[None, :]
        mean_a_bounded = tf.sin(mean_a) * mean_coeff

        mean_a_bounded_rescaled = self.shift + self.scale * mean_a_bounded

        # 1 x D
        mean_full_bounded = tf.scatter_nd(indices_a, mean_a_bounded_rescaled[0], shape=(full_dim,))
        mean_full_bounded = tf.tensor_scatter_nd_update(mean_full_bounded, indices_b, mean_b[0])[None, :]

        # Calculate the cross-covariance term
        cov_ba_bounded = tf.transpose(mean_b) * tf.sin(mean_a) + cov_ba * tf.cos(mean_a)
        cov_ba_bounded = mean_coeff * cov_ba_bounded
        cov_ba_bounded = cov_ba_bounded - tf.transpose(mean_b) * mean_a_bounded
        cov_ba_bounded_rescaled = self.scale * cov_ba_bounded

        # Calculate covariance term

        # A x A x A
        c = tf.eye(a_dim, dtype=self.dtype)[:, None, :] + tf.eye(a_dim, dtype=self.dtype)[None, :, :]
        d = tf.eye(a_dim, dtype=self.dtype)[:, None, :] - tf.eye(a_dim, dtype=self.dtype)[None, :, :]

        def calculate_cov_terms(v):
            mean_cross = tf.einsum('iu, abu -> ab', mean_a, v)
            cov_cross = tf.einsum('ij, abi, abj -> ab', cov_aa, v, v)

            return tf.cos(mean_cross) * tf.exp(-0.5 * cov_cross)

        cov_aa_bounded = calculate_cov_terms(d) - calculate_cov_terms(c)
        cov_aa_bounded = 0.5 * cov_aa_bounded - tf.transpose(mean_a_bounded) * mean_a_bounded
        cov_aa_bounded_rescaled = cov_aa_bounded * self.scale ** 2

        # Join matrices back up
        cov_full_bounded = tf.scatter_nd(indices_aa,
                                         cov_aa_bounded_rescaled,
                                         shape=(full_dim, full_dim))

        # Cross terms
        cov_full_bounded = tf.tensor_scatter_nd_update(cov_full_bounded,
                                                       indices_ab,
                                                       tf.transpose(cov_ba_bounded_rescaled))
        cov_full_bounded = tf.tensor_scatter_nd_update(cov_full_bounded,
                                                       indices_ba,
                                                       cov_ba_bounded_rescaled)

        cov_full_bounded = tf.tensor_scatter_nd_update(cov_full_bounded,
                                                       indices_bb,
                                                       cov_bb)

        return mean_full_bounded, cov_full_bounded

    def __call__(self, tensor, indices):

        # Convert inputs
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.cast(tensor, self.dtype)

        indices = tf.convert_to_tensor(indices)
        indices = tf.cast(indices, tf.int32)[:, None]

        tensor_a = tf.gather_nd(tensor, indices)

        tensor_a = self.shift + self.scale * tf.sin(tensor_a + self.phase)
        tensor = tf.tensor_scatter_nd_update(tensor, indices, tensor_a)

        return tensor


class SineTransform(SineTransformWithPhase):

    def __init__(self,
                 lower,
                 upper,
                 dtype=tf.float64,
                 name="sine_transform",
                 **kwargs):
        super().__init__(lower=lower,
                         upper=upper,
                         phase=0.,
                         dtype=dtype,
                         name=name,
                         **kwargs)


class CosineTransform(SineTransformWithPhase):

    def __init__(self,
                 lower,
                 upper,
                 dtype=tf.float64,
                 name="cosine_transform",
                 **kwargs):
        super().__init__(lower=lower,
                         upper=upper,
                         phase=np.pi / 2.,
                         dtype=dtype,
                         name=name,
                         **kwargs)
