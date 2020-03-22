import tensorflow as tf


def get_complementary_indices(indices, size):

    indices = tf.convert_to_tensor(indices)

    return tf.sparse.to_dense(tf.sets.difference(tf.range(size)[None, :], indices[None, :]))[0]


def quadratic_form(x, loc, cov):

    # Uprank tensors if needed
    if tf.rank(x) < 2:
        x = tf.reshape(x, (1,) + x.shape)
    
    if tf.rank(loc) < 2:
        loc = tf.reshape(loc, (1,) + loc.shape)
    
    if tf.rank(cov) < 3:
        cov = tf.reshape(cov, (1,) + cov.shape)
    
    # Compute quadratic form and exponentiate for each component
    diffs = x - loc
    quad = tf.matmul(diffs, tf.matmul(cov_inv, diffs))
    exp_quads = tf.math.exp(-0.5 * tf.reduce_sum(quad, axis=-1))

    # RBF output is the weighted sum of rbf components
    rbf = tf.matmul(self.rbf_weights, exp_quads)

    return rbf

