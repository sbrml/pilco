import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np


def assert_near(x, y, atol=None, rtol=None):
    assert tf.debugging.assert_near(x, y, rtol=rtol, atol=atol) is None


def get_complementary_indices(indices, size):

    indices = tf.convert_to_tensor(indices)

    return tf.sparse.to_dense(tf.sets.difference(tf.range(size)[None, :], indices[None, :]))[0]


def plot_pendulum_rollouts(steps,
                           true_traj,
                           means,
                           vars_all,
                           plot_path,
                           plot_prefix,
                           policy,
                           true_actions,
                           s_points=60,
                           s_dot_points=30):

    # Get true trajectories
    true_thetas = true_traj[:, 0]
    true_thetadots = true_traj[:, 1]

    # Slice out means and standard deviations
    theta_means = means[:, 0]
    thetadot_means = means[:, 1]
    theta_stds = vars_all[:, 0] ** 0.5
    thetadot_stds = vars_all[:, 1] ** 0.5

    # Plot figures
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.plot(steps, true_thetas, '--', color='black')
    plt.plot(steps, theta_means, color='blue')
    plt.fill_between(steps,
                     theta_means - theta_stds,
                     theta_means + theta_stds,
                     alpha=0.5,
                     color='blue')
    plt.xlabel('step')
    plt.ylabel('theta')
    plt.grid(True)
    plt.ylim([-5 * np.pi, 5 * np.pi])

    plt.subplot(222)
    plt.plot(steps, true_thetadots, '--', color='black')
    plt.plot(steps, thetadot_means, color='red')
    plt.fill_between(steps,
                     thetadot_means - thetadot_stds,
                     thetadot_means + thetadot_stds,
                     alpha=0.5,
                     color='red')
    plt.xlabel('step')
    plt.ylabel('theta dot')
    plt.grid(True)
    plt.ylim([-10., 10])

    s_linspace = tf.cast(tf.linspace(-4 * np.pi, 3 * np.pi, s_points), dtype=tf.float64)

    s_dot_linspace = tf.cast(tf.linspace(-8., 8., s_dot_points), dtype=tf.float64)

    s_grid, s_dot_grid = tf.meshgrid(s_linspace, s_dot_linspace)

    grid = tf.stack([s_grid, s_dot_grid], axis=-1)

    grid = tf.reshape(grid, (-1, 2))

    actions = tf.stack([policy(point) for point in grid], axis=0)
    actions = tf.reshape(actions, (s_dot_points, s_points))

    centroids = policy.policy.eq_locs().numpy()

    centroid_thetas = centroids[:, 0] #np.arctan2(centroids[:, 0], centroids[:, 1])
    centroid_theta_dots = centroids[:, 1]

    plt.subplot(223)
    contour = plt.contourf(s_grid, s_dot_grid, actions, cmap='coolwarm', alpha=0.5)
    plt.scatter(centroid_thetas, centroid_theta_dots, marker='x', c='k')
    plt.plot(true_thetas, true_thetadots, c='k', linestyle='--')
    plt.plot(theta_means, thetadot_means, c='k')
    plt.clim(-2, 2)
    plt.colorbar(contour)
    plt.xlabel("Theta")
    plt.ylabel("Theta dot")
    plt.xlim([-4. * np.pi, 3. * np.pi])
    plt.ylim([-8., 8.])

    plt.subplot(224)
    plt.plot(steps[:-1], true_actions, c='k')
    plt.grid(True)
    plt.xlabel("step")
    plt.ylabel("action")
    plt.ylim([-2, 2])

    plt.tight_layout()
    plt.savefig(f'{plot_path}/{plot_prefix}.png')
    plt.close()


def chol_update_by_block_lu(L, a, b, tol=1e-12):

    """
    Computes the updated cholesky factorisation of a matrix F, where:

        F = M MT = [[L LT, a],
                    [aT,   b]]

    F is assumed to be positive-definite. This uses the identity:

        M = [[L,         0],
             [(L^-1 a)T, c]]

    where c^2 = b - (L^-1 a)T (L^-1 a).

    :param L: tf.tensor, [..., N, N] lower triangular cholesky factor L
    :param a: tf.tensor, [..., N, 1] column-vector-like
    :param b: tf.tensor, [..., 1, 1] scalar-like
    :return:
    """

    # Compute (L^-1 a) and its transpose
    L_inv_a = tf.linalg.triangular_solve(L, a, lower=True)
    L_inv_aT = tf.linalg.matrix_transpose(L_inv_a)

    # Compute c - add tol because sqrt is not differentiable at origin
    c = (b - tf.einsum('...ij, ...ij -> ...', L_inv_a, L_inv_a)) ** 0.5

    # Join matrices up to original matrix
    print(L.shape, a.shape)
    print(L_inv_aT.shape, c.shape)
    M_top = tf.concat([L, tf.zeros_like(a)], axis=-1)
    M_bottom = tf.concat([L_inv_aT, c], axis=-1)

    print(M_top.shape, M_bottom.shape)

    M = tf.concat([M_top, M_bottom], axis=-2)

    return M
