import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np


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
    plt.ylim([-5 * np.pi, 3 * np.pi])

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

    s_linspace = tf.cast(tf.linspace(-4 * np.pi, 2 * np.pi, s_points), dtype=tf.float64)

    s_dot_linspace = tf.cast(tf.linspace(-8., 8., s_dot_points), dtype=tf.float64)

    s_grid, s_dot_grid = tf.meshgrid(s_linspace, s_dot_linspace)

    grid = tf.stack([s_grid, s_dot_grid], axis=-1)

    grid = tf.reshape(grid, (-1, 2))

    actions = tf.stack([policy(point) for point in grid], axis=0)
    actions = tf.reshape(actions, (s_dot_points, s_points))

    centroids = policy.eq_policy.policy.rbf_locs().numpy()

    centroid_thetas = np.arctan2(centroids[:, 0], centroids[:, 1])
    centroid_theta_dots = centroids[:, 3]

    plt.subplot(223)
    contour = plt.contourf(s_grid, s_dot_grid, actions, cmap='coolwarm', alpha=0.5)
    plt.scatter(centroid_thetas, centroid_theta_dots, marker='x', c='k')
    plt.plot(true_thetas, true_thetadots, c='k', linestyle='--')
    plt.plot(theta_means, thetadot_means, c='k')
    plt.clim(-2, 2)
    plt.colorbar(contour)
    plt.xlabel("Theta")
    plt.ylabel("Theta dot")
    plt.xlim([-4. * np.pi, 2. * np.pi])
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

