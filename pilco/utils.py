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
                           plot_prefix):

    # Get true trajectories
    true_thetas = true_traj[:, 0]
    true_thetadots = true_traj[:, 1]

    # Slice out means and standard deviations
    theta_means = means[:, 0]
    thetadot_means = means[:, 1]
    theta_stds = vars_all[:, 0] ** 0.5
    thetadot_stds = vars_all[:, 1] ** 0.5

    # Plot figures
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
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

    plt.subplot(122)
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
    plt.tight_layout()
    plt.savefig(f'{plot_path}/{plot_prefix}.png')
    plt.close()
