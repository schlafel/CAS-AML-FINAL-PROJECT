"""
This module contains functions for data augmentation and normalization of hand landmarks extracted from sign language videos.

.. list-table:: Summary of Augmentations
   :widths: 30 70
   :header-rows: 1

   * - Functions
     - Description
   * - shift_landmarks(frames, max_shift=0.01)
     - Shifts the landmark coordinates randomly by a small amount.
   * - mirror_landmarks(frames)
     - Inverts (mirrors) the landmark coordinates along the x-axis.
   * - frame_dropout(frames, dropout_rate=0.05)
     - Randomly drops frames from the input landmark data based on a specified dropout rate.
   * - random_scaling(frames, scale_range=(0.9, 1.1))
     - Applies random scaling to the landmark coordinates within a specified scale range.
   * - random_rotation(frames, max_angle=10)
     - Applies a random rotation to the landmark coordinates, with the rotation angle within a specified range.
   * - normalize(frames, mn, std)
     - Normalizes the frames using a given mean and standard deviation.
   * - standardize(frames)
     - Standardizes the frames so that they have a mean of 0 and a standard deviation of 1.

"""

import sys

sys.path.insert(0, '../src')
import numpy as np


def shift_landmarks(frames, max_shift=0.01):
    """
    Shift landmark coordinates randomly by a small amount.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        max_shift (float): Maximum shift for the random shift (default: 0.01).

    Returns:
        numpy.ndarray: An array of augmented landmarks.
    """
    # Compute random shifts for all dimensions
    shifts = np.random.uniform(-max_shift, max_shift, size=frames.shape)

    # Apply shifts to frames
    augmented_landmarks = frames + shifts
    return augmented_landmarks


def mirror_landmarks(frames):
    """
    Invert/mirror landmark coordinates along the x-axis.

    Args:
        frames (numpy.ndarray): An array of landmarks data.

    Returns:
        numpy.ndarray: An array of inverted landmarks.
    """
    if not isinstance(frames, np.ndarray):
        frames = frames.numpy()
    inverted_frames = np.copy(frames)
    inverted_frames[:, :, 0] = -inverted_frames[:, :, 0] + 1
    return inverted_frames


def frame_dropout(frames, dropout_rate=0.05):
    """
    Randomly drop frames from the input landmark data.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        dropout_rate (float): The proportion of frames to drop (default: 0.05).

    Returns:
        numpy.ndarray: An array of landmarks with dropped frames.
    """
    if not isinstance(frames, np.ndarray):
        frames = frames.numpy()
    keep_rate = 1 - dropout_rate
    keep_indices = np.random.choice(len(frames), int(len(frames) * keep_rate), replace=False)
    keep_indices.sort()
    dropped_landmarks = np.array([frames[i] for i in keep_indices])
    return dropped_landmarks


def random_scaling(frames, scale_range=(0.9, 1.1)):
    """
    Apply random scaling to landmark coordinates.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        scale_range (tuple): A tuple containing the minimum and maximum scaling factors (default: (0.9, 1.1)).

    Returns:
        numpy.ndarray: An array of landmarks with randomly scaled coordinates.
    """
    if not isinstance(frames, np.ndarray):
        frames = frames.numpy()
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return frames * scale_factor


def random_rotation(frames, max_angle=10):
    """
    Apply random rotation to landmark coordinates. (on X and Y only)

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        max_angle (int): The maximum rotation angle in degrees (default: 10).

    Returns:
        numpy.ndarray: An array of landmarks with randomly rotated coordinates.
    """
    if not isinstance(frames, np.ndarray):
        frames = frames.numpy()

    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    frames[:, :, :2] = np.einsum('ijk,kl->ijl', frames[:, :, :2], rotation_matrix)

    return frames


def normalize(frames, mn, std):
    """
    Normalize the frames with a given mean and standard deviation.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        mn (float): The mean value for normalization.
        std (float): The standard deviation for normalization.

    Returns:
        numpy.ndarray: An array of normalized landmarks.
    """
    return (frames - mn) / std


def standardize(frames):
    """
    Standardize the frames so that they have mean 0 and standard deviation 1.

    Args:
        frames (numpy.ndarray): An array of landmarks data.

    Returns:
        numpy.ndarray: An array of standardized landmarks.
    """
    frames_mean = frames.mean(axis=1, keepdims=True)
    frames_std = frames.std(axis=1, keepdims=True)
    return (frames - frames_mean) / (frames_std + np.finfo(float).eps)
