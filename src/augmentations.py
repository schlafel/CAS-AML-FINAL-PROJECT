import sys

sys.path.insert(0, '../src')
from config import *

import numpy as np

"""
Data Augmentation Methods
"""
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
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return frames * scale_factor

def random_rotation(frames, max_angle=10):
    """
    Apply random rotation to landmark coordinates.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        max_angle (int): The maximum rotation angle in degrees (default: 10).

    Returns:
        numpy.ndarray: An array of landmarks with randomly rotated coordinates.
    """
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    frames[:, :, :2] = np.einsum('ijk,kl->ijl', frames[:, :, :2], rotation_matrix)

    return frames

#TODO 
def normalize(frames, mn,std):
    pass

#TODO
def standardize():
    pass
