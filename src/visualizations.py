import sys

sys.path.insert(0, '../src')
from config import *

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import mediapipe as mp
import random

random.seed(SEED)

from data.dataset import label_dict_inference

def visualize_target_samples(landmarks_list, target_sign=None, invert=True):
    
    size = INPUT_SIZE
    offset = 20
    
    fig, ax = plt.subplots(1, figsize=(8 * len(landmarks_list) / 2, 10))    
    
    def update(frame):
        
        ax.cla()
        print('.', end='')

        for sample_idx, frames in enumerate(landmarks_list):

            landmark_lists = frames[:size]

            landmark_offset = 192 * sample_idx

            # `landmark_lists` is a list containing sequence of mediapipe landmarks for face, left_hand, pose, and right_hand
            face_landmarks = landmark_lists[:, FACE_INDICES, :]
            left_hand_landmarks = landmark_lists[:, LEFT_HAND_INDICES, :]
            pose_landmarks = landmark_lists[:, POSE_INDICES, :]
            right_hand_landmarks = landmark_lists[:, RIGHT_HAND_INDICES, :]

            face_connections = mp.solutions.face_mesh_connections.FACEMESH_CONTOURS
            pose_connections = mp.solutions.pose.POSE_CONNECTIONS
            hand_connections = mp.solutions.hands.HAND_CONNECTIONS

            new_face_landmark_map = {x: i for i, x in enumerate(USEFUL_FACE_LANDMARKS)}
            face_connections = frozenset((new_face_landmark_map[x], new_face_landmark_map[y]) for (x, y) in face_connections if x in USEFUL_FACE_LANDMARKS and y in USEFUL_FACE_LANDMARKS)

            pose_first_idx=USEFUL_POSE_LANDMARKS[0]-POSE_FEATURE_START
            new_pose_landmark_map = {i+pose_first_idx: x for i, x in enumerate(USEFUL_POSE_LANDMARKS-USEFUL_POSE_LANDMARKS[0])}
            pose_connections = frozenset((new_pose_landmark_map[x], new_pose_landmark_map[y]) for (x, y) in pose_connections if x in new_pose_landmark_map.keys() and y in new_pose_landmark_map.keys())

            multiplier = -1 if invert else 1
            face_x = [multiplier * float(x) + sample_idx * 2 for x in face_landmarks[frame][:, 0]]
            face_y = [multiplier * float(y) for y in face_landmarks[frame][:, 1]]
            pose_x = [multiplier * float(x) + sample_idx*2 for x in pose_landmarks[frame][:, 0]]
            pose_y = [multiplier * float(y)  for y in pose_landmarks[frame][:, 1]]
            lh_x = [multiplier * float(x) + sample_idx*2 for x in left_hand_landmarks[frame][:, 0]]
            lh_y = [multiplier * float(y)  for y in left_hand_landmarks[frame][:, 1]]
            rh_x = [multiplier * float(x) + sample_idx*2 for x in right_hand_landmarks[frame][:, 0]]
            rh_y = [multiplier * float(y)  for y in right_hand_landmarks[frame][:, 1]]

            ax.scatter(pose_x, pose_y)
            ax.scatter(face_x, face_y,s=5)
       
            for i in pose_connections:
                if pose_x[i[0]] !=0. and pose_x[i[1]] != 0. and pose_y[i[0]] != 0. and pose_y[i[1]] != 0.:
                    ax.plot([pose_x[i[0]], pose_x[i[1]]],[pose_y[i[0]], pose_y[i[1]]],color='k', lw=0.8)

            for i in face_connections:
                if face_x[i[0]] != 0. and face_x[i[1]] != 0. and face_y[i[0]] != 0. and face_y[i[1]] != 0.:
                    ax.plot([face_x[i[0]], face_x[i[1]]],[face_y[i[0]], face_y[i[1]]],color='k', lw=0.5)

            if round(float(left_hand_landmarks[frame][0, 0]),2)!=0.00 and round(float(left_hand_landmarks[frame][0, 1]),2)!=0.00:
                plt.scatter(lh_x, lh_y,s=10)
                for i in hand_connections:
                    ax.plot([lh_x[i[0]], lh_x[i[1]]],[lh_y[i[0]], lh_y[i[1]]],color='k', lw=0.5)

            if round(float(right_hand_landmarks[frame][0, 0]),2)!=0.00 and round(float(right_hand_landmarks[frame][0, 1]),2)!=0.00:
                plt.scatter(rh_x, rh_y,s=10)
                for i in hand_connections:
                    ax.plot([rh_x[i[0]], rh_x[i[1]]],[rh_y[i[0]], rh_y[i[1]]],color='k', lw=0.5)
                        
        ax.set_ylim(multiplier * 1.5,0.0)
        if invert:
            ax.set_xlim(-1.5, (len(landmarks_list)-1)*2 + 0.5)
        else:
            ax.set_xlim(0, (len(landmarks_list) - 1) * 1.5 + 1.5)
        
        if target_sign:
            ax.set_title(label_dict_inference[target_sign])

    animation = FuncAnimation(fig, update, frames=size, interval=50)
    print(f'\n Frame : {size}: ', end='')

    return animation    

def visualize_target_sign(dataset, target_sign, n_samples=6):
    """
    Visualize `n_samples` instances of a given target sign from the dataset.

    This function generates a visual representation of the landmarks for each sample
    belonging to the specified `target_sign`.

    Args:
        dataset (ASL_Dataset): The ASL dataset to load data from.
        target_sign (int): The target sign to visualize.
        n_samples (int, optional): The number of samples to visualize. Defaults to 6.

    Returns:
        matplotlib.animation.FuncAnimation: A matplotlib animation object displaying the landmarks for each frame.
        
    :param dataset: The ASL dataset to load data from.
    :type dataset: ASL_Dataset
    :param target_sign: The target sign to visualize.
    :type target_sign: int
    :param n_samples: The number of samples to visualize, defaults to 6.
    :type n_samples: int, optional
    
    :return: A matplotlib animation object displaying the landmarks for each frame.
    :rtype: matplotlib.animation.FuncAnimation
    """

    print('Generating ', end='')
    target_indices = []
    for i, ( _ , target) in enumerate(dataset):
        if target == target_sign:
            target_indices.append(i)
            print('.', end='')
            if len(target_indices) >= n_samples:
                break

    # Randomly choose n_samples samples from the target_indices
    selected_indices = random.sample(target_indices, min(n_samples, len(target_indices)))

    # Retrieve the samples from the dataset
    samples = [dataset[i] for i in selected_indices]

    #size = 0
    _, target = samples[0]

    #for i, sample in enumerate(samples):
    #    if sample['size'] > size:
    #        size = int(sample['size'])

    landmarks_list = [sample[0] for sample in samples]

    return visualize_target_samples(landmarks_list, target_sign=target)
        
def visualize_data_distribution(dataset):
    """
    Visualize the distribution of data in terms of the number of samples and average sequence length per class.

    This function generates two bar charts: one showing the number of samples per class, and the other showing the
    average sequence length per class.

    :param dataset: The ASL dataset to load data from.
    :type dataset: ASL_Dataset
    """
    
    class_counts = dataset.df_train['target'].value_counts()
    class_lengths = dataset.df_train.groupby('target')['size'].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    ax1.bar(class_counts.index, class_counts.values)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Number of Samples per Class')

    ax2.bar(class_lengths.index, class_lengths.values)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Average Sequence Length')
    ax2.set_title('Average Sequence Length per Class')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass
    
    