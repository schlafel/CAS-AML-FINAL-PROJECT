import matplotlib.pyplot as plt
import numpy as np
from src.augmentations import   *


def compare_augmentation(frame,frame_aug):
    fig,ax1,ax2 = plt.subplots(1,2,sharey=True,sharex=True)
    ax1.scatter(frame_array[0,:,0],
               frame_array[0,:,1],)
    ax2.scatter(shift_lm2[0,:,0],
               shift_lm2[0,:,1],)

    plt.show()

if __name__ == '__main__':


    #Load a sample....
    frame_array = np.load(r"../../data/processed/train_landmark_files/2044-14994870.npy")
    frame_array[:,:,1] =  frame_array[:,:,1]*-1


    #frame_array = np.random.random((1,20,2))-.5




    shift_lm2 = shift_landmarks2(frame_array)
    shift_lm_old = shift_landmarks(frame_array)


    fig,(ax,ax1,ax2) = plt.subplots(1,3,sharey=True,sharex=True)
    ax.scatter(frame_array[0,:,0],
               frame_array[0,:,1],)
    ax1.scatter(shift_lm2[0,:,0],
               shift_lm2[0,:,1],)
    ax2.scatter(shift_lm_old[0,:,0],
               shift_lm_old[0,:,1],)
    fig.suptitle("Mirror")
    #plt.show()



    ## Mirror Landmarks

    shift_lm2 = mirror_landmarks2(frame_array)
    shift_lm_old = mirror_landmarks(frame_array)

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, sharey=True, sharex=True)
    ax.scatter(frame_array[0, :, 0],
               frame_array[0, :, 1], )
    ax1.scatter(shift_lm2[0, :, 0],
                shift_lm2[0, :, 1], )
    ax2.scatter(shift_lm_old[0, :, 0],
                shift_lm_old[0, :, 1], )
    fig.suptitle("Mirror")



    ## Random Rotation

    shift_lm2 = random_rotation(frame_array)
    shift_lm_old = mirror_landmarks(frame_array)

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, sharey=True, sharex=True)
    ax.scatter(frame_array[0, :, 0],
               frame_array[0, :, 1], )
    ax1.scatter(shift_lm2[0, :, 0],
                shift_lm2[0, :, 1], )

    fig.suptitle("Random Rotation")














    plt.show()

