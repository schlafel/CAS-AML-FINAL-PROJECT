"""
===============
Video Utilities
===============
"""
import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from config import FACE_FEATURES, POSE_FEATURES, HAND_FEATURES, ROWS_PER_FRAME
import cv2
import mediapipe as mp

def convert_mp_to_df(arr_results):

    face_landmarks = []
    if arr_results.face_landmarks:
        for idx, landmark in enumerate(arr_results.face_landmarks.landmark):
            face_landmarks.append(dict({'x': landmark.x,
                                        'y': landmark.y,
                                        'z': landmark.z}))
        df_face = pd.DataFrame(face_landmarks)
    else:
        df_face = pd.DataFrame(np.zeros((FACE_FEATURES, 3)) * np.nan, columns=['x', 'y', 'z'])

    pose_landmarks = []
    if arr_results.pose_landmarks:
        for idx, landmark in enumerate(arr_results.pose_landmarks.landmark):
            pose_landmarks.append(dict({'x': landmark.x,
                                        'y': landmark.y,
                                        'z': landmark.z}))

        df_pose = pd.DataFrame(pose_landmarks)
    else:
        df_pose = pd.DataFrame(np.zeros((POSE_FEATURES, 3)) * np.nan, columns=['x', 'y', 'z'])

    rh_landmarks = []
    if arr_results.right_hand_landmarks:
        for idx, landmark in enumerate(arr_results.right_hand_landmarks.landmark):
            rh_landmarks.append(dict({'x': landmark.x,
                                      'y': landmark.y,
                                      'z': landmark.z}))
        df_right_hand = pd.DataFrame(rh_landmarks)
    else:
        df_right_hand = pd.DataFrame(np.zeros((HAND_FEATURES, 3)) * np.nan, columns=['x', 'y', 'z'])

    lh_landmarks = []
    if arr_results.left_hand_landmarks:
        for idx, landmark in enumerate(arr_results.left_hand_landmarks.landmark):
            lh_landmarks.append(dict({'x': landmark.x,
                                      'y': landmark.y,
                                      'z': landmark.z}))
        df_left_hand = pd.DataFrame(lh_landmarks)
    else:
        df_left_hand = pd.DataFrame(np.zeros((HAND_FEATURES, 3)) * np.nan, columns=['x', 'y', 'z'])

    df_x = pd.concat([df_face, df_left_hand, df_pose, df_right_hand], axis=0, ignore_index=True)
    assert df_x.shape[0] == ROWS_PER_FRAME

    # Order is Face, Left_Hand, POSE, RIGHT_HAND
    return df_x


def capture_frames(video_path, target_sequence=None):
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_sequence is None:
        # If target_sequence is not provided, capture all frames
        target_sequence = frame_count
        step = 1
    else:
        # Calculate the step for the provided target_sequence
        step = frame_count // target_sequence

    # Placeholder for storing frames
    frames = []

    for i in range(target_sequence):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return np.array(frames)


def draw_landmarks_on_frame(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Render the pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    # Render the hand landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                           thickness=2,
                                                                           circle_radius=4),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                              )
    mp_drawing.draw_landmarks(frame,
                              results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)

    return frame


def convert_frames_to_landmarks(video_numpy_frames):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()
    landmarks = []

    for frame in video_numpy_frames:
        results = holistic.process(frame)
        landmark = convert_mp_to_df(results).values
        landmarks.append(landmark)

    return np.array(landmarks)
