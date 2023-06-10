import os.path
import pandas as pd

import cv2
import mediapipe as mp
from config import *
from src.data import data_utils
from src.models import models
import torch
import numpy as np
import pandas as pd
import json
from mediapipe.framework.formats import landmark_pb2
from collections import deque

label_dict = json.load(open(os.path.join(ROOT_PATH,'data/raw/sign_to_prediction_index_map.json'),'r'))
label_dict_inv = dict(zip(label_dict.values(),label_dict.keys()))

def convert_mp_to_df(results):
    face_landmarks = []
    if results.face_landmarks:
        for idx, landmark in enumerate(results.face_landmarks.landmark):
            face_landmarks.append(dict({'x': landmark.x,
                                        'y': landmark.y,
                                        'z': landmark.z}))
        df_face = pd.DataFrame(face_landmarks)
    else:
        df_face = pd.DataFrame(np.zeros((468, 3)) * np.nan, columns=['x', 'y', 'z'])

    pose_landmarks = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            pose_landmarks.append(dict({'x': landmark.x,
                                        'y': landmark.y,
                                        'z': landmark.z}))

        df_pose = pd.DataFrame(pose_landmarks)
    else:
        df_pose = pd.DataFrame(np.zeros((33, 3)) * np.nan, columns=['x', 'y', 'z'])

    rh_landmarks = []
    if results.right_hand_landmarks:
        for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
            rh_landmarks.append(dict({'x': landmark.x,
                                      'y': landmark.y,
                                      'z': landmark.z}))
        df_right_hand = pd.DataFrame(rh_landmarks)
    else:
        df_right_hand = pd.DataFrame(np.zeros((21, 3)) * np.nan, columns=['x', 'y', 'z'])

    lh_landmarks = []
    if results.left_hand_landmarks:
        for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
            lh_landmarks.append(dict({'x': landmark.x,
                                      'y': landmark.y,
                                      'z': landmark.z}))
        df_left_hand = pd.DataFrame(lh_landmarks)
    else:
        df_left_hand = pd.DataFrame(np.zeros((21, 3)) * np.nan, columns=['x', 'y', 'z'])

    df_x = pd.concat([df_face, df_left_hand, df_pose, df_right_hand], axis=0, ignore_index=True)
    assert df_x.shape[0] == 543

    # Order is Face, Left_Hand, POSE, RIGHT_HAND
    return df_x


def show_camera_feed(model, LAST_FRAMES=32):
    df_deque = deque(maxlen=LAST_FRAMES)
    # Create a VideoCapture object to access the camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera index, change it if you have multiple cameras
    _i = 0
    while True:
        # Read the frame from the camera
        ret, frame = cap.read()

        # convert to rgb.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pass frame to mediapipe
        results = holistic.process(frame_rgb)

        df = convert_mp_to_df(results=results)
        df_deque.append(df)
        arr_inp = np.reshape(pd.concat(df_deque,axis = 0,ignore_index=True).values,(len(df_deque),543,3))
        arr_prep = data_utils.preprocess_data_to_same_size(arr_inp)
        perc_missing = np.sum(arr_prep[0][:,HAND_INDICES,0:2] == 0) / ((len(df_deque)+1)*192)

        if perc_missing < 0.3:
            X_in = torch.from_numpy(arr_prep[0][None,:,:,0:2]).to(DEVICE)
            pred = model(X_in)
            top_values, top_indices = torch.topk(pred, k=5)
            top_labels = [label_dict_inv[x] for x in top_indices.cpu().numpy()[0]]


            cv2.putText(frame, ', '.join(top_labels), (3,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # update deque

        print(len(df_deque))

        # Render the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame,
                                  results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)

        # Render the face landmarks on the frame
        # Extract the nose landmarks
        # nose_landmarks = results.face_landmarks.landmark[mp_holistic.FACEMESH_CONTOURS['nose']]

        # Render the nose landmarks on the frame

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

        # Display the frame
        cv2.imshow('ASL Parser', frame)

        _i += 1
        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the OpenCV windows
    cap.release()


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

if __name__ == '__main__':
    ckpt_path = r"lightning_logs/FIRST_TRANSFORMER_MODEL_2/version_9/checkpoints/FIRST_TRANSFORMER_MODEL_2-epoch=25-val_accuracy=0.78.ckpt"
    full_ckpt_path = os.path.join(os.path.dirname(__file__), "./..", ckpt_path)
    model = models.TransformerPredictor()
    model = model.load_from_checkpoint(full_ckpt_path)
    model.to(DEVICE)
    model.eval()

    show_camera_feed(model)
