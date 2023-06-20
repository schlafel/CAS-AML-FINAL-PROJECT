"""
=======================
Live Camera Predictions
=======================
This script is used to make live sign predictions from a webcam feed or a video file.

Imports:
- Required libraries and modules.
"""
import os
import sys

sys.path.insert(0, '../src')

import cv2
import mediapipe as mp
from data.data_utils import preprocess_data_to_same_size
from models.pytorch.models import *
import pandas as pd
import yaml
from collections import deque
from data.dataset import label_dict_inference
from config import *
from video_utils import convert_mp_to_df, draw_landmarks_on_frame
from augmentations import standardize

from scipy import stats


def show_camera_feed(model, last_frames=INPUT_SIZE, capture=0):
    """
    Function to show live feed from camera or predict a video.

    :param model:
        Pytorch/Tensorflow model for prediction.

    :param last_frames: int, optional
        The number of frames to use for prediction. Default is INPUT_SIZE.

    :param capture: int or str, optional
        Choose your webcam (0) or a video file (by entering a path). Default is 0.

    :return: None
        Displays live feed with prediction results.
    """

    df_deque = deque(maxlen=last_frames)
    idx_deque = deque(maxlen=last_frames)
    # Create a VideoCapture object to access the camera
    cap = cv2.VideoCapture(capture)  # 0 represents the default camera index, change it if you have multiple cameras
    _i = 0
    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        # convert to rgb.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pass frame to mediapipe
        results = holistic.process(frame_rgb)

        df = convert_mp_to_df(results)
        df_deque.append(df)
        arr_inp = np.reshape(pd.concat(df_deque, axis=0, ignore_index=True).values, (len(df_deque), ROWS_PER_FRAME, 3))
        arr_prep = preprocess_data_to_same_size(arr_inp)

        # standardize data
        landmarks = standardize(arr_prep[0])

        perc_missing = np.sum(landmarks[:, HAND_INDICES, 0:2] == 0) / ((len(df_deque) + 1) * (N_LANDMARKS * 2))

        if perc_missing < 0.3:
            X_in = torch.from_numpy(landmarks[None, :, :, 0:2].astype(np.float32)).to(DEVICE)
            pred = model(X_in)
            top_values, top_indices = torch.topk(pred, k=5)
            top_labels = [label_dict_inference[x] for x in top_indices.cpu().numpy()[0]]
            idx_deque.append(top_indices[0].cpu().numpy()[0])
            cv2.putText(frame, ', '.join(top_labels), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # update deque

        # Render the landmarks on the frame
        frame = draw_landmarks_on_frame(frame, results)

        # Display the frame
        cv2.imshow('ASL Parser', frame)

        _i += 1
        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the OpenCV windows
    cap.release()

    if isinstance(capture, str):
        pred = stats.mode(np.array(idx_deque))[0][0]
        print(f'Captur: {capture}, \n Prediction: {pred} ({label_dict_inference[pred]})')


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

if __name__ == '__main__':
    DL_FRAMEWORK = 'pytorch'
    ckpt_name = r"TransformerPredictor/2023-06-20 07_24/TransformerPredictor_best_model"
    ckpt_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, ckpt_name + '.ckpt')
    yaml_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, ckpt_name + '_params.yaml')

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    model = TransformerPredictor(**config)
    model.load_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()

    # base_path = r"C:\Users\fs.GUNDP\Python\CAS-AML-FINAL-PROJECT\data\raw\MSASL\Videos\after"
    # for file in os.listdir(base_path):
    show_camera_feed(model, capture=0)
