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


def show_camera_feed(model, last_frames=INPUT_SIZE):

    df_deque = deque(maxlen=last_frames)
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

        df = convert_mp_to_df(results)
        df_deque.append(df)
        arr_inp = np.reshape(pd.concat(df_deque, axis=0, ignore_index=True).values, (len(df_deque), ROWS_PER_FRAME, 3))
        arr_prep = preprocess_data_to_same_size(arr_inp)
        perc_missing = np.sum(arr_prep[0][:, HAND_INDICES, 0:2] == 0) / ((len(df_deque) + 1) * (N_LANDMARKS * 2))

        if perc_missing < 0.3:
            X_in = torch.from_numpy(arr_prep[0][None, :, :, 0:2].astype(np.float32)).to(DEVICE)
            pred = model(X_in)
            top_values, top_indices = torch.topk(pred, k=5)
            top_labels = [label_dict_inference[x] for x in top_indices.cpu().numpy()[0]]

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


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

if __name__ == '__main__':
    DL_FRAMEWORK='pytorch'
    ckpt_name = r"TransformerPredictor/2023-06-16 00_18/TransformerPredictor_best_model"
    ckpt_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, ckpt_name + '.ckpt')
    yaml_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, ckpt_name + '_params.yaml')

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    model = TransformerPredictor(**config)
    model.load_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()

    show_camera_feed(model)
