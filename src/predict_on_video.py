"""
=================
Video Predictions
=================
This script defines methods to predict signs from a given video.

Imports:
- Required libraries and modules.
"""
import os, sys, cv2, random

sys.path.insert(0, '../src')
from config import *

import mediapipe as mp
from video_utils import draw_landmarks_on_frame, capture_frames, convert_frames_to_landmarks
from data.data_utils import preprocess_data_to_same_size
from dl_utils import load_model_from_checkpoint
from models.pytorch.models import *
from data.dataset import label_dict_inference
from augmentations import standardize


def get_video_landmarks(video_path):
    """
    Extracts and pre-processes landmarks from a video.

    :param video_path: str
        Path to the video file.

    :return: numpy.ndarray
        Preprocessed video landmarks.
    """
    frames = capture_frames(video_path, INPUT_SIZE)

    landmarks = convert_frames_to_landmarks(frames)
    landmarks = preprocess_data_to_same_size(landmarks)
    landmarks = landmarks[0][:, :, :2]

    return landmarks


def get_top_n_predictions(model_checkpoint, landmarks, n):
    """
    Predicts the top-n signs for given landmarks using the specified model.

    :param model_checkpoint: str
        Path to the model checkpoint.

    :param landmarks: numpy.ndarray
        Preprocessed video landmarks.

    :param n: int
        Number of top predictions to return.

    :return: list
        List of top-n predicted signs.
    """
    model = load_model_from_checkpoint(model_checkpoint)
    model.eval()

    landmarks = torch.from_numpy(landmarks.astype(np.float32)).to(DEVICE)
    landmarks = torch.unsqueeze(landmarks, 0)
    pred = model(landmarks).cpu().detach().numpy()[0]

    sorted_indices = np.argsort(pred)[::-1]
    top_n_indices = sorted_indices[:n]
    top_n_values = pred[top_n_indices]

    return [label_dict_inference[i] for i in top_n_indices]


def play_video_with_predictions(video_path, model_checkpoint, num_top_predictions, sign="",  show_mesh=True):
    """
    Plays the video with predicted signs overlay.

    :param video_path: str
        Path to the video file.

    :param model_checkpoint: str
        Path to the model checkpoint.

    :param num_top_predictions: int
        Number of top predictions to overlay on the video.

    :param sign: str, optional
        Name of the sign to display. Default is an empty string.

    :param show_mesh: bool, optional
        If True, the landmark mesh is drawn on the video. Default is True.
    """
    landmarks = get_video_landmarks(video_path)
    landmarks = standardize(landmarks)
    top_n_labels = get_top_n_predictions(model_checkpoint, landmarks, num_top_predictions)

    frames = capture_frames(video_path)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    quit_loop = False
    loop_counter = 0

    while True:
        if quit_loop or loop_counter > 10:
            break

        for i, frame in enumerate(frames):

            if show_mesh and loop_counter == 0:
                results = holistic.process(frame)

                frame = draw_landmarks_on_frame(frame, results)

            cv2.putText(frame, ', '.join(top_n_labels) + f' >> [{sign}]', (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Video Predictor', frame)

            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                quit_loop = True
                break

        if key & 0xFF in [ord('q')]:
            break

        loop_counter += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    holistic.close()


def get_random_video(root_dir='../data/raw/MSASL/Videos/'):
    # Create a generator that will iterate over all mp4 files
    all_videos = (
        os.path.join(subdir, file)
        for subdir, dirs, files in os.walk(root_dir)
        for file in files if file.lower().endswith('.mp4')
    )

    # Convert the generator to a list and get its length
    all_videos = list(all_videos)
    num_videos = len(all_videos)

    # Select a random index and return the corresponding video
    random_index = random.randint(0, num_videos - 1)
    return all_videos[random_index]


if __name__ == '__main__':

    ckpt_name = r'TransformerPredictor/2023-06-16 00_18/TransformerPredictor_best_model'
    ckpt_name = r'TransformerPredictor/2023-06-20 07_24/TransformerPredictor_best_model'

    video_path = get_random_video() #"../data/raw/MSASL/Videos/apple/apple_3wq_Jcw0_24_trimmed.mp4"
    sign = os.path.basename(os.path.dirname(video_path))

    print(video_path)
    play_video_with_predictions(video_path, ckpt_name, 5, sign, show_mesh=True)
