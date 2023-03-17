import os

BASE_PATH = r"Y:\CAS_AML\FINAL_PROJECT"

PATH_DATA = os.path.join(BASE_PATH, "DATA", "asl-signs")
PATH_TRAIN_DATA = os.path.join(PATH_DATA,"train.csv")
PATH_LABELMAP = os.path.join(PATH_DATA, "sign_to_prediction_index_map.json")