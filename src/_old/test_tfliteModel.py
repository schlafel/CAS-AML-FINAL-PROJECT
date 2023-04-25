from tensorflow.lite.python.interpreter import Interpreter
from src.data.data_utils import load_relevant_data_subset
from src._old.convert_model_tflite import load_full_train_datafile
from src.config import *
from tqdm import tqdm
def get_tfliteModel(path_model):
    if os.path.exists(path_model):
        print("Found the file")

        with open(path_model, 'rb') as fid:
            tf_mod = fid.read()

        interpreter = Interpreter(path_model)
        found_signatures = list(interpreter.get_signature_list().keys())
        prediction_fn = interpreter.get_signature_runner("serving_default")


        return prediction_fn


def main(path_model  ,  total = 1000,verbose = False):

    prediction_fn = get_tfliteModel(path_model=path_model)

    arr = np.zeros((1,543,3),dtype = np.float32)
    arr[arr == 0] = np.nan
    preds = prediction_fn(inputs=arr)["outputs"].argmax()


    for i in range(1,15):
        arr = np.zeros((i, 543, 3), dtype=np.float32)
        arr[arr == 0] = np.nan
        preds = prediction_fn(inputs=arr)["outputs"].argmax()


    df_train,label_dict,label_dict_inv = load_full_train_datafile()
    n_correct = 0

    for i in tqdm(range(total),):
        if verbose:
            print(30*"*")
        idx = np.random.randint(0, len(df_train))
        sample = df_train.iloc[idx]
        demo_raw_data = load_relevant_data_subset(sample.filepath, cols_to_use=["x", "y", "z"])

        output = prediction_fn(inputs=demo_raw_data)
        sign = output['outputs'].argmax()
        if verbose:
            print("PRED : ", label_dict_inv[sign], f'[{sign}]')
            print("TRUE : ", df_train.iloc[idx].sign, f'[{df_train.iloc[idx].target}]')

        if sign == sample.target:
            n_correct+= 1

    print(f"Total correct predictions: {n_correct} ({n_correct/total*100}%)")



if __name__ == '__main__':
    path_model =os.path.join(ROOT_PATH,"models/TF_LSTM_BASELINE/Exp_04/LSTM_MODEL.tflite")

    main(path_model=path_model,
         total = 5000,
         verbose = False)