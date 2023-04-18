from src.config import *
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_utils import get_stratified_TrainValFrames


def run():

    # Load your data into a pandas DataFrame
    path_in = os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,TRAIN_CSV_FILE)

    X_train, X_val = get_stratified_TrainValFrames(path_in,
                                                   test_size = .1)

    #Save
    X_train.to_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,"X_train.csv"))
    X_val.to_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,"X_val.csv"))

    print(X_val.sign.value_counts())


    fig, ax = plt.subplots(1, figsize=(6, 13.4))
    sns.barplot(df_in.sign.value_counts().reset_index(),
                y="index",
                x="sign")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
