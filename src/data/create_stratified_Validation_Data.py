import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # Load your data into a pandas DataFrame
    df_in = pd.read_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,TRAIN_CSV_FILE))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df_in,
                                                      df_in['target'],
                                                      test_size=0.1,
                                                      random_state=42,
                                                      stratify=df_in['target'])

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