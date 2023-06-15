import sys

sys.path.insert(0, "./..")

from config import *

from src.illustrations.training_progress import  read_experiments
import pandas as pd
import os

def get_all_results(root_dir):
    # Initialize an empty list to store the directory paths
    directories = []

    # Recursively traverse the root directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if any .experiment file exists in the current directory
        if any(file.endswith('experiment') for file in filenames):
            directories.append(dirpath)

    # Print the directory paths
    for directory in directories:
        print(directory)

    return directories

if __name__ == '__main__':
    dirs = get_all_results(os.path.join(__file__, ROOT_PATH,RUNS_DIR))
    df = read_experiments(dirs)

    df.loc[df['Accuracy/Validation'] == df.groupby(by="Experiment")['Accuracy/Validation'].transform(max)]


    df_test = df.dropna(subset=['Accuracy/Test']).set_index("Experiment")
    df_test = df_test.loc[:, ~(df_test.columns.str.contains("Train") | df_test.columns.str.contains("Val"))]
    df_test.sort_values(by=['Accuracy/Test'],inplace=True)
    df_test.to_excel(os.path.join(ROOT_PATH, OUT_DIR, "Results.xlsx"))

    print(df_test.loc[:,df_test.columns.str.contains("Test")])

    df_export = df_test.loc[df_test['Accuracy/Test'] == df_test.groupby(by="MODELNAME")["Accuracy/Test"].transform(max)]
    rename_dict = dict({
        'Accuracy/Test': "Acc",
        'Loss/Test': "Loss",
        'Precision/Test': "Prec.",
        'F1Score/Test': "F1",
    })
    df_export = df_export.rename(columns=rename_dict)

    df_export = df_export[["MODELNAME"] + list(rename_dict.values())]
    df_export.select_dtypes(include=[float]).apply(lambda x: round(x, 2)).to_csv(os.path.join(ROOT_PATH,OUT_DIR,"Summary.csv"),index_label = "MODEL_NAME")
    print(df_export.reset_index(drop=True))
    print("done")
