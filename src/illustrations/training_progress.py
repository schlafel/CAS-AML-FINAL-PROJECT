import sys

sys.path.insert(0, "./..")

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
import tensorflow as tf
import glob
import pandas as pd

import os

from config import *


def load_tf(dirname):
    """
    Function to load tensorbaord-logs as a pd.DataFrame.

    Args:
        dirname (str): Which tensorbaord-log should be loaded (it loads all the checkpoints

    Functionality:
        Searches for tensorboard logs and loads scalars into a pd Dataframe

    :param dirname:
    :return:
    """

    dirname = glob.glob(dirname + '/*')[0]

    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']

    for n in mnames:
        dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')])
        dframes[n].drop("wall_time", axis=1, inplace=True)


    df_out = pd.concat([v for k, v in dframes.items()], axis=1)

    #get the hparams
    data = ea._plugin_to_tag_to_content['hparams']["_hparams_/session_start_info"]
    hparam_data = HParamsPluginData.FromString(data).session_start_info.hparams
    hparam_dict = {key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()}

    #accumulate the hparams in dataframe as well
    for key,value in hparam_dict.items():
        df_out[key] = value
    #drop duplicated epoch columns
    df_filt = df_out.loc[:, ~df_out.columns.duplicated()]

    return df_filt

def plot_training_validation(data,
                             x="epoch",
                             y="Accuracy",
                             style="Metric",
                             hue = "Experiment_FullName",
                             y_lim = (0,1)):
    """

    :param data: pd.DataFrame
    :param x:
    :param y:
    :param style:
    :param hue:
    :param y_lim:
    :return:
    """

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18 / 2.54, 3.8
                                                 ),
                                  sharex=True)
    sns.lineplot(data=data.loc[data.Metric.str.contains("Accuracy")],
                 x=x,
                 y=y,
                 hue=hue,
                 style=style,
                 ax=ax,
                 legend=True,
                 linewidth = 2)
    sns.lineplot(data=data.loc[data.Metric.str.contains("Loss")],
                 x=x,
                 y=y,
                 hue=hue,
                 style=style,
                 ax=ax2,
                 legend=False,
                 linewidth = 2)
    ax.set_ylim(y_lim)
    ax.set_ylabel("Accuracy [-]")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
    ax2.set_ylabel("Loss [-]")
    ax.set_xlabel("Epoch [-]")
    ax2.set_xlabel("Epoch [-]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=[0.5, 1.], ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(top=.8)

    return fig,(ax,ax2)

def plot_trainingLossAccuracies(ckpt_paths):
    """
    Function that plots training accuracy and loss based on checkpoints that are entered in a list


    :param ckpt_paths: List of Checkpoints to evaluate. Simply put the directory in, where tensorboard checkpoints are located
    :type ckpt_paths: list
    :return: figure object (fig,(ax,ax1))
    """
    concat_df = read_experiments(ckpt_paths)

    melted_df_train_val = pd.melt(concat_df,
        id_vars=['epoch','Experiment',],
        value_vars=[
            f'{"Accuracy"}/Train',f'{"Accuracy"}/Validation',
            f'{"Loss"}/Train',f'{"Loss"}/Validation',
                    ],
        var_name="Metric")

    # melted_dftest = pd.melt(concat_df,
    #                               id_vars=['epoch', 'Experiment', ],
    #                               value_vars=[
    #                         f'{"Accuracy"}/Test', f'{"Accuracy"}/Test',
    #                         f'{"Loss"}/Test', f'{"Loss"}/Test',
    #                          ],
    #                               var_name="Metric")

    fig,ax = plot_training_validation(data = melted_df_train_val,
                                      y="value",
                                      style = "Metric",
                                      hue = "Experiment")
    return fig,ax


def read_experiments(ckpt_paths):
    df_list = []
    for path in ckpt_paths:
        df = load_tf(path)
        df['Exp'] = os.path.basename(path)
        df['model'] = os.path.basename(os.path.split(path)[0])
        df["DL_FRAMEWORK"] = os.path.basename(os.path.split(os.path.split(path)[0])[0])
        df['Experiment'] = df.DL_FRAMEWORK + "_" + df.model + "_" + df.Exp
        # Drop duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]

        df_list.append(df)
    concat_df = pd.concat(df_list, ignore_index=True)
    return concat_df


if __name__ == '__main__':

    save_name = "compare_"
    ckpt_paths = [
        os.path.join(ROOT_PATH,r"runs\pytorch\LSTMPredictor\2023-05-29 00_56"),
        os.path.join(ROOT_PATH,r"runs/tensorflow/LSTMPredictor/2023-05-28 16_00"),
        os.path.join(ROOT_PATH,r"runs/pytorch/HybridModel/2023-06-09 23_26"),
    ]

    ckpt_paths = [
        os.path.join(ROOT_PATH,r"runs/pytorch/HybridEnsembleModel/2023-06-14 09_27",)
    ]
    fig,ax = plot_trainingLossAccuracies(ckpt_paths)

    fig.savefig(os.path.join(ROOT_PATH,OUT_DIR,f"{save_name}.svg"))

    plt.show()

