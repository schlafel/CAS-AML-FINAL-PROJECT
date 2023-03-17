import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as config
import os
import json
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from tqdm import tqdm

import mediapipe as mp
mp_pose = mp.solutions.pose


def data_checks(ASL_DATASET):
    for _i, (df,lab) in tqdm(enumerate(ASL_DATASET),total = len(ASL_DATASET)):
        ASL_DATASET.df_train.loc[_i,"n_frames"] = len(df.frame.value_counts())
        ASL_DATASET.df_train.loc[_i,"average_landmarks"] = (df.frame.value_counts()).mean()
        ASL_DATASET.df_train.loc[_i,"std_landmarks"] = (df.frame.value_counts()).std()

    ASL_DATASET.df_train.to_csv(r"Statisics_dataset.csv")


def convert_df_to_result(df):
    #iterate through each frame
    for frame_no, df_frame in df.groupby(by="frame"):
        print("done")

    pass
class ASL_DATSET(Dataset):
    def __init__(self,_path, transform = None):
        super().__init__()
        self.transform = transform
        self.path = _path

        #Function to load csv
        self.load_datas()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self,new_path):
        self._path = new_path
        self._load_label_map()
        self.load_datas()


    def _load_label_map(self):
        with open(config.PATH_LABELMAP, "r") as infile:
            ln = infile.readline()
        self.label_dict = json.loads(ln)
        self.label_dict_inv = {y:x for x,y in self.label_dict.items() }




    def load_datas(self):
        self.df_train = pd.read_csv(self._path)
        #generate Absolute path
        self.file_paths = np.array([os.path.join(config.PATH_DATA,x) for x in self.df_train["path"].values])

        self.participant_id = self.df_train["participant_id"].values
        self.sequence_id = self.df_train["sequence_id"].values
        self.sign = self.df_train["sign"].values

        self.sign_int = self.df_train["sign"].map(self.label_dict).values
        self.sign_oneHot = F.one_hot(torch.from_numpy(self.sign_int))




    def __getitem__(self, idx):
        x = self.open_parquet(self.file_paths[idx])
        label = self.sign_oneHot[idx]
        return x, label

    def __len__(self):
        return len(self.sign_int)


    @staticmethod
    def open_parquet(file_path):
        return pd.read_parquet(file_path)






class ASL_DATAModule(pl.LightningDataModule):
    def __init__(self,config,
                 batch_size = 8,
                 ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size

        self.num_workers = 1

    # @config.setter
    # def config(self,x):
    #     self.config = x

    @property
    def config(self,):
        return self._config

    @config.setter
    def config(self,_config):
        self._config = _config
        #reload also the label_dict
        self._load_label_map()
        self._getTrainCSV()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train = ASL_DATSET(self.train_dir,
                                    transform = None,)
            # self.validate = StoneDataset(self.val_dir,transform = self.transform)
        if stage == "test":
            # self.test = StoneDataset(self.test_dir,transform = self.transform)
            pass

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size,
                          num_workers=self.num_workers,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_workers,shuffle=True)








if __name__ == '__main__':
    print("done")

    asl_train_ds = ASL_DATSET(config.PATH_TRAIN_DATA)

    data_checks(asl_train_ds)

    df, label = next(iter(asl_train_ds))
    convert_df_to_result(df)
    print("done")

    # # label_dict = load_label_map(config.PATH_LABELMAP)
    # # df_train = pd.read_csv(os.path.join(config.PATH_DATA,"train.csv"))
    # #
    # dM = ASL_DATAModule(config = config)
    #
    #
    # fig,ax = plt.subplots(1)
    # ax.scatter(1,1)
    # plt.show()
    #
    #
