import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as config
import os
import json
import torch
from torch.utils.data import Dataset,DataLoader

import lightning.pytorch as pl


class ASL_DATSET(Dataset):
    def __init__(self,_path,label_dict, transform = None):
        super().__init__()
        self.transform = transform
        self._path = _path
        self.label_dict = label_dict
        self.label_dict_inv = {y:x for x,y in self.label_dict.items() }

        #Function to load csv
        self.load_datas()

    @property
    def _path(self):
        return self._path

    @_path.setter
    def _path(self,new_path):
        self._path = new_path
        self.load_datas()




    def load_datas(self):
        self.df_train = pd.read_csv(self._path)
        #generate Absolute path
        self.file_paths = np.array([os.path.join(config.PATH_DATA,x) for x in self.df_train["path"].values])

        self.participant_id = self.df_train["participant_id"].values
        self.sequence_id = self.df_train["sequence_id"].values
        self.sign = self.df_train["sign"].values


    def __getitem__(self, idx):

        x = self.open_parquet(self.file_paths[idx])
        label = self.img_labels[idx]


        return x, label

    def __len__(self):
        return len(self.img_labels)



    def open_parquet(self):
        pd.read_parquet
        pass





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


    def _load_label_map(self):
        with open(self.config.PATH_LABELMAP, "r") as infile:
            ln = infile.readline()
        self.label_dict = json.loads(ln)






if __name__ == '__main__':
    print("done")

    # label_dict = load_label_map(config.PATH_LABELMAP)
    # df_train = pd.read_csv(os.path.join(config.PATH_DATA,"train.csv"))
    #
    dM = ASL_DATAModule(config = config)


    fig,ax = plt.subplots(1)
    ax.scatter(1,1)
    plt.show()


