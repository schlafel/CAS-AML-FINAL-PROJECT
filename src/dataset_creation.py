from torch.utils.data import Dataset, DataLoader
from data_models import ASL_DATSET, ASLDataModule
import pandas as pd
from config import *
from tqdm import tqdm
import os

if __name__ == '__main__':
    MAX_SEQUENCES = 150 #Fix to 150!!

    datset = ASL_DATSET(max_seq_length=MAX_SEQUENCES)
    PROCESSED_DATA_DIR2 = r"data/processed_v2"
    #.0-
    for idx,files in tqdm(enumerate(datset.file_paths),total = len(datset.file_paths)):

        sample = datset[idx]
        #split dirs/files
        f_name = os.path.splitext(os.path.basename(files))[0]
        subdir = os.path.basename(os.path.dirname(files))

        savedir = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR2,subdir)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir,f_name+".pt"),"wb") as infile:
            torch.save(sample,infile)


        #pass
        # print("")








