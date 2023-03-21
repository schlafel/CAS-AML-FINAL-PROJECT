import time

import pandas as pd
import os
from config import *
from multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np
import datetime
def analyze_file(q,q_out,i):

    while True:
        item = q.get()
        if item is None:
            print("Process {:d}: Queue is empty".format(i))
            break
        idx,path = item
        df_in = pd.read_parquet(path)

        max_length = df_in.shape[0]/543
        q_out.put((idx,path,max_length))

        # finished = q_out.qsize()/q.qsize() *100
        # if finished%10 == 0:
        #     print()

    # print("Process:",i)
    return


def start_mp( df_train,   n_processes = 10):
    q = Queue()
    q_out = Queue(maxsize=df_train.shape[0])
    #Fill Queue

    print("Filling Queue...")
    for _,row in tqdm(df_train.iterrows(),total = df_train.shape[0]):
        q.put((_,os.path.join(ROOT_PATH,RAW_DATA_DIR,row.path)))


    procs = []
    for i in range(0,n_processes):
        p = Process(target=analyze_file, args=(q,q_out,i,))
        procs.append(p)
    t0 = datetime.datetime.now()
    print("Start Processes",t0)
    for p in procs:
        q.put(None)
        p.start()
    print("all Prcesses started")


    #retrieve elements
    _liout = []
    while not (q_out.empty() & q.empty()):
        _liout.append(q_out.get())

    print("Emptied Queue....")
    # print("Maximum Sequence Length:", np.array(_liout).max(axis=0)[1])


    #now join all the processes
    for p in procs:
        p.join()
    t1 = datetime.datetime.now()



    #now retrieve Queue
    print("Empty Queue....")
    df_out = pd.DataFrame(np.array(_liout), columns=["idx", "path", "n_frames"])
    df_out["idx"] = df_out["idx"].map(int)
    df_out["n_frames"] = df_out["n_frames"].map(float).map(int)
    return df_out.sort_values(by="idx")







if __name__ == '__main__':

    df_train = pd.read_csv(os.path.join(ROOT_PATH,RAW_DATA_DIR,"train.csv"))
    print("done", df_train.shape)

    df_out = start_mp(df_train, n_processes=10)
    print(df_out["n_frames"].max())

    df_out.to_csv("sequence_length_dataset.csv",index = False)








