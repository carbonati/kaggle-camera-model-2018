import os
import numpy as np
import pandas as pd
import datetime


def create_kfold_data(src_dir, train_fn, k=5):
    for fold_id in range(k):
        DATA_PATH = os.path.join(src_dir, train_fn)
        FOLD_DIR_NAME = "fold_{0}".format(fold_id + 1)
        FOLD_PATH = os.path.join(src_dir, FOLD_DIR_NAME)
        if not os.path.exists(FOLD_PATH):
            os.mkdir(FOLD_PATH)
        
        train_test_even_split(DATA_PATH, shuffle=True, path_out=FOLD_PATH)

        
def train_test_even_split(path_in, train_frac=0.9, shuffle=False, path_out=''):
    df = pd.DataFrame()

    for label, dir_fn in enumerate(os.listdir(path_in)):
        dir_fp = os.path.join(path_in, dir_fn)
        files = [os.path.join(dir_fp, fn) for fn in os.listdir(dir_fp)]
        df_tmp = pd.DataFrame({'image_path': files, 'label': label})
        df = pd.concat((df, df_tmp))

    n_classes = df['label'].nunique()
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    N = df.shape[0]
    n_train = np.floor(N * train_frac).astype(int)
    n_per_class = np.floor(n_train / n_classes).astype(int)

    df_train = df.groupby('label')['image_path'].apply(lambda x: x.iloc[:n_per_class])
    df_train = df_train.reset_index().drop('level_1', axis=1)
    df_val = df.groupby('label')['image_path'].apply(lambda x: x.iloc[n_per_class:])
    df_val = df_val.reset_index().drop('level_1', axis=1)
    
    csv_suffix = datetime.datetime.now().strftime('_%Y_%m_%d.csv')
    train_fp = os.path.join(path_out, 'train' + csv_suffix)
    val_fp = os.path.join(path_out, 'val' + csv_suffix)
    
    # save cv dataframes to disk
    df_train.to_csv(train_fp, index=False)
    df_val.to_csv(val_fp, index=False)