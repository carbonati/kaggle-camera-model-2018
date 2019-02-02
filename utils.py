import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import datetime


def save_full_train(src_dir, train_dir, filename_out, extra_dir=None):
    """Saves all train image paths to a DataFrame with corresponding labels"""
    TRAIN_PATH = os.path.join(src_dir, train_dir)
    if extra_dir:
        EXTRA_PATH = os.path.join(src_dir, extra_dir)
        train_path_list = [TRAIN_PATH, EXTRA_PATH]
    else:
        train_path_list = [TRAIN_PATH]
    
    df_full = pd.DataFrame()
    for path in train_path_list:
        for dir_name in os.listdir(path):
            DIR_PATH = os.path.join(path, dir_name)
            filenames = os.listdir(DIR_PATH)
            filepaths = [os.path.join(DIR_PATH, fn) for fn in filenames]
            target = target_map[dir_name]
            df_tmp = pd.DataFrame({'image_path': filepaths,
                                   'label': target})
            df_full = pd.concat((df_full, df_tmp))
    
    path_out = os.path.join(src_dir, filename_out)
    print("Saving full train DataFrame to {}".format(path_out))
    df_full.to_csv(path_out, index=False)
    

def save_kfold_data(src_dir, train_fn, dir_out, k=5):
    """Saves `k` fold train/validation DataFrames to `dir_out`"""
    dir_path_out = os.path.join(src_dir, dir_out)
    if not os.path.exists(dir_path_out):
        os.mkdir(dir_path_out)
    
    skf = StratifiedKFold(n_splits=5)
    
    skf_gen = skf.split(df_full['image_path'], df_full['label'])
    for fold_id, (train_ind, val_ind) in enumerate(skf_gen):
        df_train = df_full.iloc[train_ind]
        df_val = df_full.iloc[val_ind]

        csv_suffix = '_{}.csv'.format(fold_id + 1)
        filename = os.path.join(dir_path_out, 'train' + csv_suffix)

        # save dataframes to disk
        df_train.to_csv(os.path.join(dir_path_out, 'train' + csv_suffix),
                        index=False)
        df_val.to_csv(os.path.join(dir_path_out, 'val' + csv_suffix),
                      index=False)


def prep_test_data(src_dir, test_dir, test_data_name):
    TEST_PATH = os.path.join(src_dir, test_dir)
    filenames = os.listdir(TEST_PATH)
    files = [os.path.join(TEST_PATH, fn) for fn in filenames]

    df = pd.DataFrame({'image_path': files, 'label': -1})
    
    test_data_name = 'test.csv'
    test_data_path = os.path.join(src_dir, test_data_name)
    print("Saving full test DataFrame to {}".format(test_data_path))
    df.to_csv(test_data_path, index=False)