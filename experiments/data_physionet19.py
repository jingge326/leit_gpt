import os
import zipfile
import csv
import urllib.request

import numpy as np
import pandas as pd
import sklearn
from experiments.data_mimic import DatasetBiClass, scale_tvt


def download_p19(path_p19):
    loc_Azip = path_p19/'training_setA.zip'
    loc_Bzip = path_p19/'training_setB.zip'
    if not os.path.exists(loc_Azip):
        if not os.path.exists(path_p19):
            os.mkdir(path_p19)
        urllib.request.urlretrieve(
            'https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
            str(loc_Azip),)
        urllib.request.urlretrieve(
            'https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
            str(loc_Bzip),)

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(str(path_p19))
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(str(path_p19))
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(path_p19/folder):
                if os.path.exists(path_p19/filename):
                    raise RuntimeError
                os.rename(path_p19/folder/filename, path_p19/filename)


def process_p19(path_p19, path_processed):
    os.mkdir(path_processed)
    time_series_list = []
    times = []
    label_list = []
    ids_dup_list = []
    ids_list = []
    nan_list = ['NaN'] * 34
    for filename in os.listdir(path_p19):
        if filename.endswith('.psv'):
            with open(path_p19/filename) as file:
                label = 0.0
                reader = csv.reader(file, delimiter='|')
                reader = iter(reader)
                next(reader)  # first line is headings
                for line in reader:
                    assert len(line) == 41
                    ts_values = line[:34]
                    if ts_values == nan_list:
                        # Acturally, the first row of each file doesn't contain time series
                        continue
                    timestamp = float(line[39])
                    sepsislabel = line[40]
                    if timestamp > 72:  # keep at most the first three days
                        break
                    label = max(label, float(sepsislabel))
                    time_series_list.append(ts_values)
                    times.append(timestamp)
                    ids_dup_list.append(filename[:-4])

                label_list.append(label)
                ids_list.append(filename[:-4])

    arr_values = np.array(time_series_list).astype(np.float)
    arr_mask = (~np.isnan(arr_values)).astype(np.float)
    arr_values = np.nan_to_num(arr_values)

    df_times = pd.DataFrame(times, columns=['Time'])
    df_values = pd.DataFrame(arr_values, columns=[
        'Value_'+str(i) for i in range(34)])
    df_mask = pd.DataFrame(
        arr_mask, columns=['Mask_'+str(i) for i in range(34)])
    df_labels = pd.DataFrame(label_list, columns=['labels'])

    df_p19_data = pd.concat([pd.DataFrame(ids_dup_list, columns=[
                            'ID']), df_times, df_values, df_mask], axis=1)
    df_p19_labels = pd.concat(
        [pd.DataFrame(ids_list, columns=['ID']), df_labels], axis=1)

    df_p19_data.to_csv(path_processed/'p19_data.csv', index=False)
    df_p19_labels.to_csv(path_processed/'p19_labels.csv', index=False)


def load_tvt(args, path_p19):
    path_processed = path_p19/'processed_data'
    if os.path.exists(path_p19):
        pass
    else:
        download_p19(path_p19)

    if os.path.exists(path_processed):
        pass
    else:
        process_p19(path_p19, path_processed)

    data_p19 = pd.read_csv(path_processed/'p19_data.csv', index_col=0)

    if args.num_samples != -1:
        tvt_ids = pd.DataFrame(data_p19.index.unique(), columns=['ID']).sample(
            n=args.num_samples, random_state=args.random_state)
        data_tvt = data_p19.loc[tvt_ids['ID']]
    else:
        data_tvt = data_p19

    # Within the 72 hours stay, there is no observation at time 0 and no time series at time 1
    # - 2.0 to make the timestamp of the first observation as 0 which is the same as M4 and P12
    data_tvt['Time'] = data_tvt['Time'] - 2.0

    # Splitting
    ids_train, ids_vt = sklearn.model_selection.train_test_split(
        data_tvt.index.unique(),
        train_size=0.8,
        random_state=args.random_state,
        shuffle=True)

    ids_valid, ids_test = sklearn.model_selection.train_test_split(
        ids_vt,
        train_size=0.5,
        random_state=args.random_state,
        shuffle=True)

    data_train = data_tvt.loc[ids_train]
    data_validation = data_tvt.loc[ids_valid]
    data_test = data_tvt.loc[ids_test]
    return data_train, data_validation, data_test


def get_p19_tvt_datasets(args, proj_path, logger):
    path_p19 = proj_path/'data'/'PhysioNet19'
    data_train, data_validation, data_test = load_tvt(args, path_p19)
    data_train, data_validation, data_test = scale_tvt(
        args, data_train, data_validation, data_test)

    label_data = pd.read_csv(
        proj_path/'data'/'PhysioNet19'/'processed_data'/'p19_labels.csv')

    train = DatasetBiClass(
        data_train.reset_index(), label_df=label_data, ts_full=args.ts_full)
    val = DatasetBiClass(
        data_validation.reset_index(), label_df=label_data, ts_full=args.ts_full)
    test = DatasetBiClass(
        data_test.reset_index(), label_df=label_data, ts_full=args.ts_full)

    return train, val, test
