import sklearn
import pandas as pd

from experiments.data_mimic import DatasetExtrap, DatasetSynthExtrap


def load_tvt(args, path_synthetic):

    data_synthetic = pd.read_csv(path_synthetic/'synthetic.csv', index_col=0)

    if args.num_samples != -1:
        tvt_ids = pd.DataFrame(data_synthetic.index.unique(), columns=['ID']).sample(
            n=args.num_samples, random_state=args.random_state)
        data_tvt = data_synthetic.loc[tvt_ids['ID']]
    else:
        data_tvt = data_synthetic

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


def get_synthetic_tvt_datasets(args, proj_path, logger):
    path_synthetic = proj_path/'data/synthetic'
    data_train, data_validation, data_test = load_tvt(
        args, path_synthetic)

    if args.ml_task == "syn_extrap":
        val_options = {"N_val": 100,
                       "max_val_samples": args.next_headn}
        train = DatasetSynthExtrap(data_train.reset_index(), val_options)
        val = DatasetSynthExtrap(data_validation.reset_index(), val_options)
        test = DatasetSynthExtrap(data_test.reset_index(), val_options)

    else:
        raise ValueError("Unknown ML mode!")

    return train, val, test
