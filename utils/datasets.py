import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold


def np_load_data(directory, filename):
    """
        Loads numpy ndarray form a npy file given a folder
        Return a numpy ndarray
    """

    npy_path = os.path.join(directory, filename)
    return np.load(npy_path, allow_pickle=True)


def pd_load_data(filename, directory):  # define functions load
    """
        Loads pandas DataFrame form a csv file given a folder
        Return a pandas DataFrame
    """
    csv_path = os.path.join(directory, filename)
    return pd.read_csv(csv_path, skipinitialspace=True)


def load_all(directory):
    """
        Loads all csv files contained in a specified folder and merge them in a single pandas DataFrame
    """
    final_dataset = None  # final dataset
    files = os.listdir(directory)
    for path in files:
        if os.path.isfile(path):
            data = pd_load_data(path, directory)  # Base
        else:
            data = load_all(path)  # Recursively load dataset in folders

        if final_dataset is None:
            final_dataset = data
        else:
            final_dataset = pd.concat([final_dataset, data], ignore_index=True)
    return final_dataset


def prepare_dataset(dataset: pd.DataFrame, drop_columns=None, shuffle=False, dropna=False):
    """
    Pre-process a dataset following specified parameters

    :param dataset: pandas DataFrame to prepare
    :param drop_columns: list of columns names to delete from database (default: None)
    :param shuffle: if should shuffle rows in the DataFrame (default: False)
    :param dropna: if should remove rows containing any N/A values (default: False)
    :return: the prepared dataset
    """
    new_dataset = dataset.copy()

    if drop_columns is not None:
        for column in drop_columns:
            new_dataset = new_dataset.drop(column, axis=1)

    if dropna:
        new_dataset = new_dataset.dropna()

    if shuffle:
        new_dataset = new_dataset.reindex(np.random.permutation(new_dataset.index))

    return new_dataset


def separate_labels(dataset, column_name="Label", encode=False):
    """
    Separate labels' column from the dataset. Optionally encode categorical labels into numbers

    :param dataset: starting dataset
    :param column_name: labels column's name (default 'Label')
    :param encode: if should encode categorical labels
    :return: dataset WITHOUT labels and labels separated from dataset
    """
    labels = dataset[column_name]
    new_dataset = dataset.drop(column_name, axis=1)

    if encode:
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    return new_dataset, labels


def drop_variance(dataset, threshold=0):
    """
    Remove columns from dataset with variance below a given threshold

    :param dataset: starting dataset
    :param threshold: Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.
    :return:
    """
    variance_thres = VarianceThreshold(threshold=threshold)
    return variance_thres.fit_transform(dataset)


def pd_save_data_csv(dataset: pd.DataFrame, foldername, filename):
    """
    Save a pandas DataFrame as csv

    :param dataset: dataset to save
    :param foldername: target folder
    :param filename: file name WITHOUT the '.csv' extension
    """
    filepath = os.path.join(foldername, filename)
    filepath = filepath + ".csv"
    dataset.to_csv(filepath)


def np_double_save(ndarray, foldername, filename, as_npy=True, as_csv=False):
    """
    Save a numpy ndarray as npy and/or csv file

    :param ndarray: array to save
    :param foldername: target folder
    :param filename: file name WITHOUT the extension
    :return:
    """
    file_path = os.path.join(foldername, filename)
    if as_csv:
        np.savetxt(file_path + ".csv", ndarray, delimiter=",", fmt="%s")

    if as_npy:
        np.save(file_path, ndarray)