import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, validation_curve

from utils import datasets, scoring, plot

DATASET_NAME = "vpn"
RESULTS_FOLDER_PATH = os.path.join("results", DATASET_NAME, "2_model_tuning")


def save_result(roc_curve, auc_score, classifier_name, rocs_array, auc_scores_array):
    roc_curve.insert(0, classifier_name)
    auc_score.insert(0, classifier_name)
    rocs_array.append(roc_curve)
    auc_scores_array.append(auc_score)

    # plt_add_roc_curve(fpr, tpr, label=str(n_estimators))

    np_roc_array = np.array(rocs_array)
    np_roc_auc_scores = np.array(auc_scores_array)

    datasets.np_double_save(np_roc_array, RESULTS_FOLDER_PATH,
                            "roc_fpr_tpr_thres", as_csv=True, as_npy=True)
    datasets.np_double_save(np_roc_auc_scores, RESULTS_FOLDER_PATH,
                            "roc_auc_scores", as_csv=True, as_npy=True)


def save_result2(results_array, results: dict, name: str):
    results['label'] = name
    results_array.append(results)
    datasets.pk_save(results_array, RESULTS_FOLDER_PATH, 'results')


def load_data():
    return datasets.load_all(os.path.join("datasets"))  # load dataset from csv


def calc():
    if not os.path.exists(RESULTS_FOLDER_PATH):
        os.makedirs(RESULTS_FOLDER_PATH)

    logfile = os.path.join(RESULTS_FOLDER_PATH, "log.log")
    if os.path.exists(logfile):
        os.remove(logfile)

    # logging stuff
    level = logging.INFO
    formats = {"console": '\u001b[37m %(message)s\033[0m', "file": '%(message)s'}

    file_handler, console_handler = logging.FileHandler(logfile, "x"), logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(formats["console"]))
    file_handler.setFormatter(logging.Formatter(formats["file"]))

    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # begin calc
    loaded_dataset = load_data()
    logger.info("{} {}".format("loaded_dataset shape", loaded_dataset.shape))
    logger.info(loaded_dataset['class1'].value_counts())

    # loaded_dataset["Label"] = DATASET_NAME.upper()

    logger.info(loaded_dataset.head())
    loaded_dataset.info()

    dataset = None

    logger.info("{} {}".format("Dataset shape BEFORE preparation", loaded_dataset.shape))
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       # drop_columns=["Flow Bytes/s", "Flow Packets/s", "Fwd Header Length.1"],
                                       shuffle=True, dropna_axis=[1])

    loaded_dataset = None

    logger.info("{} {}".format("Dataset shape AFTER preparation", dataset.shape))

    xTest, yTest = datasets.separate_labels(dataset, encode=True, column_name="class1")

    dataset = None

    xTest = datasets.drop_variance(xTest)
    standardScaler = StandardScaler()
    xTestScaled = standardScaler.fit_transform(xTest)
    results = []
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_name = "max_depth"
    param_range = [2 ** i for i in range(1, 11)]
    training_score, test_score = validation_curve(clf, xTest, yTest, param_name=param_name,
                                                  param_range=param_range,
                                                  scoring="roc_auc", cv=3, verbose=1, n_jobs=-1)

    results.append([param_name, param_range, training_score, test_score])
    datasets.np_double_save(results, RESULTS_FOLDER_PATH, "results", as_csv=True, as_npy=True)

    param_name = "min_samples_leaf"
    param_range = [i for i in range(1, 11)]
    print(param_range)
    training_score, test_score = validation_curve(clf, xTest, yTest, param_name=param_name,
                                                  param_range=param_range,
                                                  scoring="roc_auc", cv=3, verbose=1, n_jobs=-1)

    results.append([param_name, param_range, training_score, test_score])
    datasets.np_double_save(results, RESULTS_FOLDER_PATH, "results", as_csv=True, as_npy=True)

    param_name = "n_estimators"
    param_range = [i**2 for i in range(4, 25, 4)]
    print(param_range)
    training_score, test_score = validation_curve(clf, xTest, yTest, param_name=param_name,
                                                  param_range=param_range,
                                                  scoring="roc_auc", cv=3, verbose=1, n_jobs=-1)

    results.append([param_name, param_range, training_score, test_score])
    datasets.np_double_save(results, RESULTS_FOLDER_PATH, "results", as_csv=True, as_npy=True)

    # plot.plt_validation_curve(training_score, test_score, param_range)
    console_handler.close()
    file_handler.close()


def show():
    results = datasets.np_load_data(RESULTS_FOLDER_PATH, "results.npy")
    for result in results:
        plot.plt_validation_curve(result[2], result[3], result[1], result[0],
                                  plot_tile="Validation curve for VPN/NonVPN dataset with Random Forest")


if __name__ == "__main__":
    # calc()
    show()


