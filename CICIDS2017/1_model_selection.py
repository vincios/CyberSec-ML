import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import datasets, scoring, plot

DATASET_NAME = "ddos"
RESULTS_FOLDER_PATH = os.path.join("results", DATASET_NAME, "1_model_selection")


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
    datasets.np_double_save(results_array, RESULTS_FOLDER_PATH, 'results', as_csv=True)


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
    loaded_dataset = datasets.load_all(os.path.join("datasets"))  # load dataset from csv
    logger.info("{} {}".format("loaded_dataset shape", loaded_dataset.shape))


    #loaded_dataset["Label"] = DATASET_NAME.upper()

    logger.info(loaded_dataset.head())
    loaded_dataset.info()

    dataset = None

    logger.info("{} {}".format("Dataset shape BEFORE preparation", loaded_dataset.shape))
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       drop_columns=["Flow Bytes/s", "Flow Packets/s", "Fwd Header Length.1"],
                                       shuffle=True, dropna=True)

    loaded_dataset = None

    logger.info("{} {}".format("Dataset shape AFTER preparation", dataset.shape))

    xTest, yTest = datasets.separate_labels(dataset, encode=True)

    dataset = None

    xTest = datasets.drop_variance(xTest)
    standardScaler = StandardScaler()
    xTestScaled = standardScaler.fit_transform(xTest)

    results_array = []

    logger.info("Logistic Regression")
    log_reg = LogisticRegression(verbose=1, n_jobs=-1, max_iter=1000)
    results = scoring.cross_validate_scoring(log_reg, xTest, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Logistic Regression")
    logger.info(results)

    try:
        logger.info("SVC Classifier")
        linearSvc = LinearSVC(verbose=1)  # svc classifier
        results = scoring.cross_validate_scoring(linearSvc, xTest, yTest, cv=3,
                                                 scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                                 return_train_score=True)
        save_result2(results_array, results, "SVC Normal")
        logger.info(results)
    except Exception as e:
        logger.warning(e)

    logger.info("SVC Classifier Scaled")
    linearSvc = LinearSVC(verbose=1)        #svc classifier
    results = scoring.cross_validate_scoring(linearSvc, xTestScaled, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)
    save_result2(results_array, results, "SVC Scaled")
    logger.info(results)


    console_handler.close()
    file_handler.close()


def show(scor='f1'):
    results = datasets.pk_load(RESULTS_FOLDER_PATH, 'results')
    plot.plt_cross_validate_results(results, scor, 'Models ROC')


if __name__ == "__main__":
    # calc()
    show('roc_auc')


