import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from utils import datasets, scoring, plot

DATASET_NAME = "ransomware"
DATASET_NAME_2 = "benign"
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


def load_data(logger):
    benign = datasets.load_all(os.path.join("datasets", DATASET_NAME_2))  # load dataset from csv
    ransomware = datasets.load_all(os.path.join("datasets", DATASET_NAME))  # load dataset from csv
    logger.info("{} {}".format("benign shape", benign.shape))
    logger.info("{} {}".format("ransomware shape", ransomware.shape))

    benign = datasets.prepare_dataset(benign, shuffle=True)
    ransomware = datasets.prepare_dataset(ransomware, shuffle=True)

    n_elements = min(benign.shape[0], ransomware.shape[0], 150000)

    benign = benign.head(n_elements)
    ransomware = ransomware.head(n_elements)

    logger.info("{} {}".format("benign shape after balancing", benign.shape))
    logger.info("{} {}".format("ransomware shape after balancing", ransomware.shape))

    ransomware["Label"] = DATASET_NAME.upper()

    return pd.concat([benign, ransomware], ignore_index=True)  # union dataset


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
    loaded_dataset = load_data(logger)
    logger.info("{} {}".format("loaded_dataset shape", loaded_dataset.shape))

    # loaded_dataset["Label"] = DATASET_NAME.upper()

    logger.info(loaded_dataset.head())
    loaded_dataset.info()

    dataset = None

    logger.info("{} {}".format("Dataset shape BEFORE preparation", loaded_dataset.shape))
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       drop_columns=["Flow Bytes/s", "Flow Packets/s", "Flow ID", "Source IP",
                                                     "Destination IP", "Timestamp", "Fwd Header Length.1"],
                                       shuffle=True, dropna=True)

    loaded_dataset = None

    logger.info("{} {}".format("Dataset shape AFTER preparation", dataset.shape))

    xTest, yTest = datasets.separate_labels(dataset, encode=True)

    dataset = None

    logger.info('Scaling dataset')
    xTest = datasets.drop_variance(xTest)
    standardScaler = StandardScaler()
    xTestScaled = standardScaler.fit_transform(xTest)

    logger.info("Performing PCA")
    pca = PCA(random_state=42, n_components=0.95)
    xTestPCA = pca.fit_transform(xTest)
    logger.info("Dataset shape with PCA {}".format(xTestPCA.shape))

    results_array = []

    logger.info("Logistic Regression")
    log_reg = LogisticRegression(verbose=1, n_jobs=-1, random_state=42, max_iter=1000)
    results = scoring.cross_validate_scoring(log_reg, xTest, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Logistic Regression")
    logger.info(results)

    logger.info("Logistic Regression PCA")
    log_reg = LogisticRegression(verbose=1, n_jobs=-1, random_state=42, max_iter=1000)
    results = scoring.cross_validate_scoring(log_reg, xTestPCA, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Logistic Regression PCA")
    logger.info(results)

    logger.info("SVC Classifier")
    linearSvc = LinearSVC(random_state=42, verbose=1)  # svc classifier
    results = scoring.cross_validate_scoring(linearSvc, xTest, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)
    save_result2(results_array, results, "SVC Normal")
    logger.info(results)

    logger.info("SVC Classifier PCA")
    linearSvc = LinearSVC(random_state=42, verbose=1)  # svc classifier
    results = scoring.cross_validate_scoring(linearSvc, xTestPCA, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)
    save_result2(results_array, results, "SVC Normal PCA")
    logger.info(results)

    logger.info("SVC Classifier Scaled")
    linearSvc = LinearSVC(random_state=42, verbose=1)        #svc classifier
    results = scoring.cross_validate_scoring(linearSvc, xTestScaled, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)
    save_result2(results_array, results, "SVC Scaled")
    logger.info(results)

    logger.info("Decision Tree")
    dec_tree = DecisionTreeClassifier(random_state=42)
    results = scoring.cross_validate_scoring(dec_tree, xTest, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Decision Tree")
    logger.info(results)

    logger.info("Decision Tree PCA")
    dec_tree = DecisionTreeClassifier(random_state=42)
    results = scoring.cross_validate_scoring(dec_tree, xTestPCA, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Decision Tree PCA")
    logger.info(results)

    logger.info("Random Forest")
    rnd_forest = RandomForestClassifier(random_state=42, verbose=1, n_jobs=-1)
    results = scoring.cross_validate_scoring(rnd_forest, xTest, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Random Forest")
    logger.info(results)

    logger.info("Random Forest PCA")
    rnd_forest = RandomForestClassifier(random_state=42, verbose=1, n_jobs=-1)
    results = scoring.cross_validate_scoring(rnd_forest, xTestPCA, yTest, cv=3,
                                             scoring=['roc_auc', 'f1', 'roc', 'precision', 'recall'],
                                             return_train_score=True)

    save_result2(results_array, results, "Random Forest PCA")
    logger.info(results)

    console_handler.close()
    file_handler.close()


def show(scor='f1'):
    results = datasets.pk_load(RESULTS_FOLDER_PATH, 'results')
    plot.plt_cross_validate_results(results, scor, 'Models ROC')


if __name__ == "__main__":
    # calc()
    show('f1')


