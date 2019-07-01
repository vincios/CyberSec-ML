import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import datasets, random_forest, plot

DATASET_NAME = "scareware"
DATASET_NAME_2 = "benign"
RESULTS_FOLDER_PATH = os.path.join("results", DATASET_NAME, "2_model_tuning")


def save_result(roc_curve, auc_score, parameter_name, parameter_value, rocs_array, auc_scores_array):
    roc_curve.insert(0, parameter_value)
    auc_score.insert(0, parameter_value)
    rocs_array.append(roc_curve)
    auc_scores_array.append(auc_score)

    # plt_add_roc_curve(fpr, tpr, label=str(n_estimators))

    np_roc_array = np.array(rocs_array)
    np_roc_auc_scores = np.array(auc_scores_array)

    save_dir = os.path.join(RESULTS_FOLDER_PATH, parameter_name)
    os.makedirs(save_dir, exist_ok=True)

    datasets.np_double_save(np_roc_array, save_dir,
                            "rnd_forest_roc_fpr_tpr_thres", as_csv=True, as_npy=True)
    datasets.np_double_save(np_roc_auc_scores, save_dir,
                            "rnd_forest_roc_auc_scores", as_csv=True, as_npy=True)


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
    benign = datasets.load_all(os.path.join("datasets", DATASET_NAME_2))  # load dataset from csv
    scareware = datasets.load_all(os.path.join("datasets", DATASET_NAME))  # load dataset from csv
    logger.info("{} {}".format("benign shape", benign.shape))
    logger.info("{} {}".format("scareware shape", scareware.shape))

    benign = datasets.prepare_dataset(benign, shuffle=True)
    scareware = datasets.prepare_dataset(scareware, shuffle=True)

    n_elements = min(benign.shape[0], scareware.shape[0], 150000)

    benign = benign.head(n_elements)
    scareware = scareware.head(n_elements)

    logger.info("{} {}".format("benign shape after balancing", benign.shape))
    logger.info("{} {}".format("scareware shape after balancing", scareware.shape))

    scareware["Label"] = DATASET_NAME.upper()

    loaded_dataset = pd.concat([benign, scareware], ignore_index=True)  # union dataset
    logger.info(loaded_dataset.head())
    loaded_dataset.info()

    benign = None
    scareware = None

    logger.info("{} {}".format("Dataset shape BEFORE preparation", loaded_dataset.shape))
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       drop_columns=["Flow Bytes/s", "Flow Packets/s", "Flow ID", "Source IP",
                                                     "Destination IP", "Timestamp", "Fwd Header Length.1"],
                                       shuffle=True, dropna=True)

    loaded_dataset = None

    logger.info("{} {}".format("Dataset shape AFTER preparation", dataset.shape))

    xTest, yTest = datasets.separate_labels(dataset, encode=True)

    dataset = None

    xTest = datasets.drop_variance(xTest)

    roc_auc_scores = []
    roc_fpr_tpr_thres = []

    # Estimators number test
    logger.info("Estimators number test")

    for i in range(4, 30, 4):
        n_estimators = i**2
        logger.info("Training random forest with {} estimators ({})".format(n_estimators, i))
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(clf, xTest, yTest)
        save_result(roc, auc_score, "estimators", n_estimators, roc_fpr_tpr_thres, roc_auc_scores)


    # Max depth number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    logger.info("max depth number test")
    for i in range(1, 11):
        max_depth = 2**i
        logger.info("Training random forest with {} max depth ({})".format(max_depth, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=max_depth, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(rnd_forest, xTest, yTest)
        save_result(roc, auc_score, "max_depth", max_depth, roc_fpr_tpr_thres, roc_auc_scores)


    # Min Sample Leaf number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    logger.info("Min Sample Leaf number test")
    for i in range(1, 11):
        min_sample_leaf = i
        logger.info("Training random forest with {} min sample leaf ({})".format(min_sample_leaf, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=32, min_samples_leaf=min_sample_leaf, n_jobs=-1,
                                            random_state=42)  # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(rnd_forest, xTest, yTest)
        save_result(roc, auc_score, "min_sample_leaf", min_sample_leaf, roc_fpr_tpr_thres, roc_auc_scores)

    roc_auc_scores, roc_fpr_tpr_thres = [], []
    xTest = None
    yTest = None
    file_handler.close()
    console_handler.close()


def show():
    for parameter_dir in os.listdir(RESULTS_FOLDER_PATH):
        result_dir = os.path.join(RESULTS_FOLDER_PATH, parameter_dir)
        if not os.path.isdir(result_dir):
            continue
        parameter_name = parameter_dir.replace("_", " ").capitalize()

        for file in os.listdir(result_dir):
            if file.endswith(".npy"):
                if "roc_fpr_tpr_thres" in file:
                    fpr_tpr_thres = datasets.np_load_data(result_dir, file)
                    plot.initialize_roc_plt(parameter_name)
                    for i in range(0, len(fpr_tpr_thres)):
                        plot.plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])
                elif "roc_auc_scores" in file:
                    auc_score = datasets.np_load_data(result_dir, file)
                    plot.plot_auc_score(auc_score[:, 0], auc_score[:, 1], parameter_name)

    plot.show()


if __name__ == "__main__":
    calc()
    # show()

