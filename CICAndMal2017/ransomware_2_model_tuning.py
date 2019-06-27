import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import datasets, random_forest, plot

DATASET_NAME = "ransomware"
DATASET_NAME_2 = "benign"
RESULTS_FOLDER_PATH = os.path.join("results", DATASET_NAME, "2_model_tuning")


def calc():
    if not os.path.exists(RESULTS_FOLDER_PATH):
        os.makedirs(RESULTS_FOLDER_PATH)

    benign = datasets.load_all(os.path.join("datasets", DATASET_NAME_2))  # load dataset from csv
    ransomware = datasets.load_all(os.path.join("datasets", DATASET_NAME))  # load dataset from csv
    ransomware["Label"] = DATASET_NAME.upper()

    print("benign shape", benign.shape)
    print("ransomware shape", ransomware.shape)

    loaded_dataset = pd.concat([benign, ransomware], ignore_index=True)  # union dataset
    print(loaded_dataset.head())
    loaded_dataset.info()

    benign = None
    ransomware = None

    print("Dataset shape BEFORE preparation", loaded_dataset.shape)
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       drop_columns=["Flow Bytes/s", "Flow Packets/s", "Flow ID", "Source IP",
                                                     "Destination IP", "Timestamp", "Fwd Header Length.1"],
                                       shuffle=True, dropna=True)


    loaded_dataset = None

    print("Dataset shape AFTER preparation", dataset.shape)

    xTest, yTest = datasets.separate_labels(dataset, encode=True)

    dataset = None

    xTest = datasets.drop_variance(xTest)

    roc_auc_scores = []
    roc_fpr_tpr_thres = []

    # Estimators number test
    print("Estimators number test")
    for i in range(4, 30, 4):
        n_estimators = i**2
        print("Training random forest with {} estimators ({})".format(n_estimators, i))
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(clf, xTest, yTest)
        roc.insert(0, n_estimators)
        auc_score.insert(0, n_estimators)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

        # plt_add_roc_curve(fpr, tpr, label=str(n_estimators))

    np_roc_auc_scores = np.array(roc_auc_scores)
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    datasets.np_double_save(np_roc_fpr_tpr_thres, RESULTS_FOLDER_PATH,
                            "rnd_forest_estimators_roc_fpr_tpr_thres", as_csv=True, as_npy=True)
    datasets.np_double_save(np_roc_auc_scores, RESULTS_FOLDER_PATH,
                            "rnd_forest_estimators_roc_auc_scores", as_csv=True, as_npy=True)

    # Max depth number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    print("max depth number test")
    for i in range(1, 11):
        max_depth = 2**i
        print("Training random forest with {} max depth ({})".format(max_depth, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=max_depth, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(rnd_forest, xTest, yTest)
        roc.insert(0, max_depth)
        auc_score.insert(0, max_depth)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

    np_roc_auc_scores = np.array(roc_auc_scores)
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    datasets.np_double_save(np_roc_fpr_tpr_thres, RESULTS_FOLDER_PATH,
                            "rnd_forest_depth_roc_fpr_tpr_thres", as_csv=True, as_npy=True)
    datasets.np_double_save(np_roc_auc_scores, RESULTS_FOLDER_PATH,
                            "rnd_forest_depth_roc_auc_scores", as_csv=True, as_npy=True)

    # Min Sample Leaf number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    print("Min Sample Leaf number test")
    for i in range(1, 11):
        min_sample_leaf = i
        print("Training random forest with {} min sample leaf ({})".format(min_sample_leaf, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=32, min_samples_leaf=min_sample_leaf, n_jobs=-1,
                                            random_state=42)  # Random Forest Classifier
        roc, auc_score = random_forest.fit_and_roc(rnd_forest, xTest, yTest)
        roc.insert(0, min_sample_leaf)
        auc_score.insert(0, min_sample_leaf)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

    np_roc_auc_scores = np.array(roc_auc_scores)
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    datasets.np_double_save(np_roc_fpr_tpr_thres, RESULTS_FOLDER_PATH,
                            "rnd_forest_samples_roc_fpr_tpr_thres", as_csv=True, as_npy=True)
    datasets.np_double_save(np_roc_auc_scores, RESULTS_FOLDER_PATH,
                            "rnd_forest_samples_roc_auc_scores", as_csv=True, as_npy=True)

    xTest = None
    yTest = None


def show():
    fpr_tpr_thres = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_estimators_roc_fpr_tpr_thres.npy")
    plot.initialize_roc_plt("Estimators")
    for i in range(0, len(fpr_tpr_thres)):
        plot.plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])
    auc_score = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_estimators_roc_auc_scores.npy")
    plot.plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Estimators")

    fpr_tpr_thres = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_depth_roc_fpr_tpr_thres.npy")
    plot.initialize_roc_plt("Max Depth")
    for i in range(0, len(fpr_tpr_thres)):
        plot.plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])
    auc_score = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_depth_roc_auc_scores.npy")
    plot.plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Max Depth")

    fpr_tpr_thres = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_samples_roc_fpr_tpr_thres.npy")
    plot.initialize_roc_plt("Min Samples Leaf")
    for i in range(0, len(fpr_tpr_thres)):
        plot.plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])

    auc_score = datasets.np_load_data(RESULTS_FOLDER_PATH, "rnd_forest_samples_roc_auc_scores.npy")
    plot.plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Min Samples Leaf")

    plot.show()


if __name__ == "__main__":
    calc()
