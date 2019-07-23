import os
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import datasets, scoring, plot

DATASET_NAME = "spam"
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

    # loaded_dataset["Label"] = DATASET_NAME.upper()

    logger.info(loaded_dataset.head())
    loaded_dataset.info()
    logger.info(loaded_dataset['URL_Type_obf_Type'].value_counts())

    dataset = None

    logger.info("{} {}".format("Dataset shape BEFORE preparation", loaded_dataset.shape))
    dataset = datasets.prepare_dataset(loaded_dataset,
                                       # drop_columns=["Flow Bytes/s", "Flow Packets/s", "Fwd Header Length.1"],
                                       shuffle=True, dropna_axis=[0, 1])

    loaded_dataset = None

    logger.info("{} {}".format("Dataset shape AFTER preparation", dataset.shape))

    xTest, yTest = datasets.separate_labels(dataset, encode=True, column_name="URL_Type_obf_Type")

    dataset = None

    xTest = datasets.drop_variance(xTest)
    standardScaler = StandardScaler()
    xTestScaled = standardScaler.fit_transform(xTest)

    results_array = []

    logger.info("Logistic Regression")
    log_reg = LogisticRegression(verbose=1, n_jobs=-1, max_iter=1000)

    console_handler.close()
    file_handler.close()


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
    #show()


