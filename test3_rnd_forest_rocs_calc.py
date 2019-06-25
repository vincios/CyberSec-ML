import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import VarianceThreshold

BENING_PATH = os.path.join("datasets", "benign")
RANSOMWARE_PATH = os.path.join("datasets", "ransomware")
ADWARE_PATH = os.path.join("datasets", "adware")
SCAREWARE_PATH = os.path.join("datasets", "scareware")
SMSMALWARE_PATH = os.path.join("datasets", "smsmalware")


def load_data(filename, housing_path):  # define functions load
    csv_path = os.path.join(housing_path, filename)
    return pd.read_csv(csv_path, skipinitialspace=True)


def merge_datasets(folder_name):
    final_dataset = None  # final dataset
    FOLDER_PATH = os.path.join("datasets", folder_name)
    files = os.listdir(FOLDER_PATH)
    for file in files:
        data = load_data(file, FOLDER_PATH)
        if final_dataset is None:
            final_dataset = data
        else:
            final_dataset = pd.concat([final_dataset, data])
    return final_dataset


def np_double_save(filename, ndarray):
    np.savetxt("results/" + filename + ".csv", ndarray, delimiter=",", fmt="%s")
    np.save("results/" + filename, ndarray)



if __name__ == '__main__':
    benign = merge_datasets("benign")  # load dataset from csv
    ransomware = merge_datasets("ransomware")  # load dataset from csv
    ransomware['Label'] = 'RANSOMWARE'
    print("benign shape", benign.shape)
    print("ransomware shape", ransomware.shape)
    dataset = pd.concat([benign, ransomware])  # union dataset
    print(dataset.head())
    dataset.info()

    np.random.permutation(dataset)

    dataset.dropna(inplace=True)    #drop instances with null values

    dataset_clean = dataset.drop("Flow Bytes/s", axis=1)  # data cleaning, drop label: Flow Bytes/s null values
    dataset_clean = dataset_clean.drop("Flow Packets/s", axis=1)  # data cleaning, drop label: Flow Packets/s null values
    dataset_clean = dataset_clean.drop("Flow ID", axis=1)  # data cleaning, drop label: Flow Bytes/s null values
    dataset_clean = dataset_clean.drop("Source IP", axis=1)  # data cleaning, drop label: Flow Packets/s null values
    dataset_clean = dataset_clean.drop("Destination IP", axis=1)  # data cleaning, drop label: Flow Bytes/s null values
    dataset_clean = dataset_clean.drop("Timestamp", axis=1)  # data cleaning, drop label: Flow Packets/s null values
    dataset_clean = dataset_clean.drop("Packet Length Std", axis=1)  # data cleaning, drop label: Flow Packets/s null values
    dataset_clean = dataset_clean.drop("CWE Flag Count", axis=1)  # data cleaning, drop label: Flow Packets/s null values
    dataset_clean = dataset_clean.drop("Down/Up Ratio", axis=1)  # data cleaning, drop label: Flow Packets/s null values

    housing_labels = dataset_clean["Label"]  # separating labels
    xTest = dataset_clean.drop("Label", axis=1)
    le = preprocessing.LabelEncoder()
    le.fit(housing_labels)
    yTest = le.fit_transform(housing_labels)  # transform text in number
    print(yTest)
    print(le.classes_)

    variance_thres = VarianceThreshold(threshold=0)  # feature selection with variance 0
    xTest = variance_thres.fit_transform(xTest)
    print(xTest.shape)

    '''
    # initialize_roc_plt("Estimators")
    roc_auc_scores = []
    roc_fpr_tpr_thres = []

    # Estimators number test
    print("Estimators number test")
    for i in range(4, 30, 4):
        n_estimators = i**2
        print("Training random forest with {} estimators ({})".format(n_estimators, i))
        rnd_forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = fit_and_roc(rnd_forest, xTest, yTest)
        roc.insert(0, n_estimators)
        auc_score.insert(0, n_estimators)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

        # plt_add_roc_curve(fpr, tpr, label=str(n_estimators))

    np_roc_auc_scores = np.array(roc_auc_scores).astype("float64")
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    np_double_save("rnd_forest_estimators_roc_fpr_tpr_thres", np_roc_fpr_tpr_thres)
    np_double_save("rnd_forest_estimators_roc_auc_scores", np_roc_auc_scores)

    # Max depth number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    print("max depth number test")
    for i in range(1, 11):
        max_depth = 2**i
        print("Training random forest with {} max depth ({})".format(max_depth, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=max_depth, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = fit_and_roc(rnd_forest, xTest, yTest)
        roc.insert(0, max_depth)
        auc_score.insert(0, max_depth)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

    np_roc_auc_scores = np.array(roc_auc_scores).astype("float64")
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    np_double_save("rnd_forest_depth_roc_fpr_tpr_thres", np_roc_fpr_tpr_thres)
    np_double_save("rnd_forest_depth_roc_auc_scores", np_roc_auc_scores)
    '''
    # Min Sample Leaf number test
    roc_auc_scores = []
    roc_fpr_tpr_thres = []
    print("Min Sample Leaf number test")
    for i in range(1, 11):
        min_sample_leaf = i
        print("Training random forest with {} min sample leaf ({})".format(min_sample_leaf, i))
        rnd_forest = RandomForestClassifier(n_estimators=144, max_depth=32, min_samples_leaf=min_sample_leaf, n_jobs=-1, random_state=42)     # Random Forest Classifier
        roc, auc_score = fit_and_roc(rnd_forest, xTest, yTest)
        roc.insert(0, min_sample_leaf)
        auc_score.insert(0, min_sample_leaf)
        roc_fpr_tpr_thres.append(roc)
        roc_auc_scores.append(auc_score)

    np_roc_auc_scores = np.array(roc_auc_scores).astype("float64")
    np_roc_fpr_tpr_thres = np.array(roc_fpr_tpr_thres)

    np_double_save("rnd_forest_samples_roc_fpr_tpr_thres", np_roc_fpr_tpr_thres)
    np_double_save("rnd_forest_samples_roc_auc_scores", np_roc_auc_scores)

    # fig = plt.figure()
    # fig.suptitle("Roc auc scores")
    # plt.plot(roc_auc_scores[:, 0], roc_auc_scores[:, 1])
    # plt.legend()
    # plt.show()