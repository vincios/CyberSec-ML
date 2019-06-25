import matplotlib.pyplot as plt
import numpy as np


def plt_add_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.legend(loc="lower right")

def initialize_roc_plt(title=None):
    fig = plt.figure()
    fig.suptitle(title + " ROC")
    plt.plot([0, 1], [0, 1], "--k")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")



def plot_auc_score(x, y, title=None):
    fig = plt.figure()
    fig.suptitle(title + " AUC Scores")
    plt.xlabel(title)
    plt.ylabel("Score")
    plt.plot(x, y)
    plt.plot(x, y, "o")
    plt.legend(loc="lower right")

def np_load_data(filename):
    return np.load("results/" + filename + ".npy", allow_pickle=True)


if __name__ == '__main__':
    fpr_tpr_thres = np_load_data("rnd_forest_estimators_roc_fpr_tpr_thres")
    initialize_roc_plt("Estimators")
    for i in range(0, len(fpr_tpr_thres)):
        plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])
    auc_score = np_load_data("rnd_forest_estimators_roc_auc_scores")
    plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Estimators")

    fpr_tpr_thres = np_load_data("rnd_forest_depth_roc_fpr_tpr_thres")
    initialize_roc_plt("Max Depth")
    for i in range(0, len(fpr_tpr_thres)):
        plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])
    auc_score = np_load_data("rnd_forest_depth_roc_auc_scores")
    plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Max Depth")

    fpr_tpr_thres = np_load_data("rnd_forest_samples_roc_fpr_tpr_thres")
    initialize_roc_plt("Min Samples Leaf")
    for i in range(0, len(fpr_tpr_thres)):
        plt_add_roc_curve(fpr_tpr_thres[i][1], fpr_tpr_thres[i][2], fpr_tpr_thres[i][0])

    auc_score = np_load_data("rnd_forest_samples_roc_auc_scores")
    plot_auc_score(auc_score[:, 0], auc_score[:, 1], "Min Samples Leaf")

    plt.show()
    plt.legend()
