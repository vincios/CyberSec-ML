import matplotlib.pyplot as plt


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


def show():
    plt.show()
