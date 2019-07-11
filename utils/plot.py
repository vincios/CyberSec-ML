import matplotlib.pyplot as plt
import numpy as np


def plt_add_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.legend(loc="lower right")


def initialize_roc_plt(title=None):
    fig = plt.figure(figsize=(20, 10))
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


def plt_cross_validate_results(results: dict, scoring='f1', title=None):
    if scoring == 'roc':
        initialize_roc_plt(title)
        for result in results:
            fpr, tpr, thres = result['roc_curve']
            plt_add_roc_curve(fpr, tpr, result['label'])
    else:
        test_key, train_key = 'test_' + scoring, 'train_' + scoring
        test_scores, train_scores, labels = [], [], []
        for result in results:
            test_scores.append(np.mean(result[test_key]))
            train_scores.append(np.mean(result[train_key]))
            labels.append(result['label'])

        ind = np.arange(len(test_scores))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(20, 10))
        rects1 = ax.bar(ind - width / 2, test_scores, width,
                        label='Test')
        rects2 = ax.bar(ind + width / 2, train_scores, width,
                        label='Train')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(scoring + ' scores by model')
        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}

            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format("%.2f" % height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                            textcoords="offset points",  # in both directions
                            ha=ha[xpos], va='bottom')

        autolabel(rects1, "left")
        autolabel(rects2, "right")

        fig.tight_layout()

    plt.show()