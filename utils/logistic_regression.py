from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict


def fit_and_roc(clf, xTest, yTest, cv=3):
    clf.fit(xTest, yTest)
    y_probas_forest = cross_val_predict(clf, xTest, yTest, cv=cv, method="predict_proba", verbose=2, n_jobs=-1)
    y_scores_forest = y_probas_forest[:, 1]  # score = probabilities of positive class
    fpr, tpr, thresholds = roc_curve(yTest, y_scores_forest)
    roc_auc_score_val = roc_auc_score(yTest, y_scores_forest)
    return [fpr, tpr, thresholds], [roc_auc_score_val]