from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_validate


def fit_and_roc(clf, xTest, yTest, cv=3):
    clf.fit(xTest, yTest)
    y_probas_forest = cross_val_predict(clf, xTest, yTest, cv=cv, method="predict_proba", verbose=2, n_jobs=-1)
    y_scores_forest = y_probas_forest[:, 1]  # score = probabilities of positive class
    fpr, tpr, thresholds = roc_curve(yTest, y_scores_forest)
    roc_auc_score_val = roc_auc_score(yTest, y_scores_forest)
    return [fpr, tpr, thresholds], [roc_auc_score_val]


def cross_validate_scoring(clf, xTest, yTest, cv=3, scoring=None, return_train_score=False, return_estimator=False):
    '''
    Perform cross validation for multiple scores and return as result the mean across all the cross validation scores
    :param clf:
    :param xTest:
    :param yTest:
    :param cv:
    :param scoring: ['roc', 'roc_auc', 'f1', 'precision', 'recall']
    :param return_train_score:
    :param return_estimator:
    :return:
    '''
    perform_roc = 'roc' in scoring
    if perform_roc:
        scoring.remove('roc')

    should_return_estimator = return_estimator or perform_roc
    results = {}
    results = cross_validate(clf, xTest, yTest,
                             cv=cv, scoring=scoring,
                             return_train_score=return_train_score, return_estimator=should_return_estimator,
                             verbose=2, n_jobs=-1)

    if perform_roc:
        fitted_clf = results.pop('estimator', None)[0]
        probas = fitted_clf.predict_proba(xTest)
        fpr, tpr, thres = roc_curve(yTest, probas[:, 1])
        results['roc_curve'] = [fpr, tpr, thres]

    return results
