from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import cross_validate


def cross_validate_scoring(clf, xTest, yTest, cv=3, scoring=None, return_train_score=False, return_estimator=False):
    '''
    Perform cross validation for multiple scores and return as result the mean across all the cross validation scores
    :param clf:
    :param xTest:
    :param yTest:
    :param cv:
    :param scoring: ['roc', 'roc_auc', 'f1', 'precision', 'recall', 'confusion_matrix']
    :param return_train_score:
    :param return_estimator:
    :return:
    '''
    perform_roc = 'roc' in scoring
    perform_confusion_matrix = 'confusion_matrix' in scoring

    if perform_roc:
        scoring.remove('roc')

    if perform_confusion_matrix:
        scoring.remove('confusion_matrix')

    should_return_estimator = return_estimator or perform_roc or perform_confusion_matrix

    results = cross_validate(clf, xTest, yTest,
                             cv=cv, scoring=scoring,
                             return_train_score=return_train_score, return_estimator=should_return_estimator,
                             verbose=2, n_jobs=-1)

    if should_return_estimator:
        fitted_clf = results.pop('estimator', None)[0]
        if perform_confusion_matrix:
            predictions = fitted_clf.predict(xTest)
            cm = confusion_matrix(yTest, predictions)
            results['confusion_matrix'] = cm
        if perform_roc:
            if hasattr(fitted_clf, "predict_proba"):
                score = fitted_clf.predict_proba(xTest)
                fpr, tpr, thres = roc_curve(yTest, score[:, 1])
            else:
                score = fitted_clf.decision_function(xTest)
                fpr, tpr, thres = roc_curve(yTest, score)
            results['roc_curve'] = [fpr, tpr, thres]
        if return_estimator:
            results['estimator'] = fitted_clf

    return results
