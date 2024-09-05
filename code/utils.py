import os
import importlib
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, make_scorer


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)

 
def get_metric(metric_name, metric_pkg='sklearn.metrics'):
    metric_dict = {
        'mcc': 'matthews_corrcoef',
        'f1': 'f1_score',
        'acc': 'accuracy_score',
    }

    metric_name = metric_dict.get(metric_name, None)
    if metric_name is None:
        raise NotImplementedError(f'The {metric_name} metric is not implemeneted')
    
    return getattr(importlib.import_module(metric_pkg), metric_name)


def get_SVM_performance(val_metric, train_X, train_y, test_X, test_y, folds, metric_params={}):
    val_metric_func = get_metric(val_metric)
    if val_metric == 'f1':
        metric_params = {'average': 'macro'}

    scorer = make_scorer(
        val_metric_func, response_method='predict', **metric_params
    )
    clf = SVC(probability=True)
    params_grid = {
        'gamma': ['scale', 0.0001, 0.001, 0.01],
        'C': [0.1, 1.0, 10.0, 100.0]
    }
    grid_search = GridSearchCV(clf, params_grid, cv=folds, scoring=scorer, n_jobs=5)
    grid_search.fit(train_X, train_y)
    preds = grid_search.best_estimator_.predict(test_X)
    probs = grid_search.best_estimator_.predict_proba(test_X)

    results = {
        'test_y': test_y,
        'probs': probs,
        'best_params': grid_search.best_params_,
        f'train_{val_metric}': grid_search.best_score_,
        'test_acc': accuracy_score(test_y, preds),
        'test_mcc': matthews_corrcoef(test_y, preds),
        'test_f1': f1_score(test_y, preds, average='macro')
    }

    return results


def get_pos_neg_masks(y, device):
    pos_mask = torch.eq(y, y.T)
    neg_mask = torch.logical_not(pos_mask)

    diag = torch.logical_not(torch.eye(y.size(0))).to(device)
    pos_mask *= diag

    return pos_mask, neg_mask
