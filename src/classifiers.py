import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import  confusion_matrix
from model_perf import report, print_misses
from pathlib import Path

def make_rf(n_estimators=1000, random_state=42):
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])


def make_svm(C=1.0, kernel="rbf", random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel=kernel, random_state=random_state, probability=True))
    ])


def make_lr(C=1.0, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, random_state=random_state, max_iter=10000))
    ])


CLASSIFIERS = {
    "RandomForest": make_rf,
    "SVM": make_svm,
    "LogisticRegression": make_lr,
}

# Parameter grids for GridSearchCV
PARAM_GRIDS = {
    "RandomForest": {
        "clf__n_estimators": [100, 500, 1000],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["lbfgs", "saga"],
    },
}


def evaluate(X, y, label_map, kind, run_indices, print_miss = False, clf_name="RandomForest", n_splits=5, random_state=None, **clf_kwargs):
    '''
    This functions runs startified k-fold with n_split to evaluate model performance.
    Each fold has inner splits of 3 to find params. The splits are done only on the "test" data
    for that split. 
    Model's weights are not save to disk. Use train_and_save for that instead
    '''
    make_clf = CLASSIFIERS[clf_name]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_true = []
    all_y_pred = []
    all_indices = []
    fold_accuracies = []
    best_params_per_fold = []

    for train_idx, test_idx in skf.split(X, y):
        clf = make_clf(random_state=random_state, **clf_kwargs)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        clf = GridSearchCV(clf, PARAM_GRIDS[clf_name], cv=inner_cv, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        best_params_per_fold.append(clf.best_params_)
        preds = clf.predict(X[test_idx])

        fold_acc = np.mean(preds == y[test_idx])
        fold_accuracies.append(fold_acc)

        all_y_true.extend(y[test_idx])
        all_y_pred.extend(preds)
        all_indices.extend(test_idx)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_indices = np.array(all_indices)

    if(print_miss):
         print_misses(all_y_true, all_y_pred, kind, label_map, None, run_indices)
    acc_pt, (ci_lo, ci_hi) = report(all_y_true, all_y_pred, kind, label_map)
    cm = confusion_matrix(all_y_true, all_y_pred)

    return {
        "clf_name": clf_name,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": np.mean(fold_accuracies),
        "aggregated_accuracy": acc_pt,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "confusion_matrix": cm,
        "best_params_per_fold": best_params_per_fold,
    }


def train_and_save(X, y, label_map, clf_name="RandomForest", save_path="model.pkl", random_state=None, inner_splits=5, **clf_kwargs):
    make_clf = CLASSIFIERS[clf_name]
    clf = make_clf(random_state=random_state, **clf_kwargs)

    print(f"Len of Train Set = {len(X)}")

   
    t0 = time.time()
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(clf, PARAM_GRIDS[clf_name], cv=cv, n_jobs=-1)
    search.fit(X, y)
    clf = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV score: {search.best_score_:.3f}")
    elapsed = time.time() - t0
    print(f"Elapse time = {elapsed}s")
  
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump({"model": clf, "label_map": label_map}, f)

    print(f"Saved {clf_name} model to {save_path}")
    return clf,label_map