import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from classifiers import CLASSIFIERS, PARAM_GRIDS

import model_perf
from model_perf import bootstrap_proportion

from frames_resampler import build_dataset, DEFAULT_STREAMS_DIR

CLF_NAMES = ["RandomForest", "SVM", "LogisticRegression"]
TASKS = ["chip", "mfr"]


STREAMS_DIR = DEFAULT_STREAMS_DIR
RESULTS_DIR = "process_res"
FIGS_DIR = "figs"

## Colors and textures for color blind friendly graphs
TASK_STYLE = {
    "mfr": ("Manufacturer", "#4477AA", "o"),
    "chip": ("Wi-Fi Chip", "#EE6677", "s"),
}
PLOT_ORDER = ("mfr", "chip")


Y_VALUES = [1, 5, 10, 50, 100, 250, 500, 750, 1000]   # p, num transmitted frames
Y_DEFAULT = 1000                                        # p used by the step sweep
STEP_VALUES = [1, 2, 4, 8, 16, 32]                    # k: every k-th STF level
N_STEP_BASE = 5                                       # base n increment (5 STF samples)

TX_RESULTS = "results_ablation_tx.pkl"
STEP_RESULTS = "results_ablation_step.pkl"


def evaluate_one(X, y, clf_name):
    ### 80/20 stratified split, GridSearchCV on train, accuracy on test.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
    pipe = CLASSIFIERS[clf_name](random_state=None)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True)
    search = GridSearchCV(pipe, PARAM_GRIDS[clf_name], cv=inner_cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    preds = search.predict(X_te)

    acc, (ci_lo, ci_hi) = bootstrap_proportion(preds == y_te)
    return acc, (float(ci_lo), float(ci_hi)), search.best_params_


def evaluate_all_clf(X, y_chip, y_mfr, cell):
    ## evaluate all classifiers for a specif resampling "cell" (p,s combination)
    for clf in CLF_NAMES:
        cell.setdefault(clf, {})
        for task, y in zip(TASKS, (y_chip, y_mfr)):
            ### check if specific cell has been computed before
            if "acc" in cell[clf].get(task, {}):
                continue

            acc, ci, best = evaluate_one(X, y, clf)
            cell[clf][task] = {"acc": acc, "ci": ci, "best_params": best}
            hw = (ci[1] - ci[0]) / 2
            print(f"{clf:<20} {task:<4} -- acc={acc:.3f} ± {hw:.3f}")


def load_results(results_path):
    if Path(results_path).exists():
        with open(results_path, "rb") as f:
            return pickle.load(f).get("results", {})
    return {} ## no temp res


def save_results(results_path, meta_key, values, results):
    with open(results_path, "wb") as f:
        pickle.dump({meta_key: list(values), "clf_names": CLF_NAMES, "results": results}, f)


def check_sweep_val_done(results, sweep_value):
    if sweep_value not in results:
        return False
    return all(
        clf in results[sweep_value]
        and all(t in results[sweep_value][clf] and "acc" in results[sweep_value][clf][t] for t in TASKS)
        for clf in CLF_NAMES
    )



def sweep_tx_num(results_path):
    #### sweep for num of tx frames (p)


    ### check for partial completion
    results = load_results(results_path)

    for Y in Y_VALUES:
        if check_sweep_val_done(results, Y):
            print(f"Y={Y}: already computed, skipping")
            continue

        t0 = time.time()

        print(f"Building dataset...")
        X, y_chip, y_mfr, chip_lm, mfr_lm, _ = build_dataset(STREAMS_DIR, Y=Y, seed=None)
        print("#" * 70)
        print(f"  Num of Transmitted Frames Ablation p={Y}")
        print("#" * 70)
        print(f"Dataset: X={X.shape}, {len(chip_lm)} chips, "f"{len(mfr_lm)} manufacturers")
        cell = results.setdefault(Y, {})
        evaluate_all_clf(X, y_chip, y_mfr, cell)
        save_results(results_path, "Y_values", Y_VALUES, results)
        print(f"Y={Y} done in {time.time() - t0:.1f}s")

    return results


def sweep_tx_step(results_path):
    ### sweep for STF alteration step (s)
 
    ### check for partial completion
    results = load_results(results_path)

    todo = [k for k in STEP_VALUES if not check_sweep_val_done(results, k)]
    for k in STEP_VALUES:
        if k not in todo:
            print(f"k={k}: already computed, skipping")
    if not todo:
        return results

    print(f"Building base dataset")
    X_full, y_chip, y_mfr, chip_lm, mfr_lm, _ = build_dataset(STREAMS_DIR, Y=Y_DEFAULT, seed=None)
    print(f"Base dataset: X={X_full.shape}, {len(chip_lm)} chips, "f"{len(mfr_lm)} manufacturers")

    for k in todo:
        X = X_full[:, ::k]
        print("#" * 70)
        print(f"Step Size for n Ablation s={k*5}")
        print("#" * 70)
        t0 = time.time()

        cell = results.setdefault(k, {})
        evaluate_all_clf(X, y_chip, y_mfr, cell)
        save_results(results_path, "step_values", STEP_VALUES, results)

        print(f"k={k} done in {time.time() - t0:.1f}s")

    return results


######### Plotting Functions ##################
def ci_cell_calc(cells, acc_key="acc", ci_key="ci"):
    accs = [c[acc_key] for c in cells]
    lo, hi = [], []
    for a, c in zip(accs, cells):
        ci = c.get(ci_key)
        if ci is None:
            lo.append(0.0)
            hi.append(0.0)
        else:
            lo.append(a - ci[0])
            hi.append(ci[1] - a)
    return accs, np.array([lo, hi])


def plot_one_classifier(results, clf_name, values, xlabel, xticklabels, out_path):
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    legend_handles = []
    for task in PLOT_ORDER:
        label, color, marker = TASK_STYLE[task]
        cells = [results[v][clf_name][task] for v in values]
        acc, err = ci_cell_calc(cells)
        ax.errorbar(x, acc, yerr=err, fmt=f"-{marker}", color=color,
                    linewidth=2, markersize=5, capsize=4, elinewidth=1.2,
                    label=label)
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, markersize=5,
                   linewidth=2, label=label)
        )

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_axisbelow(True)

    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(top=0.86)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ablation(results, values, xlabel, xticklabels, fig_prefix, figs_dir):
    for clf in CLF_NAMES:
        out = Path(figs_dir) / f"{fig_prefix}_{clf.lower()}.pdf"
        plot_one_classifier(results, clf, values, xlabel, xticklabels, out)




parser = argparse.ArgumentParser()
parser.add_argument("--tx_num",  action="store_true",  help="Run the number-of-transmissions (p) ablation")
parser.add_argument("--tx_step", action="store_true", help="Run the STF step-size (s) ablation")
args = parser.parse_args()

# Neither flag run both.
if not (args.tx_num or args.tx_step):
    args.tx_num = args.tx_step = True

model_perf.rng = np.random.default_rng()

results_dir = Path.home() / Path(RESULTS_DIR)
figs_dir = Path(FIGS_DIR)
results_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)

if args.tx_num:
    results_path = results_dir / TX_RESULTS
    print(f"\n{'#' * 70}\n# Saving intermediate results of tx_num to  {results_path} ")
    results = sweep_tx_num(results_path)
    plot_ablation(results, Y_VALUES,
                    r"Number of Transmitted Frames ($p$)",
                    [str(Y) for Y in Y_VALUES],
                    "ablation_tx", figs_dir)

if args.tx_step:
    results_path = results_dir / STEP_RESULTS
    print(f"\n{'#' * 70}\n# Saving intermediate results of tx_step to  {results_path} ")
    results = sweep_tx_step(results_path)
    plot_ablation(results, STEP_VALUES,
                    r"Step Size for $n$ ($s$)",
                    [str(k * N_STEP_BASE) for k in STEP_VALUES],
                    "ablation_step", figs_dir)