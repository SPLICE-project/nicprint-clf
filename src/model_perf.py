import pickle
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os 

# Bootstrap vars
BOOTSTRAP_B = 1000
CI_ALPHA = 0.05

CLF_LABELS = {
    "RandomForest": "RF",
    "SVM": "SVM",
    "LogisticRegression": "LR",
}

## fixed seed for repet of bootstrap selection
rng = np.random.default_rng(42)
    
def load_model(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["label_map"]

def ci_hw(arr, axis=0):
    lo = np.percentile(arr, 100 * CI_ALPHA / 2, axis=axis)
    hi = np.percentile(arr, 100 * (1 - CI_ALPHA / 2), axis=axis)
    return (hi - lo) / 2

def bootstrap_proportion(hits):
    ### bostrapping resample for 0/1 cases in top2 matrix
    hits = np.asarray(hits, dtype=np.int8)
    n = len(hits)
    boot_idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
    acc_mean = hits[boot_idx].mean(axis=1)
    lo = np.percentile(acc_mean, 100 * CI_ALPHA / 2, axis=0)
    hi = np.percentile(acc_mean, 100 * (1 - CI_ALPHA / 2), axis=0)
    return float(hits.mean()), (lo, hi)

def report(test_y, preds, kind, label_map):
    present = sorted(set(test_y)) ## present labels (i.e., labels actually in this test)
    lab_present = len(present)
    n_test = len(test_y)

    #### P/R/F1 ########
    p_pt, r_pt, f_pt, _ = precision_recall_fscore_support(test_y, preds, labels=present, zero_division=0)
    pa, ra, fa, _ = precision_recall_fscore_support(test_y, preds, labels=present, average="macro", zero_division=0)
    acc_pt = accuracy_score(test_y, preds)

    ### bootstrap matrix --- samples drawn with replacement (i.e., test samples)
    boot_idx = rng.integers(0, n_test, size=(BOOTSTRAP_B, n_test))

    ### bootstrap matrix -- classifier results (per class)
    p_cls_b = np.zeros((BOOTSTRAP_B, lab_present))
    r_cls_b = np.zeros((BOOTSTRAP_B, lab_present))
    f_cls_b = np.zeros((BOOTSTRAP_B, lab_present))

    ### bootstrap average resulsts
    p_avg_b = np.zeros(BOOTSTRAP_B) 
    r_avg_b = np.zeros(BOOTSTRAP_B)
    f_avg_b = np.zeros(BOOTSTRAP_B) 
    acc_b   = np.zeros(BOOTSTRAP_B)

    for b in range(BOOTSTRAP_B):
        yb = test_y[boot_idx[b]]
        pb = preds[boot_idx[b]]
        pc, rc, fc, _ = precision_recall_fscore_support(yb, pb, labels=present, zero_division=0)
        p_cls_b[b] = pc
        r_cls_b[b] = rc
        f_cls_b[b] = fc

        ## averages 
        p_avg_b[b], r_avg_b[b], f_avg_b[b], _ = precision_recall_fscore_support(yb, pb, labels=present, average="macro", zero_division=0)
        acc_b[b] = (yb == pb).mean()

    

    p_cls_hw = ci_hw(p_cls_b)
    r_cls_hw = ci_hw(r_cls_b)
    f_cls_hw = ci_hw(f_cls_b)

    p_avg_hw = ci_hw(p_avg_b)
    r_avg_hw = ci_hw(r_avg_b)
    f_avg_hw = ci_hw(f_avg_b)
    acc_lo = float(np.percentile(acc_b, 100 * CI_ALPHA / 2))
    acc_hi = float(np.percentile(acc_b, 100 * (1 - CI_ALPHA / 2)))
    acc_hw = (acc_hi - acc_lo) / 2

    #### Printing ##########   
    print(f"\n Average {kind}:"
            f"accuracy={acc_pt:.3f} ± {acc_hw:.3f}  "
            f"precision={pa:.3f} ± {p_avg_hw:.3f}  "
            f"recall={ra:.3f} ± {r_avg_hw:.3f}  "
            f"f1={fa:.3f} ± {f_avg_hw:.3f}   (95% CI, B={BOOTSTRAP_B})")

    ## column width sizes
    NAME_W = 14
    NUM_W  = 15
    print("\n")
    print(f"    {'':<{NAME_W}}{'precision':>{NUM_W}}{'recall':>{NUM_W}}{'f1-score':>{NUM_W}}")
    print()
    for k, lbl in enumerate(present):
        name = label_map[lbl]
        print(f"    {name:<{NAME_W}}"
                f"{f'{p_pt[k]:.3f} ± {p_cls_hw[k]:.3f}':>{NUM_W}}"
                f"{f'{r_pt[k]:.3f} ± {r_cls_hw[k]:.3f}':>{NUM_W}}"
                f"{f'{f_pt[k]:.3f} ± {f_cls_hw[k]:.3f}':>{NUM_W}}")
    print("\n")

    return acc_pt, (acc_lo, acc_hi)

def print_misses(test_y, preds, kind, label_map, group_names, run_indices):
    missclf = np.where(preds != test_y)[0]
    if len(missclf):
        print(f"\n  {kind} misclassified samples:")
        for i in missclf:
            true_name = label_map[test_y[i]]
            pred_name = label_map[preds[i]]
            if group_names is not None:
                print(f"    [{group_names[i]} run {run_indices[i]}]: true={true_name}, predicted={pred_name}")
            else:
                print(f"    run {run_indices[i]}: true={true_name}, predicted={pred_name}")
    else:
        print("No misclassified samples!!!")

def score(model, label_map, X, y, true_label_map, run_indices, kind, group_names=None, print_misclf=False):

    ## convert names (chipset name or mfr) to class index 
    model_label_to_idx = {name: idx for idx, name in label_map.items()}

    missing = sorted({true_label_map[yi] for yi in y if true_label_map[yi] not in model_label_to_idx})
    if missing:
        print(f"Label {missing} not in model map")
        sys.exit(-2)

    print(f"Len of Test Set = {len(X)}")
    
    test_y = np.array([model_label_to_idx[true_label_map[yi]] for yi in y])
    preds = model.predict(X)

    if print_misclf:
        print_misses(test_y, preds, kind, label_map, group_names, run_indices)
       
    return report(test_y, preds, kind, label_map)


##### Plotting functions
COLORS = {"RandomForest": "#4477AA", "SVM": "#EE6677", "LogisticRegression": "#228833"}
HATCHES = {"RandomForest": "//", "SVM": "", "LogisticRegression": ".."}
CLFS = list(COLORS)
SPACING = 1.2   # gap between position groups
GROUP_W = 0.9   # total width of the bars in one group
 
def draw_bars(ax, accs, cis, groups, rotate):
    x = np.arange(len(groups)) * SPACING
    bar_w = GROUP_W / len(CLFS)
 
    for i, clf in enumerate(CLFS):
        vals = [accs[g].get(clf) or 0.0 for g in groups]
        lo, hi = [], []
        for v, g in zip(vals, groups):
            ci = cis[g].get(clf) or (v, v)
            lo.append(v - ci[0])
            hi.append(ci[1] - v)
        ax.bar(x - GROUP_W / 2 + bar_w * (i + 0.5), vals,
               width=bar_w, color=COLORS[clf], hatch=HATCHES[clf],
               edgecolor="black", linewidth=0.8,
               yerr=np.array([lo, hi]), capsize=3, ecolor="black",
               error_kw={"linewidth": 1.0})
 
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11, fontweight="bold",
                       rotation=25 if rotate else 0,
                       ha="right" if rotate else "center",
                       rotation_mode="anchor")
    ax.set_xlim(x[0] - SPACING / 2, x[-1] + SPACING / 2)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelleft=True)
 
def save_fig(fig, out_name):
    fig.legend(
        handles=[Patch(facecolor=COLORS[c], hatch=HATCHES[c], edgecolor="black",
                       label=CLF_LABELS.get(c, c)) for c in CLFS],
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=len(CLFS), frameon=False, fontsize=12,
    )
    out_path = os.path.join("figs", out_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy bar plot to {out_path}")

def plot_accuracy_summary_pos(panels, out_name, ylabel="Accuracy"):
    panels = [p if len(p) == 3 else (*p, None) for p in panels]
    positions = list(panels[0][0])
 
    fig, axes = plt.subplots(
        1, len(panels), sharey=True, squeeze=False,
        figsize=(max(6.0, 1.2 * len(positions)) * len(panels), 4.2),
    )
    for ax, (accs, cis, caption) in zip(axes[0], panels):
        draw_bars(ax, accs, cis, positions, rotate=True)
        if caption:
            ax.text(0.5, -0.38, caption, transform=ax.transAxes,
                    ha="center", va="top", fontsize=13, fontweight="bold")
 
    axes[0][0].set_ylabel(ylabel, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.85, bottom=0.28)
    save_fig(fig, out_name)


def plot_accuracy_summary(groups, out_name, ylabel="Accuracy"):
    names = list(groups)
    accs = {g: groups[g][0] for g in names}
    cis = {g: groups[g][1] for g in names}
 
    fig, ax = plt.subplots(figsize=(8, 3.5))
    draw_bars(ax, accs, cis, names, rotate=False)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.88)
    save_fig(fig, out_name)


def plot_confusion_matrix(cm, label_map, clf_name, output_path):
    labels = [label_map[i] for i in sorted(label_map)]
    k = len(labels)

    side = max(6, 0.5 * k)
    fig, ax = plt.subplots(figsize=(side, side))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d",
              text_kw={"fontsize": 7})
    ax.set_title(CLF_LABELS.get(clf_name, clf_name), fontsize=16)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")