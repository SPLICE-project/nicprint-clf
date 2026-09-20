import sys
import os
import argparse
import numpy as np
import ana_funcs as af
from data_processing import load_device_caps, build_dataset, build_dataset_by_manufacturer
from model_perf import load_model, report, print_misses, bootstrap_proportion, plot_accuracy_summary_pos



parser = argparse.ArgumentParser()
parser.add_argument("--models_mfr_dir", default="models/base", help="Directory containing saved .pkl models for manufacturers")
parser.add_argument("--models_chip_dir", default="models/vote_spc", help="Directory containing saved .pkl models for manufacturers")
parser.add_argument("--print_missclf", action="store_true", help="Print Missclassified Samples?")
parser.add_argument("--verb", default=0, help="Verbosity 0 - Just Chip Output, 1 - Mfr Output Too, 2 -- All output")
parser.add_argument("--plot", action="store_true", help="Plot Results?")
parser.add_argument("--top2", action="store_true", help="Show top2 accuracy")
args = parser.parse_args()

verbs = int(args.verb)
print_wrong = args.print_missclf
top2_arg = args.top2

# dicts for mapping classifiers
CLF_NAMES = ["RandomForest", "SVM", "LogisticRegression"]

CLF_TAGS = {
    "RandomForest":       "randomforest",
    "SVM":                "svm",
    "LogisticRegression": "logisticregression",
}


## correct env profile
af.use_profile("DIFFPOS")

# Per-manufacturer device classifiers: lazily loaded as needed
device_model_cache = {}  # (clf_name, mfr_tag) -> {model, label_map}

def get_device_model(clf_name, mfr):
    mfr_tag = mfr.lower().replace(" ", "_")
    key = (clf_name, mfr_tag)
    if key not in device_model_cache:
        path = os.path.join(args.models_chip_dir, f"{CLF_TAGS[clf_name]}_{mfr_tag}_device_model.pkl")
        if not os.path.exists(path):
            print(f"No specialized {clf_name} for manufacturer: {mfr_tag}")
            sys.exit(1)
        else:
            model, label_map = load_model(path)
            device_model_cache[key] = {"model": model, "label_map": label_map}
    return device_model_cache[key]


######################### Position 1 -- Behind Wall 3 m ##############################
device_caps_pos1 = [
    ("MT6765",    "Motorola G Power Pos 1",  load_device_caps("Moto G Power Pos 1", "all_moto_pos1/moto_pos1_{i}_acks.pkl", 30)),
    ("AC9560",    "ROG Pos 1",               load_device_caps("ROG Pos 1", "all_rog_pos1/rog_pos1_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 1",   load_device_caps("BrosTrend AC650 Pos 1", "all_brostrend_pos1/brostrend_pos1_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 1",  load_device_caps("MacBook Pro 2012 Pos 1", "all_mac_pos1/mac_pos1_{i}_acks.pkl", 30)),
]

######################### Position 2 -- Inside Metal Box 5 m ##############################
device_caps_pos2 = [
    ("AC9560",    "ROG Pos 2",               load_device_caps("ROG Pos 2", "all_rog_pos2/rog_pos2_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 2",  load_device_caps("MacBook Pro 2012 Pos 2", "all_mac_pos2/mac2_pos_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 2",   load_device_caps("BrosTrend AC650 Pos 2", "all_brostrend_pos2/brostrend_pos2_{i}_acks.pkl", 30)),
    ("MT6765",    "Motorola G Power Pos 2",        load_device_caps("Moto G Power 2", "all_moto_pos2/moto_pos2_{i}_acks.pkl", 30)),
]

######################### Position 3 -- LOS 5m ##############################
device_caps_pos3 = [
    ("AC9560",    "ROG Pos 3",               load_device_caps("ROG Pos 3", "all_rog_pos3/rog_pos3_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 3",  load_device_caps("MacBook Pro 2012 Pos 3", "all_mac_pos3/mac_pos3_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 3",   load_device_caps("BrosTrend AC650 Pos 3", "all_brostrend_pos3/brostrend_pos3_{i}_acks.pkl", 30)),
    ("MT6765",    "Motorola G Power Pos 3",   load_device_caps("Moto G Power 3", "all_moto_pos3/moto_pos3_{i}_acks.pkl", 30)),
]

######################### Position 4 -- Inside Carbord Box 8m  ##############################
device_caps_pos4 = [
    ("AC9560",    "ROG Pos 4",               load_device_caps("ROG Pos 4", "all_rog_pos4/rog_pos4_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 4",  load_device_caps("MacBook Pro 2012 Pos 4", "all_mac_pos4/mac_pos4_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 4",   load_device_caps("BrosTrend AC650 Pos 4", "all_brostrend_pos4/brsotrend_pos4_{i}_acks.pkl", 30)),
    ("MT6765",    "Motorola G Power Pos 4",   load_device_caps("Moto G Power 4", "all_moto_pos4/moto_pos4_{i}_acks.pkl", 30)),
]

######################### Position 5 -- LOS 1m  ##############################
device_caps_pos5 = [
    ("AC9560",    "ROG Pos 5",               load_device_caps("ROG Pos 5", "all_rog_pos5/rog_pos5_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 5",  load_device_caps("MacBook Pro 2012 Pos 5", "all_mac_pos5/mac_pos5_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 5",   load_device_caps("BrosTrend AC650 Pos 5", "all_brostrend_pos5/brostrend_pos5_{i}_acks.pkl", 30)),
    ("MT6765",    "Motorola G Power Pos 5",   load_device_caps("Moto G Power 5", "all_moto_pos5/moto_pos5_{i}_acks.pkl", 30)),
]

positions = {
    "Position 1": device_caps_pos1,
    "Position 2": device_caps_pos2,
    "Position 3": device_caps_pos3,
    "Position 4": device_caps_pos4,
    "Position 5": device_caps_pos5
}

positions = {name: caps for name, caps in positions.items() if caps}
if not positions:
    print("No test data!")
    sys.exit(1)


### Run inferences
chip_acc       = {pos: {} for pos in positions}
chip_ci        = {pos: {} for pos in positions}
chip_top2_acc  = {pos: {} for pos in positions}
chip_top2_ci   = {pos: {} for pos in positions}


## loading manufacturer models
mfr_models = {} ## dict with model +label for mfrk key
for clf_name in CLF_NAMES:
    path = os.path.join(args.models_mfr_dir, f"{CLF_TAGS[clf_name]}_manufacturer_model.pkl")
    model, label_map = load_model(path)
    mfr_models[clf_name] = {"model": model, "label_map": label_map}
    print(f"Loaded manufacturer model {path}")


### this assumes all manufacturer models have the same label map (which should be the case)
model_label_map = mfr_models[CLF_NAMES[0]]["label_map"]
mfr_labels = [model_label_map[c] for c in range(len(model_label_map))]
mfr_name_to_idx = {name: i for i, name in model_label_map.items()}


for pos_name, device_caps in positions.items():
    print("#" * 70)
    print(f"  {pos_name}")
    print("#" * 70)
    X_dev, y_dev, dev_label_map, run_indices_dev, group_pos_label = build_dataset(device_caps, mcs_index=0)
    _, y_mfr, mfr_label_map, run_indices_mfr, group_pos_label = build_dataset_by_manufacturer(device_caps, mcs_index=0)
    true_mfr = np.array([mfr_name_to_idx[mfr_label_map[yi]] for yi in y_mfr]) ## convert data idx to lab index
    true_chip_names = [dev_label_map[yi] for yi in y_dev] ## convert data idx to labels

   #### Manufacturer soft voting (stage 1)
    num_caps = len(X_dev)
    sum_probs = np.zeros((num_caps, len(mfr_labels)))
    for clf_name in CLF_NAMES:
        m = mfr_models[clf_name]["model"]
        sum_probs += m.predict_proba(X_dev)     

    voted_mfrs =  np.argmax(sum_probs, axis=1)  
    voted_mfrs_name = [model_label_map[c] for c in voted_mfrs]
    
    if(verbs > 0):
        if(print_wrong):
            print_misses(true_mfr, voted_mfrs, "Manufacturer (Soft Vote)", model_label_map, group_pos_label, run_indices_mfr)
        report(true_mfr, voted_mfrs, "Manufacturer (Soft Vote)", model_label_map)

    ### chip voting (stage 2)
    for clf_name in CLF_NAMES:
        chip_pred_names = []
        top2_hits = []
        for cap in range(num_caps):
            cap_voted_mfr = voted_mfrs_name[cap]
            chip_model = get_device_model(clf_name, cap_voted_mfr)
            m = chip_model["model"]
            lm = chip_model["label_map"]
            cap_probs = m.predict_proba(X_dev[cap:cap+1])[0]
            order_idx = np.argsort(cap_probs)[::-1]
            top_names = [lm[m.classes_[k]] for k in order_idx[:2]]
            chip_pred_names.append(top_names[0]) ## top 1 prediction
            top2_hits.append(true_chip_names[cap] in top_names)

        ### sadly, this self mapping is necessary to make report() compatible, possible modularization of score can eliminate this
        identity_map = {name: name for name in set(true_chip_names) | set(chip_pred_names)} 
        if(print_wrong):
            print_misses(np.asarray(true_chip_names), np.asarray(chip_pred_names),f"{pos_name} - Chip via {clf_name} (routed by voted mfr)",identity_map, group_pos_label, run_indices_dev)
        chip_acc[pos_name][clf_name], chip_ci[pos_name][clf_name] = report(np.asarray(true_chip_names), np.asarray(chip_pred_names), 
                            f"{pos_name} - Chip via {clf_name} (routed by voted mfr)", identity_map)    
        if(top2_arg):
            top2_acc, top2_ci = bootstrap_proportion(top2_hits)
            chip_top2_acc[pos_name][clf_name] = top2_acc
            chip_top2_ci[pos_name][clf_name]  = top2_ci
            top2_hw = (top2_ci[1] - top2_ci[0])/2
            print(f"Top-2 accuracy: {sum(top2_hits)}/{num_caps} = {top2_acc:.3f} ± {top2_hw:.3f}")

if args.plot:
    outname = os.path.join("diff_pos", "diff_pos_vote.pdf")
    plot_accuracy_summary_pos([(chip_acc, chip_ci, "NIC Chip")], outname)
    if top2_arg:
        outname = os.path.join("diff_pos", "diff_pos_vote_top2.pdf")
        plot_accuracy_summary_pos([(chip_top2_acc, chip_top2_ci, "NIC Chip Top-2 Accuracy")], outname)