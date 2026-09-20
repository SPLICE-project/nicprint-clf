import sys
import os
import argparse
from data_processing import build_dataset, build_dataset_by_manufacturer, load_device_caps
import ana_funcs as af
from model_perf import load_model, score, plot_accuracy_summary_pos


parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", default="models/base", help="Directory containing saved .pkl models")
parser.add_argument("--print_missclf",action="store_true", help="Print Missclassified Samples?")
parser.add_argument("--plot", action="store_true", help="Plot Results?")
args = parser.parse_args()

## name of model files
MODEL_FILES = {
    "RandomForest":       ("randomforest_chip_model.pkl",       "randomforest_manufacturer_model.pkl"),
    "SVM":                ("svm_chip_model.pkl",                "svm_manufacturer_model.pkl"),
    "LogisticRegression": ("logisticregression_chip_model.pkl", "logisticregression_manufacturer_model.pkl"),
}

## correct profile from .env
af.use_profile("DIFFPOS")
print("####### Cap Loading Started #########", flush=True)
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
    ("MT6765",    "Motoroal G Power Pos 2",        load_device_caps("Moto G Power 2", "all_moto_pos2/moto_pos2_{i}_acks.pkl", 30)),
]

######################### Position 3 -- LOS 5m ##############################
device_caps_pos3 = [
    ("AC9560",    "ROG Pos 3",               load_device_caps("ROG Pos 3", "all_rog_pos3/rog_pos3_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 3",  load_device_caps("MacBook Pro 2012 Pos 3", "all_mac_pos3/mac_pos3_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 3",   load_device_caps("BrosTrend AC650 Pos 3", "all_brostrend_pos3/brostrend_pos3_{i}_acks.pkl", 30)),
    ("MT6765",    "Motoroal G Power Pos 3",   load_device_caps("Moto G Power 3", "all_moto_pos3/moto_pos3_{i}_acks.pkl", 30)),
]

######################### Position 4 -- Inside Carbord Box 8m  ##############################
device_caps_pos4 = [
    ("AC9560",    "ROG Pos 4",               load_device_caps("ROG Pos 4", "all_rog_pos4/rog_pos4_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 4",  load_device_caps("MacBook Pro 2012 Pos 4", "all_mac_pos4/mac_pos4_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 4",   load_device_caps("BrosTrend AC650 Pos 4", "all_brostrend_pos4/brsotrend_pos4_{i}_acks.pkl", 30)),
    ("MT6765",    "Motoroal G Power Pos 4",   load_device_caps("Moto G Power 4", "all_moto_pos4/moto_pos4_{i}_acks.pkl", 30)),
]

######################### Position 5 -- LOS 1m  ##############################
device_caps_pos5 = [
    ("AC9560",    "ROG Pos 5",               load_device_caps("ROG Pos 5", "all_rog_pos5/rog_pos5_{i}_acks.pkl", 30)),
    ("BCM4331",   "MacBook Pro 2012 Pos 5",  load_device_caps("MacBook Pro 2012 Pos 5", "all_mac_pos5/mac_pos5_{i}_acks.pkl", 30)),
    ("RTL8821AU", "BrosTrend AC650 Pos 5",   load_device_caps("BrosTrend AC650 Pos 5", "all_brostrend_pos5/brostrend_pos5_{i}_acks.pkl", 30)),
    ("MT6765",    "Motoroal G Power Pos 5",   load_device_caps("Moto G Power 5", "all_moto_pos5/moto_pos5_{i}_acks.pkl", 30)),
]

# Collapase all positions into a single dir (keyed by the position number)
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


# Per-(position, classifier) metrics
chip_acc = {pos: {} for pos in positions}
mfr_acc  = {pos: {} for pos in positions}
chip_ci  = {pos: {} for pos in positions}
mfr_ci   = {pos: {} for pos in positions}

#### Run Inferences ########
for pos_name, device_caps in positions.items():
    print("#" * 70)
    print(f"  {pos_name}")
    print("#" * 70)

    X_dev, y_dev, dev_label_map, run_indices_dev, group_pos_label = build_dataset(device_caps, mcs_index=0)
    X_mfr, y_mfr, mfr_label_map, run_indices_mfr, group_pos_label = build_dataset_by_manufacturer(device_caps, mcs_index=0)
 
    print(f"\nTest set: {X_dev.shape[0]} samples, {X_dev.shape[1]} features\n")

    for clf_name, (dev_file, mfr_file) in MODEL_FILES.items():
        print("=" * 60)
        print(f"  {pos_name} - {clf_name}")
        print("=" * 60)

        dev_model, dev_lm = load_model(os.path.join(args.models_dir, dev_file))
        mfr_model, mfr_lm = load_model(os.path.join(args.models_dir, mfr_file))

        chip_acc[pos_name][clf_name], chip_ci[pos_name][clf_name] = score(dev_model, dev_lm, X_dev, y_dev, 
                dev_label_map, run_indices_dev,kind="Chip", group_names=group_pos_label, print_misclf=args.print_missclf)
        mfr_acc[pos_name][clf_name], mfr_ci[pos_name][clf_name] = score(mfr_model, mfr_lm, X_mfr, y_mfr, 
                mfr_label_map, run_indices_mfr,kind="Manufacturer", group_names=group_pos_label, print_misclf=args.print_missclf)


### Plot Results (Optional via arg)########

if (args.plot):
     outname = os.path.join("diff_pos", "diff_pos.pdf")
     plot_accuracy_summary_pos([(mfr_acc, mfr_ci, "NIC Manufacturer"), (chip_acc, chip_ci, "NIC Chipset")], outname)