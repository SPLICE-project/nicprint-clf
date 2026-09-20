import sys
import os
import argparse
import ana_funcs as af
from data_processing import build_dataset, build_dataset_by_manufacturer, load_device_caps
from model_perf import load_model, score, plot_accuracy_summary

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", default="models/base", help="Directory containing saved .pkl models")
parser.add_argument("--print_missclf", action="store_true", help="Print Missclassified Samples?")
parser.add_argument("--plot", action="store_true", help="Plot Results?")
args = parser.parse_args()

## name of model files
MODEL_FILES = {
    "RandomForest":       ("randomforest_chip_model.pkl",       "randomforest_manufacturer_model.pkl"),
    "SVM":                ("svm_chip_model.pkl",                "svm_manufacturer_model.pkl"),
    "LogisticRegression": ("logisticregression_chip_model.pkl", "logisticregression_manufacturer_model.pkl"),
}

## correct profile from .env
af.use_profile("UNSEEN")


######################### Load data from different devices w same chipset ##############################
print("####### Cap Loading Started #########", flush=True)
device_caps = {
    "MT7961":    load_device_caps("AWUS036AXML Second Dev",    "all_awusacm_diff_media/acm_diff_media_{i}_acks.pkl", 30), ## note incorrect naming given when saving files
    "QCA9377": load_device_caps("QCA9377 Second Dev", "all_qca_scnd/qca_scnd_{i}_acks.pkl",30),
    "BCM4360": load_device_caps("BCM4360 iMac 2014", "all_imac_bcm/imac_bcm_{i}_acks.pkl", 30),
    "RTL8821AU": load_device_caps("RTL8821AU BrosTrend AC650", "all_brostrend_rtl/brostrend_rtl_{i}_acks.pkl", 30),
    "AC8265":load_device_caps("Dell Latitude Second Dev","all_latitude_scnd/latitude_scnd_intel_{i}_acks.pkl", 30),
}
print("####### Cap Loading Done #########")

if not device_caps:
    print("No test data!")
    sys.exit(-1)


############# Put data in expected format #############
X_dev, y_dev, dev_label_map, run_indices_dev, cap_label = build_dataset(device_caps, mcs_index=0)
X_mfr, y_mfr, mfr_label_map, run_indices_mfr, cap_label = build_dataset_by_manufacturer(device_caps, mcs_index=0)

print(f"\nTest set: {X_dev.shape[0]} samples, {X_dev.shape[1]} features\n")


#### Run Inferences ########
chip_acc = {} ## chip accuracy
mfr_acc  = {} ## manufacturer accuracy
chip_ci  = {} ## chip 95% confidence intervals
mfr_ci   = {} ## manufacturer 95% confidence interval

for clf_name, (dev_file, mfr_file) in MODEL_FILES.items():
    print("=" * 60)
    print(f"  {clf_name}")
    print("=" * 60)

    dev_model, dev_lm = load_model(os.path.join(args.models_dir, dev_file))
    mfr_model, mfr_lm = load_model(os.path.join(args.models_dir, mfr_file))

    chip_acc[clf_name], chip_ci[clf_name] = score(dev_model, dev_lm, X_dev, y_dev, dev_label_map, run_indices_dev, kind="Chip", 
                                                  group_names=cap_label, print_misclf=args.print_missclf)

    mfr_acc[clf_name], mfr_ci[clf_name] = score(mfr_model, mfr_lm, X_mfr, y_mfr, mfr_label_map, 
                                                run_indices_mfr,kind="Manufacturer", group_names=cap_label, print_misclf=args.print_missclf)
    print("\n")


### Plot Results (Optional via arg) ########

if (args.plot):
    outname = os.path.join("diff_dev", "diff_dev.pdf")
    plot_accuracy_summary({"NIC Chip": (chip_acc, chip_ci), "Manufacturer": (mfr_acc, mfr_ci)}, outname)