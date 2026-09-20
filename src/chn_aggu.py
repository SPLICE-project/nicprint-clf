import os
import argparse
import ana_funcs as af
from data_processing import build_dataset, build_dataset_by_manufacturer, load_device_caps
from model_perf import load_model, score, plot_accuracy_summary_pos
from classifiers import train_and_save

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", default="models/chn_aug", help="Directory to write tuned per-fold pickles")
parser.add_argument("--print_missclf", action="store_true", help="Print Missclassified Samples?")
parser.add_argument("--plot", action="store_true", help="Plot Results?")
parser.add_argument("--train", action="store_true", help="Train model?")

args = parser.parse_args()
train_arg = args.train

## name of model files
CLF_NAMES = ["RandomForest", "SVM", "LogisticRegression"]
## internal names (for printing)
CLF_TAGS = {
    "RandomForest":       "randomforest",
    "SVM":                "svm",
    "LogisticRegression": "logisticregression",
}

### base caps only loaded if train arg is given
base_items = []
if(train_arg):
    print("#" * 70)
    print("  Loading original base data")
    print("#" * 70)
    ## base data profile
    af.use_profile("SCAL")

    base_items = [
        ### BCM
        ("BRCM4345C0", "base", load_device_caps("BRCM4345C0 (RPi 4)",           "all_pi_bcm/pi_bcm_{i}_acks.pkl", 100)),
        ("BCM4331",    "base", load_device_caps("BCM4331 (MacBook Pro 2012)",   "all_mac_bcm/mac_bcm_{i}_acks.pkl", 100)),
        ("BCM4356",    "base", load_device_caps("BCM4356 (Motorola Nexus 6)",   "all_nexus_bcm/nexus_bcm_{i}_acks.pkl", 100)),
        ("BCM4360",    "base", load_device_caps("BCM4360 (MacBook Pro 2013)",   "all_mac2015_bcm/mac2015_bcm_{i}_acks.pkl", 100)),
        ("BCM4364",    "base", load_device_caps("BCM4364 (MacBook Pro 2019)",   "all_mac2019_bcm/mac2019_bcm_{i}_acks.pkl", 100)),

        ### REALTEK
        ("RTL8821AU",  "base", load_device_caps("RTL8821AU (TP-Link AC600)",    "all_tpac_real/tpac_real_{i}_acks.pkl", 100)),
        ("RTL8812AU",  "base", load_device_caps("RTL8812AU (Linksys WUSB6300)", "all_linksys_real/linksys_real_{i}_acks.pkl", 100)),
        ("RTL8852CE",  "base", load_device_caps("RTL8852CE",                    "all_rtl8852_rtl/rtl8852_rtl_{i}_acks.pkl", 100)),
        ("RTL8822CE",  "base", load_device_caps("RTL8822CE",                    "all_rtl8822_rtl/rtl8822_rtl_{i}_acks.pkl", 100)),
        ("RTL8814AU",  "base", load_device_caps("RTL8814AU",                    "all_awus19/awus19_media_{i}_acks.pkl", 100)),

        ### MEDIATEK
        ("MT7961",     "base", load_device_caps("MT7961 (AWUS036AXML)",          "all_awusxml_media/awusxml_media_{i}_acks.pkl", 100)),
        ("MT7612U",    "base", load_device_caps("MT7612U (AWUS036ACM)",          "all_awusacm_media/awusacm_media_{i}_acks.pkl", 100)),
        ("MT6765",     "base", load_device_caps("MT6765 Helio (Moto G Power)",   "all_moto_media/moto_media_{i}_acks.pkl", 100)),
        ("MT7925",     "base", load_device_caps("MT7925",                        "all_mt7925_media/mt7925_media_{i}_acks.pkl", 100)),
        ("MT7921",     "base", load_device_caps("MT7921",                        "all_mt7921_media/mt7921_media_{i}_acks.pkl", 100)),

        ### INTEL
        ("AC9560",     "base", load_device_caps("Intel AC9560 (Asus ROG)",       "all_rog_intel/rog_intel_{i}_acks.pkl", 100)),
        ("AX201",      "base", load_device_caps("Intel AX201 (Lenovo Yoga i7)",  "all_yoga_intel/yoga_intel_{i}_acks.pkl", 100)),
        ("AC8265",     "base", load_device_caps("Intel AC8265 (Dell Latitude)",  "all_latitude_intel/latitude_intel_{i}_acks.pkl", 100)),
        ("AX210",      "base", load_device_caps("Intel AX210",                   "all_ax210_intel/ax210_intel_{i}_acks.pkl", 100)),
        ("AC3160",     "base", load_device_caps("Intel AC3160",                  "all_ac3160_intel/ac3160_intel_{i}_acks.pkl", 100)),

        ### QUALCOMM
        ("QCA9377",    "base", load_device_caps("QCA9377",                       "all_qca9377_qcm/qca9377_qca_{i}_acks.pkl", 100)),
        ("QCA6174",    "base", load_device_caps("QCA6174",                       "all_qca6174_qcm/qca6174_qcm_{i}_acks.pkl", 100)),
        ("QCA9984",    "base", load_device_caps("QCA9984",                       "all_qca9984/qca9984_{i}_acks.pkl", 100)),
        ("WCN7850",    "base", load_device_caps("QCNCM865",                      "all_qcawifi7/qcawifi7_qcm_{i}_acks.pkl", 100)),
        ("IPQ4019",    "base", load_device_caps("IPQ4019 (Netgear Orbi)",        "all_ipq4019/ipq4019_qcm_{i}_acks.pkl", 100)),
    ]


##### Position caps always loaded ####
print("\n" + "#" * 70)
print("  Loading DIFFPOS position data")
print("#" * 70)
af.use_profile("DIFFPOS")

pos_items_by_name = {
    ### Position 1 -- Behind Wall 3 m
    "Position 1": [
        ("MT6765",    "Position 1", load_device_caps("Moto G Power Pos 1",       "all_moto_pos1/moto_pos1_{i}_acks.pkl", 30)),
        ("AC9560",    "Position 1", load_device_caps("ROG Pos 1",                "all_rog_pos1/rog_pos1_{i}_acks.pkl", 30)),
        ("RTL8821AU", "Position 1", load_device_caps("BrosTrend AC650 Pos 1",    "all_brostrend_pos1/brostrend_pos1_{i}_acks.pkl", 30)),
        ("BCM4331",   "Position 1", load_device_caps("MacBook Pro 2012 Pos 1",   "all_mac_pos1/mac_pos1_{i}_acks.pkl", 30)),
    ],
    ### Position 2 -- Inside Metal Box 5 m
    "Position 2": [
        ("AC9560",    "Position 2", load_device_caps("ROG Pos 2",                "all_rog_pos2/rog_pos2_{i}_acks.pkl", 30)),
        ("BCM4331",   "Position 2", load_device_caps("MacBook Pro 2012 Pos 2",   "all_mac_pos2/mac2_pos_{i}_acks.pkl", 30)),
        ("RTL8821AU", "Position 2", load_device_caps("BrosTrend AC650 Pos 2",    "all_brostrend_pos2/brostrend_pos2_{i}_acks.pkl", 30)),
        ("MT6765",    "Position 2", load_device_caps("Moto G Power 2",           "all_moto_pos2/moto_pos2_{i}_acks.pkl", 30)),
    ],
    ### Position 3 -- LOS 5 m
    "Position 3": [
        ("AC9560",    "Position 3", load_device_caps("ROG Pos 3",                "all_rog_pos3/rog_pos3_{i}_acks.pkl", 30)),
        ("BCM4331",   "Position 3", load_device_caps("MacBook Pro 2012 Pos 3",   "all_mac_pos3/mac_pos3_{i}_acks.pkl", 30)),
        ("RTL8821AU", "Position 3", load_device_caps("BrosTrend AC650 Pos 3",    "all_brostrend_pos3/brostrend_pos3_{i}_acks.pkl", 30)),
        ("MT6765",    "Position 3", load_device_caps("Moto G Power 3",           "all_moto_pos3/moto_pos3_{i}_acks.pkl", 30)),
    ],
    ### Position 4 -- Inside Cardboard Box 8 m
    "Position 4": [
        ("AC9560",    "Position 4", load_device_caps("ROG Pos 4",                "all_rog_pos4/rog_pos4_{i}_acks.pkl", 30)),
        ("BCM4331",   "Position 4", load_device_caps("MacBook Pro 2012 Pos 4",   "all_mac_pos4/mac_pos4_{i}_acks.pkl", 30)),
        ("RTL8821AU", "Position 4", load_device_caps("BrosTrend AC650 Pos 4",    "all_brostrend_pos4/brsotrend_pos4_{i}_acks.pkl", 30)),
        ("MT6765",    "Position 4", load_device_caps("Moto G Power 4",           "all_moto_pos4/moto_pos4_{i}_acks.pkl", 30)),
    ],
    ### Position 5 -- LOS 1 m
    "Position 5": [
        ("AC9560",    "Position 5", load_device_caps("ROG Pos 5",                "all_rog_pos5/rog_pos5_{i}_acks.pkl", 30)),
        ("BCM4331",   "Position 5", load_device_caps("MacBook Pro 2012 Pos 5",   "all_mac_pos5/mac_pos5_{i}_acks.pkl", 30)),
        ("RTL8821AU", "Position 5", load_device_caps("BrosTrend AC650 Pos 5",    "all_brostrend_pos5/brostrend_pos5_{i}_acks.pkl", 30)),
        ("MT6765",    "Position 5", load_device_caps("Moto G Power 5",           "all_moto_pos5/moto_pos5_{i}_acks.pkl", 30)),
    ],
}

pos_items = [item for items in pos_items_by_name.values() for item in items]
all_items = base_items + pos_items

############# Put data in expected format #############
X_dev, y_chip, chip_label_map, run_indices_dev, pos_grp = build_dataset(all_items, mcs_index=0)
X_mfr, y_mfr, mfr_label_map, run_indices_mfr, pos_grp = build_dataset_by_manufacturer(all_items, mcs_index=0)

##### Inferences #######
pos_names = list(pos_items_by_name.keys())
# Per-(position, classifier) metrics
chip_acc = {pos: {} for pos in pos_names}
mfr_acc  = {pos: {} for pos in pos_names}
chip_ci  = {pos: {} for pos in pos_names}
mfr_ci   = {pos: {} for pos in pos_names}


### make dirs for models if missing
os.makedirs(args.models_dir, exist_ok=True)

for held_out in pos_names:
    print("#" * 70)
    print(f"  LOPO fold: hold out {held_out}")
    print("#" * 70)

    train_mask = pos_grp != held_out
    test_mask  = pos_grp == held_out

    X_train_chip, X_test_chip = X_dev[train_mask], X_dev[test_mask]
    X_tran_mfr, X_test_mfr = X_mfr[train_mask], X_mfr[test_mask]

    y_train_chip, y_test_chip = y_chip[train_mask], y_chip[test_mask]
    y_train_mfr, y_test_mfr = y_mfr[train_mask], y_mfr[test_mask]

    for clf_name in CLF_NAMES:
        print("=" * 60)
        print(f" Held Out Pos: {held_out} - {clf_name}")
        print("=" * 60)

        ### classifer name mapping 
        pos_tag = held_out.lower().replace(" ", "_")
        dev_file = f"{CLF_TAGS[clf_name]}_lopo_{pos_tag}_chip.pkl"
        mfr_file = f"{CLF_TAGS[clf_name]}_lopo_{pos_tag}_manufacturer.pkl"
        
        if(train_arg):
            ### Train chipset model on all data minus held out
            print("Training Chipset Model")

            save_path = os.path.join(args.models_dir, dev_file)
            dev_model, dev_lm = train_and_save(X_train_chip, y_train_chip,chip_label_map, clf_name, save_path, random_state=None,  inner_splits=3)

            print("Training Manufacturer Model")
            save_path = os.path.join(args.models_dir, mfr_file)
            mfr_model, mfr_lm = train_and_save(X_tran_mfr, y_train_mfr,mfr_label_map, clf_name, save_path, random_state=None,  inner_splits=3)

            
        else:
            dev_model, dev_lm = load_model(os.path.join(args.models_dir, dev_file))
            mfr_model, mfr_lm = load_model(os.path.join(args.models_dir, mfr_file))

        chip_acc[held_out][clf_name], chip_ci[held_out][clf_name] = score(dev_model, dev_lm, X_test_chip, y_test_chip, 
                        chip_label_map, run_indices_dev,kind="Chip", group_names=pos_grp, print_misclf=args.print_missclf)

        mfr_acc[held_out][clf_name], mfr_ci[held_out][clf_name] = score(mfr_model, mfr_lm, X_test_mfr, y_test_mfr, mfr_label_map, 
                                                        run_indices_mfr,kind="Manufacturer", group_names=pos_grp, print_misclf=args.print_missclf)


### Plot (optional)
if (args.plot):
    outname = os.path.join("chn_agu", "chn_agu_held_out.pdf")
    plot_accuracy_summary_pos([(mfr_acc, mfr_ci, "NIC Manufacturer"), (chip_acc, chip_ci, "NIC Chipset")], outname)