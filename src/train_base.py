import os
import time
import argparse
import ana_funcs as af
from data_processing import build_dataset, load_device_caps, build_dataset_by_manufacturer
from classifiers import evaluate, train_and_save
from model_perf import plot_confusion_matrix

def eval_train_save(X, y, lm, ri, kind):
    for clf_name in ["RandomForest", "SVM", "LogisticRegression"]:
        print("=" * 60)
        print(f"  {clf_name} - {kind}")
        print("=" * 60)
        t0 = time.time()
        results = evaluate(X, y, lm, kind, ri, args.print_missclf, clf_name=clf_name, n_splits=5)
        elapsed = time.time() - t0

        print("=" * 30)
        print(f"  {clf_name} - {kind} Summary")
        print("=" * 30)
        print(f"Evaluation time: {elapsed:.3f}s")
        print(f"Fold accuracies: {[f'{a:.3f}' for a in results['fold_accuracies']]}")
        print(f"Mean accuracy:   {results['mean_accuracy']:.3f}")
        ci_hw = (results["ci_hi"] - results["ci_lo"]) / 2
        print(f"Aggregated CV accuracy: {results['aggregated_accuracy']:.3f} ± {ci_hw:.3f}"
                f"(95% CI, B=1000, on pooled out-of-fold predictions)\n")
       
        for fold_i, params in enumerate(results["best_params_per_fold"]):
            print(f"Fold {fold_i+1} best params: {params}")

        if(args.cfm):
            save_path = os.path.join("figs/cfm", f"{clf_name.lower()}_model_{kind}_cm.pdf")
            plot_confusion_matrix(results["confusion_matrix"], lm, clf_name, save_path)

        ### retrain with full set 
        if(args.save):
            save_path = os.path.join(args.models_dir, f"{clf_name.lower()}_{kind}_model.pkl")
            train_and_save(X,y, lm, clf_name,save_path)

parser = argparse.ArgumentParser()
parser.add_argument("--cfm", action="store_true", help="Save confusion matrix to disk")
parser.add_argument("--print_missclf", action="store_true", help="Print Missclassified Samples?")
parser.add_argument("--models_dir", default="models/base", help="Directory where model .pkl will be save")
parser.add_argument("--save", action="store_true", help="Train on all data and save models to disk")
parser.add_argument("--kind", default="both", help="Options: chip, mfr, both (chip & mfr), special (per manufacturer chip clf)")

args = parser.parse_args()


## set profile to 1m-LOS 
af.use_profile("SCAL")

device_caps = {
    ### BCM
    "BRCM4345C0":    load_device_caps("BRCM4345C0 (RPi 4)",    "all_pi_bcm/pi_bcm_{i}_acks.pkl"),
    "BCM4331": load_device_caps("BCM4331 (MacBook Pro 2012)", "all_mac_bcm/mac_bcm_{i}_acks.pkl"),
    "BCM4356": load_device_caps("BCM4356 (Motorola Nexus 6)", "all_nexus_bcm/nexus_bcm_{i}_acks.pkl"),
    "BCM4360": load_device_caps("BCM4360 (MacBook Pro 2013)", "all_mac2015_bcm/mac2015_bcm_{i}_acks.pkl"),
    "BCM4364": load_device_caps("BCM4364 (MacBook Pro 2019)", "all_mac2019_bcm/mac2019_bcm_{i}_acks.pkl"),
    
    ### REALTEK
    "RTL8821AU":      load_device_caps("RTL8821AU (TP-Link AC600)", "all_tpac_real/tpac_real_{i}_acks.pkl"),
    "RTL8812AU": load_device_caps("RTL8812AU (Linksys WUSB6300)", "all_linksys_real/linksys_real_{i}_acks.pkl"),
    "RTL8852CE": load_device_caps("RTL8852CE", "all_rtl8852_rtl/rtl8852_rtl_{i}_acks.pkl"),
    "RTL8822CE": load_device_caps("RTL8822CE", "all_rtl8822_rtl/rtl8822_rtl_{i}_acks.pkl"),
    "RTL8814AU": load_device_caps("RTL8814AU", "all_awus19/awus19_media_{i}_acks.pkl"),
    
    ##Mediatek
    "MT7961": load_device_caps("MT7961 (AWUS036AXML)", "all_awusxml_media/awusxml_media_{i}_acks.pkl"),
    "MT7612U": load_device_caps("MT7612U (AWUS036ACM)", "all_awusacm_media/awusacm_media_{i}_acks.pkl"),
    "MT6765": load_device_caps("MT6765 Helio (Motorola G Power)", "all_moto_media/moto_media_{i}_acks.pkl" ),
    "MT7925":load_device_caps("MT7925", "all_mt7925_media/mt7925_media_{i}_acks.pkl"),
    "MT7921":load_device_caps("MT7921", "all_mt7921_media/mt7921_media_{i}_acks.pkl"),
    
    ## Intel
    "AC9560": load_device_caps("Intel AC9560 (Asus ROG)", "all_rog_intel/rog_intel_{i}_acks.pkl"),
    "AX201": load_device_caps("Intel AX201 (Lenovo Yoga i7)", "all_yoga_intel/yoga_intel_{i}_acks.pkl"),
    "AC8265": load_device_caps("Intel AC8265 (Dell Latitude)", "all_latitude_intel/latitude_intel_{i}_acks.pkl"),
    "AX210": load_device_caps("Intel AX210", "all_ax210_intel/ax210_intel_{i}_acks.pkl"),
    "AC3160": load_device_caps("Intel AC3160", "all_ac3160_intel/ac3160_intel_{i}_acks.pkl"),

    ## Qualcomn
    "QCA9377": load_device_caps("QCA9377", "all_qca9377_qcm/qca9377_qca_{i}_acks.pkl"),
    "QCA6174": load_device_caps("QCA6174", "all_qca6174_qcm/qca6174_qcm_{i}_acks.pkl"),
    "QCA9984": load_device_caps("QCA9984", "all_qca9984/qca9984_{i}_acks.pkl"),
    "WCN7850": load_device_caps("QCNCM865", "all_qcawifi7/qcawifi7_qcm_{i}_acks.pkl"),
    "IPQ4019": load_device_caps("IPQ4019 (Netgear Orbi)", "all_ipq4019/ipq4019_qcm_{i}_acks.pkl")
}

# Remove any devices that had zero successful loads
device_caps = {k: v for k, v in device_caps.items() if len(v) > 0}

##################### Training Device models ###########################################
if(args.kind == "chip" or args.kind == "both"):
    print("=" * 80)
    print(f"Running Chipset Model Training")
    print("=" * 80)
    X_dev, y_dev, dev_label_map, dev_run_indices, _ = build_dataset(device_caps, mcs_index=0)
    eval_train_save(X_dev, y_dev,dev_label_map, dev_run_indices, "chip")
if(args.kind == "mfr" or args.kind == "both"):
    print("=" * 80)
    print(f"Running Manufacturer Model Training")
    print("=" * 80)
    X_mfr, y_mftr, mfr_label_map, mfr_run_indeices, _ = build_dataset_by_manufacturer(device_caps, mcs_index=0)
    eval_train_save(X_mfr, y_mftr, mfr_label_map, mfr_run_indeices, "manufacturer")
if(args.kind == "special"):
    print("=" * 80)
    print(f"Running Chipset Model via Manufacturer Specialized Training")
    print("=" * 80)
    #### organize caps by manufacturer
    manufacturer_groups = {} 
    for device_name, caps in device_caps.items():
        mfr = caps[0].manufacturer
        if mfr not in manufacturer_groups:
            manufacturer_groups[mfr] = {}
        manufacturer_groups[mfr][device_name] = caps

    for mfr, mfr_caps in manufacturer_groups.items():
        X, y, label_map, run_indices, _ = build_dataset(mfr_caps, mcs_index=0)
        eval_train_save(X,y,label_map,run_indices, f"{mfr.lower()}_device")

if args.kind not in ("chip", "mfr", "both", "special"):
    print(f"Error: Invalid kind argument {args.kind}")