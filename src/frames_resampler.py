'''
Script for resampling "real" frames! Note that 
this script expects a set of pcaps in the format 
produced by proc_ack_ablation.py

You should run proc_ack_ablation.py on the data first
'''

import os
import pickle
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()
## directory where the processed frame sets should be 
DEFAULT_STREAMS_DIR = os.path.expanduser(os.environ[f"BASE_DIR_ABL"])

DEVICE_LABELS = {
    "all_pi_bcm":         ("BRCM4345C0", "BCM"),
    "all_mac_bcm":        ("BCM4331",    "BCM"),
    "all_nexus_bcm":      ("BCM4356",    "BCM"),
    "all_mac2015_bcm":    ("BCM4360",    "BCM"),
    "all_mac2019_bcm":    ("BCM4364",    "BCM"),
    "all_tpac_real":      ("RTL8821AU",  "RTL"),
    "all_linksys_real":   ("RTL8812AU",  "RTL"),
    "all_rtl8852_rtl":    ("RTL8852CE",  "RTL"),
    "all_rtl8822_rtl":    ("RTL8822CE",  "RTL"),
    "all_awus19":         ("RTL8814AU",  "RTL"),
    "all_awusxml_media":  ("MT7961",     "MTK"),
    "all_awusacm_media":  ("MT7612U",    "MTK"),
    "all_moto_media":     ("MT6765",     "MTK"),
    "all_mt7925_media":   ("MT7925",     "MTK"),
    "all_mt7921_media":   ("MT7921",     "MTK"),
    "all_rog_intel":      ("AC9560",     "Intel"),
    "all_yoga_intel":     ("AX201",      "Intel"),
    "all_latitude_intel": ("AC8265",     "Intel"),
    "all_ax210_intel":    ("AX210",      "Intel"),
    "all_ac3160_intel":   ("AC3160",     "Intel"),
    "all_qca9377_qcm":    ("QCA9377",    "QCM"),
    "all_qca6174_qcm":    ("QCA6174",    "QCM"),
    "all_qca9984":        ("QCA9984",    "QCM"),
    "all_qcawifi7":       ("WCN7850",    "QCM"),
    "all_ipq4019":        ("IPQ4019",    "QCM"),
}

# default stf reduction level order: minus_5_stf, minus_10_stf, ..., minus_160_stf  (32 levels)
STF_LEVELS = [f"minus_{x}_stf" for x in range(5, 161, 5)]
assert len(STF_LEVELS) == 32


def resample_vector(run_data, Y, rng):
    """
    Params:
        run_data: dict {stf_level: {"n_observed": int, "is_valid_ack": np.uint8(1000)}} 
                Note that this should be a single stf level (i.e. unpack the dict in the pickle)
        Y: window size (number of assumed tx attempts)
        rng: np.random.Generator 

    Output: 
        vec: Feature vector for model input 
        note that if file is missing then vec entry is given 0.0. 
        there are few cases where this happens.
    """
    vec = np.empty(len(STF_LEVELS), dtype=np.float64)
    for i, stf in enumerate(STF_LEVELS):
        if stf not in run_data:
            vec[i] = 0.0
            continue
        arr = run_data[stf]["is_valid_ack"]
        n = arr.size
        start = int(rng.integers(0, n))
        idx = (start + np.arange(Y)) % n
        vec[i] = float(arr[idx].sum()) / Y
    return vec


def build_dataset(streams_root, Y, seed, device_labels=None):
    """
    Walk every (device, run) under streams_root, sample a Y-window per stf-level,
    and creates sets of 32-d vectors ready for model input (for step ablation the 32-d vectors 
    are narrowed in a different function).

    Note that this function is very similar to data_processing but with the ability to 
    resample a specific Y-window. It also returns both mfr and chip label at once.
    This is something data_processing should probably do to. 

    Params:
        streams_root: directory containing <device>/<run>.pkl tree
        Y: window size
        seed: int for the global RNG (same seed -> same windows across classifiers)
        device_labels: optional override for DEVICE_LABELS

    Output:
        X:               feature vecs stack together
        y_chip:          chip label
        y_mfr:           mfr label
        chip_label_map:  
        mfr_label_map:   
        run_indices:     
    """
    if device_labels is None:
        device_labels = DEVICE_LABELS

    rng = np.random.default_rng(seed)
    streams_root = Path(streams_root)

    X_list, chip_labels, mfr_labels, run_indices = [], [], [], []
    chip_to_idx, mfr_to_idx = {}, {}

    for device, (chip, mfr) in device_labels.items():
        ddir = streams_root / device
        if not ddir.is_dir():
            print(f"WARNING: {ddir} not found, skipping {device}")
            continue

        if chip not in chip_to_idx:
            chip_to_idx[chip] = len(chip_to_idx)
        if mfr not in mfr_to_idx:
            mfr_to_idx[mfr] = len(mfr_to_idx)

        for run_pkl in sorted(ddir.glob("*.pkl")):
            with open(run_pkl, "rb") as f:
                run_data = pickle.load(f)
            vec = resample_vector(run_data, Y, rng)
            X_list.append(vec)
            chip_labels.append(chip_to_idx[chip])
            mfr_labels.append(mfr_to_idx[mfr])
            try:
                run_idx = int(run_pkl.stem.rsplit("_", 1)[-1])
            except ValueError:
                run_idx = -1
            run_indices.append(run_idx)

    X = np.array(X_list)
    y_chip = np.array(chip_labels)
    y_mfr  = np.array(mfr_labels)
    run_indices = np.array(run_indices)
    chip_label_map = {v: k for k, v in chip_to_idx.items()}
    mfr_label_map  = {v: k for k, v in mfr_to_idx.items()}

    return X, y_chip, y_mfr, chip_label_map, mfr_label_map, run_indices
