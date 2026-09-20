import ana_funcs as af
import numpy as np

def load_device_caps(device_name, file_pattern, num_runs=100):
    caps = []
    for i in range(num_runs):
        fname = file_pattern.format(i=i)
        caps.append(af.MY_CAP.load(fname))
    print(f"Loaded {len(caps)} runs for {device_name}")
    return caps

def correct_vector(cap, mcs_index=0):
    helper = cap.dec_acks_helper
    sdr = cap.dec_acks
    norm = 1000 
   

    # Correct curve
    corrected = np.empty(len(helper))
    for i in range(len(helper)):
        val = helper[i][mcs_index]
        if val == -1 or val > 1000:
            corrected[i] = sdr[i][mcs_index] 
        else:
            corrected[i] = val 

    corrected = corrected / norm

    return np.clip(corrected, 0, 1)


def build_dataset(device_caps, mcs_index=0):
    """
    Parameters:
        device_caps: 
            - dict[str, list[MY_CAP]]  --- This is the format used in all py script                          
            - list of (label, group, caps) --- Kept for backcomp with old scripts              
        mcs_index: which MCS column to use (default 0).

    Returns:
        X: np.ndarray of shape (n_samples, 32) -- Data data in the format the model expects
        y: np.ndarray of shape (n_samples) with integer labels -- Labels as integers
        label_map: dict mapping integer label -> label string
        run_indices: np.ndarray, run_index used to tag the file from which the capture comes from 
        group_names: np.ndarray of str; an arbitrary group tag for each sample (equals the label when no explicit group was provided)
    """
    if isinstance(device_caps, dict):
        items = [(name, name, caps) for name, caps in device_caps.items()]
    else:
        items = [t if len(t) == 3 else (t[0], t[0], t[1]) for t in device_caps]

    X_list = []
    y_list = []
    run_indices = []
    group_names = []
    label_map = {}
    label_to_idx = {}

    for label, group, caps in items:
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
            label_map[label_to_idx[label]] = label
        label_idx = label_to_idx[label]
        for run_i, cap in enumerate(caps):
            vec = correct_vector(cap, mcs_index)
            X_list.append(vec)
            y_list.append(label_idx)
            run_indices.append(run_i)
            group_names.append(group)

    X = np.array(X_list)
    y = np.array(y_list)
    run_indices = np.array(run_indices)
    group_names = np.array(group_names)

    return X, y, label_map, run_indices, group_names


def build_dataset_by_manufacturer(device_caps, mcs_index=0):
    ### Same as function above but returns manufacturer labels instead
    if isinstance(device_caps, dict):
        items = [(name, name, caps) for name, caps in device_caps.items()]
    else:
         items = [t if len(t) == 3 else (t[0], t[0], t[1]) for t in device_caps]

    X_list = []
    y_list = []
    run_indices = []
    group_names = []
    manufacturer_to_label = {}
    label_map = {}
    next_label = 0

    for _, group, caps in items:
        for run_i, cap in enumerate(caps):
            mfr = cap.manufacturer
            if mfr not in manufacturer_to_label:
                manufacturer_to_label[mfr] = next_label
                label_map[next_label] = mfr
                next_label += 1
                
            vec = correct_vector(cap, mcs_index)
            X_list.append(vec)
            y_list.append(manufacturer_to_label[mfr])
            run_indices.append(run_i)
            group_names.append(group)

    X = np.array(X_list)
    y = np.array(y_list)
    run_indices = np.array(run_indices)
    group_names = np.array(group_names)
    
    return X, y, label_map, run_indices, group_names