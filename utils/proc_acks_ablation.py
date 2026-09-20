'''
Resampler for ablation study. Takes existing pcaps from real 
transmission and creates basically hot-one-encondings for 
each "real" transmission. This hot-one-encoding can be used
to create resampled versions of a experimet simulating that less
tranmssion were conducted by randomly piciking L transmissions. 

You can use inspect_stram.py to visualize how the output
of this scripts looks like

Mechanics:
Takes <device>/<run>/minus_<x>_stf/help_cap.pcap, walks the file once and
emits a length-1000 array per stf-corruption n. Each entry is one assumed tx
attempt (1000 attempts as default). Note that currently the 1000 attempts are
hardcoded, but in the future it should be cmd arg. 

Walk possibilities for a frame in a  "real" pcap:
    Type A: visible tx frame followed by a valid ACK   -> obs = 1 (success)
    Type B: visible tx frame, next packet is not a valid ACK -> obs = 0 (fail)
    Type C: valid ACK with no preceding tx (ghost tx)  -> obs = 1 (success, attempt was made)
    Type D: tx + ACK both invisible -> not observed (assumed 0 )

Note that the existing frames are assumed to be uniformly distributed across the 
transmission window. Type D frames are filled after the distribution of "real" 
frames

Tx-frame match: addr2 = TX_TA AND addr3 = TX_SA.
Valid ACK match: control-type ACK (type=1, subtype=13) AND addr1 = ACK_RA.

Output layout:
    dict[str, dict]: { "minus_5_stf": {"n_observed": int, "is_valid_ack": np.uint8(1000)}, ... }

Re-runs skip already-processed runs unless --force is passed.
'''
import time
import os
import re
import pickle
import argparse
from pathlib import Path
import numpy as np
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_CAPTURES_DIR = os.path.expanduser(os.environ[f"BASE_DIR_SCAL"])
DEFAULT_STREAMS_DIR  = os.path.expanduser(os.environ[f"BASE_DIR_ABL"])

# Address constants — these are hard coded as per the experiment
## probably should be passes as cmd line args
TX_TA  = "22:23:23:23:23:00"  # transmitter (addr2) of a tx frame
TX_SA  = "23:23:23:23:23:23"  # source (addr3) of a tx frame
ACK_RA = "22:23:23:23:23:00"  # addr1 of a valid ACK


DOT11_TYPE_CONTROL = 1
DOT11_SUBTYPE_ACK  = 13

NUM_TX = 1000  

STF_PATTERN = re.compile(r"^minus_(\d+)_stf$")


def is_tx_frame(pkt):
    if not pkt.haslayer(Dot11):
        return False
    d = pkt[Dot11]
    return d.addr2 == TX_TA and d.addr3 == TX_SA


def is_valid_ack(pkt):
    if not pkt.haslayer(Dot11):
        return False
    d = pkt[Dot11]
    if d.type != DOT11_TYPE_CONTROL or d.subtype != DOT11_SUBTYPE_ACK:
        return False
    return d.addr1 == ACK_RA


def build_observations(pkts):
    obs = []
    n = len(pkts)
    skip_next = False
    for i in range(n):
        if skip_next:
            skip_next = False
            continue
        pkt = pkts[i]
        if is_tx_frame(pkt):
            if i + 1 < n and is_valid_ack(pkts[i + 1]):
                obs.append(1)        # Type A
                skip_next = True     # consume the matched ACK
            else:
                obs.append(0)        # Type B
        elif is_valid_ack(pkt):
            obs.append(1)            # Type C — ghost tx
    return obs


def uniform_spread(obs, n_slots=NUM_TX):
    arr = np.zeros(n_slots, dtype=np.uint8)
    n_obs = len(obs)
    if n_obs == 0:
        return arr
    if n_obs >= n_slots:
        # More (or equal) observations than assumed attempts — keep first n_slots in order
        ## this should not happen
        arr[:] = obs[:n_slots]
        return arr
    for i, v in enumerate(obs):
        slot = int(round(i * n_slots / n_obs))
        if slot >= n_slots:
            slot = n_slots - 1
        arr[slot] = v
    return arr


def process_pcap(pcap_path):
    ## Parse one help_cap.pcap. Returns (n_observed, length-1000 uint8 array).
    pkts = rdpcap(str(pcap_path))
    obs = build_observations(pkts)
    arr = uniform_spread(obs)
    return len(obs), arr


def process_run(run_path):
    ## Walk all minus_*_stf dirs in one run dir
    out = {}
    for stf_dir in sorted(run_path.iterdir(), key=lambda p: p.name):
        if not stf_dir.is_dir() or not STF_PATTERN.match(stf_dir.name):
            continue
        pcap_path = stf_dir / "help_cap.pcap"
        if not pcap_path.exists():
            # print(f"ERROR! MISSING: {pcap_path} -- Ignoring cap")
            continue
        try:
            n_obs, arr = process_pcap(pcap_path)
        except Exception as e:
            # print(f" ERROR parsing {pcap_path}: {e}")
            continue
        out[stf_dir.name] = {"n_observed": n_obs, "is_valid_ack": arr}
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--captures_dir", default=DEFAULT_CAPTURES_DIR, help="directory where the pcaps are (BASE_DIR_SCAL in .env used as default)")
parser.add_argument("--streams_dir",  default=DEFAULT_STREAMS_DIR, help="Directory where to save the output (BASE_DIR_ABL in .env used as default)")
parser.add_argument("--devices", nargs="+", default=None, help="Optional list of device dir names to process (default: all)")
parser.add_argument("--force", action="store_true", help="Reparse even if output pkl already exists")
args = parser.parse_args()

captures_root = Path(args.captures_dir)
streams_root  = Path(args.streams_dir)
streams_root.mkdir(parents=True, exist_ok=True)

if args.devices:
    device_dirs = args.devices
else:
    device_dirs = sorted(d.name for d in captures_root.iterdir() if d.is_dir())

t0 = time.time()
for device in device_dirs:
    device_path = captures_root / device
    print(f"Processing {device_path}")
    if not device_path.is_dir():
        print(f"Skipping {device} (not a dir under {captures_root})")
        continue

    out_device_dir = streams_root / device
    out_device_dir.mkdir(parents=True, exist_ok=True)

    ### run = a stf level (i.e. a specific run of frames with <stf level> samples set to zero)
    runs = sorted(r.name for r in device_path.iterdir() if r.is_dir())
    # print(f"\nDevice {device}: {len(runs)} runs")

    for run in tqdm(runs, desc=device, unit="run"):
        out_path = out_device_dir / f"{run}.pkl"
        if out_path.exists() and not args.force:
            continue
        data = process_run(device_path / run) 
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        # tqdm.write(f" Saved {out_path.name} ({len(data)} stf levels)")
elapsed = time.time() - t0
print(f"Total Elapsed Time = {elapsed} s")
