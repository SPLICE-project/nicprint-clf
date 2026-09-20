"""
Sanity-check one IR pickle from proc_acks_ablation.py.

Usage:
    python inspect_stream.py                                  # default: all_pi_bcm/pi_bcm_0
    python inspect_stream.py --device all_mac_bcm --run mac_bcm_5
    python inspect_stream.py --pkl /full/path/to/run.pkl
"""
import os
import pickle
import argparse
from dotenv import load_dotenv

load_dotenv()

DEFAULT_STREAMS_DIR = os.path.expanduser(os.environ[f"BASE_DIR_ABL"])


parser = argparse.ArgumentParser()
parser.add_argument("--streams-dir", default=DEFAULT_STREAMS_DIR)
parser.add_argument("--device", help='subdir under streams-dir, e.g. "all_mac_bcm"')
parser.add_argument("--run",    help='run pickle stem (no .pkl), e.g. "mac_bcm_5"')
parser.add_argument("--pkl", default=None, help="Full path to a run pkl (overrides --device/--run)")
args = parser.parse_args()

if args.pkl:
    path = args.pkl
else:
    if (args.device is None) != (args.run is None):
        parser.error(
            "--device and --run must be passed together. "
            'Example: --device all_mac_bcm --run mac_bcm_5  '
            "(loads <streams-dir>/all_mac_bcm/mac_bcm_5.pkl)"
        )
    device = args.device or "all_pi_bcm"
    run = args.run or "pi_bcm_0"
    path = os.path.join(args.streams_dir, device, f"{run}.pkl")

print(f"Loading {path}\n")
with open(path, "rb") as f:
    d = pickle.load(f)

keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
print(f"{len(keys)} stf levels (expect 32)")
print(f"first 5 keys: {keys[:5]}")
print(f"last  5 keys: {keys[-5:]}\n")

print(f"{'stf_level':<15} {'n_obs':>6} {'len':>5} {'valid_ack_ratio':>18}  {'first_20_flags'}")
for k in keys:
    entry = d[k]
    n_obs = entry["n_observed"]
    arr = entry["is_valid_ack"]
    ratio = float(arr.mean()) if arr.size else float("nan")
    head = " ".join(str(int(v)) for v in arr[:20])
    print(f"{k:<15} {n_obs:>6} {arr.size:>5} {ratio:>18.3f}  {head}")