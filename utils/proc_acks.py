import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ana_funcs

NUM_MCS = 1


parser = argparse.ArgumentParser()
parser.add_argument("--captures_dir", help="root holding the raw pcap")
parser.add_argument("--out_dir", help="root where .pkl files are written")
parser.add_argument("--mfrs_map", help="JSON file mapping device-chipset dir name -> manufacturer")
parser.add_argument("--force", action="store_true", help="Reprocess even if output pkl already exists")
args = parser.parse_args()

captures_root = Path(args.captures_dir).expanduser()
out_root = Path(args.out_dir).expanduser()

## ana_funcs globals are None until use_profile() is called; set them from argv.
## get_acks_mcs() inserts 'captures/' itself, so base_dir is one level up.
## save_obj() prepends save_pick, so it gets a path relative to out_root.
ana_funcs.base_dir = str(captures_root)
ana_funcs.save_pick = str(out_root)

with open(args.mfrs_map) as f:
    MANUFACTURERS = json.load(f)

chipsets = sorted(d.name for d in captures_root.iterdir() if d.is_dir())
print(f"{captures_root}: {len(chipsets)} chipsets")

t0 = time.time()
n_done, n_skipped, n_failed = 0, 0, 0

for chipset in chipsets:
    if chipset not in MANUFACTURERS:
        print(f"No manufacturer for '{chipset}' in {args.mfrs_map} — skipping chipset")
        continue
    mft = MANUFACTURERS[chipset]

    chipset_path = captures_root / chipset
    out_chipset = out_root / chipset
    out_chipset.mkdir(parents=True, exist_ok=True)

    runs = sorted(d.name for d in chipset_path.iterdir() if d.is_dir())
    print(f"\n{chipset}: {len(runs)} runs, manufacturer={mft}")

    for run in tqdm(runs, desc=chipset, unit="run"):
        name = f"{chipset}/{run}"
        out_path = out_chipset / f"{run}_acks.pkl"

        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        try:
            mycap = ana_funcs.MY_CAP(name, NUM_MCS, mft)
            mycap.process_acks()
            mycap.save_obj(f"{chipset}/{run}_acks.pkl")
        except Exception as e:
            tqdm.write(f"ERROR processing {name}: {e}")
            n_failed += 1
            continue

        n_done += 1

elapsed = time.time() - t0
print(f"\nDone: {n_done} processed, {n_skipped} skipped, {n_failed} failed")
print(f"Total Elapsed Time = {elapsed} s")