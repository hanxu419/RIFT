"""Create an ICEWS05-15 dataset with chronologically ordered time IDs."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_dataset import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/icews05-15")
    parser.add_argument("--output", type=Path, default=ROOT / "data/icews05-15-chrono")
    args = parser.parse_args()
    print(prepare(args.source, args.output, "icews05-15-chrono"))
