"""Measure the memory one raw daily file needs under four loading styles."""

import argparse
import os
import sys

import pandas as pd

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "telecom")
DEFAULT_FILE = "sms-call-internet-mi-2013-11-01.txt"

COLUMNS = ["square_id", "time_interval", "country_code",
           "sms_in", "sms_out", "call_in", "call_out", "internet"]
# These are the narrow types validated against the observed ranges.
NARROW = {"square_id": "int16", "time_interval": "int64", "internet": "float32"}
CHUNK_ROWS = 500000


def frame_mib(frame):
    """Return the memory held by one dataframe in mebibytes."""
    return frame.memory_usage(deep=True).sum() / 1048576.0


def read(path, columns, dtype=None, nrows=None):
    """Read the raw tab-separated file with the given columns and types."""
    return pd.read_csv(path, sep="\t", header=None, names=COLUMNS,
                       usecols=columns, dtype=dtype, nrows=nrows)


def main():
    parser = argparse.ArgumentParser(
        description="Measure dataframe memory for one raw daily file.")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="name of the raw daily file inside data/raw/telecom/")
    args = parser.parse_args()

    path = os.path.join(RAW_DIR, args.file)
    if not os.path.exists(path):
        print("Raw file not found: %s" % path)
        print("Run scripts/download_data.py first, because this benchmark reads raw data.")
        return 1

    rows = []
    full = read(path, COLUMNS)
    rows.append(("Full 8-column load, default dtypes", 8, frame_mib(full)))
    row_count = len(full)
    del full

    three = read(path, ["square_id", "time_interval", "internet"], NARROW)
    rows.append(("3 columns, reduced dtypes", 3, frame_mib(three)))
    del three

    two = read(path, ["square_id", "internet"],
               {"square_id": "int16", "internet": "float32"})
    rows.append(("2 columns used for aggregation", 2, frame_mib(two)))
    del two

    chunk = read(path, ["square_id", "internet"],
                 {"square_id": "int16", "internet": "float32"}, nrows=CHUNK_ROWS)
    rows.append(("One chunk of %s rows" % format(CHUNK_ROWS, ","), 2, frame_mib(chunk)))
    del chunk

    table = pd.DataFrame(rows, columns=["approach", "columns", "memory_MiB"])
    baseline = table.loc[0, "memory_MiB"]
    table["reduction_vs_baseline_%"] = (
        100 * (baseline - table["memory_MiB"]) / baseline).round(2)
    table["memory_MiB"] = table["memory_MiB"].round(2)

    print("file        : %s" % args.file)
    print("rows        : %s" % format(row_count, ","))
    print()
    print(table.to_string(index=False))
    print()
    print("These figures are the memory held by the dataframe itself.")
    print("The peak working set of the whole process is a different measurement,")
    print("and scripts/prepare_data.py reports it for the totals phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
