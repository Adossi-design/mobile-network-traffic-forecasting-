"""Build the processed traffic files from the raw telecom dataset."""

import argparse
import ctypes
import ctypes.wintypes
import glob
import os
import time

import numpy as np
import pandas as pd

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "telecom")
PROCESSED = os.path.join(ROOT, "data", "processed")
TOTALS_OUT = os.path.join(PROCESSED, "total_traffic_by_square.csv")
SERIES_OUT = os.path.join(PROCESSED, "selected_area_timeseries.csv")

COLS = ["square_id", "time_interval", "country_code",
        "sms_in", "sms_out", "call_in", "call_out", "internet"]

# Phase 1 reads two columns and phase 2 also needs the timestamp.
TOTALS_USE = ["square_id", "internet"]
TOTALS_DTYPES = {"square_id": "int16", "internet": "float32"}
SERIES_USE = ["square_id", "time_interval", "internet"]
SERIES_DTYPES = {"square_id": "int16", "time_interval": "int64",
                 "internet": "float32"}

MAX_SQUARE = 10000
MB = 1048576.0

# These are the three busiest areas plus the two the assignment asks for.
SQUARES = [5161, 5059, 5259, 4159, 4556]
TOP_AREAS = [5161, 5059, 5259]

CHUNK = 500000
TZ = "Europe/Rome"
PERIOD_START = "2013-11-01 00:00"
PERIOD_END = "2014-01-01 23:50"


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb():
    """Return the peak working set of this process in megabytes."""
    try:
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
    except (AttributeError, OSError):
        return float("nan")  # The measurement needs Windows.
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE,
                                           ctypes.POINTER(_MemCounters),
                                           ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

    counters = _MemCounters()
    counters.cb = ctypes.sizeof(_MemCounters)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                    ctypes.byref(counters), counters.cb)
    return (counters.PeakWorkingSetSize / MB) if ok else float("nan")


def raw_files():
    return sorted(glob.glob(os.path.join(RAW, "sms-call-internet-mi-*.txt")))


def expected_raw_names():
    """Return the 62 daily filenames the dataset is expected to contain."""
    days = pd.date_range("2013-11-01", "2014-01-01", freq="D")
    return ["sms-call-internet-mi-%s.txt" % d.strftime("%Y-%m-%d")
            for d in days]


def check_raw_complete():
    """Stop before writing anything unless all 62 daily files are present."""
    expected = expected_raw_names()
    present = {os.path.basename(path) for path in raw_files()}

    missing = [name for name in expected if name not in present]
    unexpected = sorted(present - set(expected))

    if missing or unexpected:
        lines = ["Raw telecom dataset is not complete, so no processed file "
                 "was written.",
                 "  expected : %d files (%s .. %s)"
                 % (len(expected), expected[0], expected[-1]),
                 "  present  : %d matching files" % (len(present) - len(unexpected))]
        if missing:
            shown = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else " (+%d more)" % (len(missing) - 5)
            lines.append("  missing  : %d -> %s%s" % (len(missing), shown, more))
        if unexpected:
            shown = ", ".join(unexpected[:5])
            more = ("" if len(unexpected) <= 5
                    else " (+%d more)" % (len(unexpected) - 5))
            lines.append("  unexpected: %d -> %s%s"
                         % (len(unexpected), shown, more))
        lines.append("Run scripts/download_data.py to complete the download.")
        raise SystemExit("\n".join(lines))

    print("raw input verified: all %d expected daily files present (%s .. %s)"
          % (len(expected), expected[0], expected[-1]))


def aggregate(chunk_size, verbose=True):
    """Sum the Internet traffic of every square across all daily files."""
    files = raw_files()
    totals = np.zeros(MAX_SQUARE + 1, dtype=np.float64)

    stats = {
        "files": len(files),
        "rows": 0,
        "chunks": 0,
        "max_chunk_mb": 0.0,
        "blank_internet": 0,
        "out_of_range": 0,
        "per_file": [],
        "seconds": 0.0,
    }

    started = time.time()
    for index, path in enumerate(files, 1):
        file_started = time.time()
        file_rows = 0
        file_total = 0.0

        reader = pd.read_csv(path, sep="\t", header=None, names=COLS,
                             usecols=TOTALS_USE, dtype=TOTALS_DTYPES,
                             chunksize=chunk_size)
        for chunk in reader:
            stats["chunks"] += 1
            used = chunk.memory_usage(deep=True).sum() / MB
            if used > stats["max_chunk_mb"]:
                stats["max_chunk_mb"] = used

            ids = chunk["square_id"].to_numpy()
            net = chunk["internet"].to_numpy()

            # Square ids are checked before they are used as NumPy indices.
            if ids.size:
                lo, hi = int(ids.min()), int(ids.max())
                if lo < 1 or hi > MAX_SQUARE:
                    stats["out_of_range"] += 1
                    raise SystemExit(
                        "square_id outside the expected range 1..%d in %s "
                        "(found %d..%d); refusing to aggregate."
                        % (MAX_SQUARE, os.path.basename(path), lo, hi))

            blank = np.isnan(net)
            stats["blank_internet"] += int(blank.sum())

            # A blank Internet value contributes nothing to the sum.
            weights = np.where(blank, 0.0, net).astype(np.float64)

            # bincount adds up in float64, so no float32 drift occurs.
            totals += np.bincount(ids.astype(np.intp), weights=weights,
                                  minlength=MAX_SQUARE + 1)

            file_rows += len(chunk)
            file_total += float(weights.sum())
            # The current chunk is released before the next one is loaded.
            del chunk, ids, net, blank, weights

        stats["rows"] += file_rows
        stats["per_file"].append({
            "file": os.path.basename(path),
            "rows": file_rows,
            "total_internet": file_total,
            "seconds": time.time() - file_started,
        })
        if verbose:
            print("[%2d/%d] %s  rows=%d  day_total=%.3f  %.1fs"
                  % (index, len(files), os.path.basename(path), file_rows,
                     file_total, time.time() - file_started), flush=True)

    stats["seconds"] = time.time() - started
    stats["peak_working_set_mb"] = peak_working_set_mb()
    return totals, stats


def run_totals(chunk_size, verbose=True):
    print("PHASE 1: total Internet traffic per square")
    print("  source     : %s" % RAW)
    print("  chunk size : %d rows" % chunk_size)
    print("  columns    : %s" % ", ".join(TOTALS_USE))
    print("  dtypes     : %s" % TOTALS_DTYPES)
    print()

    totals, stats = aggregate(chunk_size, verbose=verbose)

    frame = pd.DataFrame({
        "square_id": np.arange(1, MAX_SQUARE + 1, dtype=np.int32),
        "total_internet_traffic": totals[1:],
    })
    os.makedirs(PROCESSED, exist_ok=True)
    frame.to_csv(TOTALS_OUT, index=False)

    print()
    print("=" * 66)
    print("AGGREGATION COMPLETE")
    print("=" * 66)
    print("files processed         : %d" % stats["files"])
    print("rows processed          : %d" % stats["rows"])
    print("chunks processed        : %d" % stats["chunks"])
    print("chunk size              : %d rows" % chunk_size)
    print("max chunk memory        : %.2f MB" % stats["max_chunk_mb"])
    print("peak process working set: %.1f MB" % stats["peak_working_set_mb"])
    print("blank internet fields   : %d (%.2f%% of rows)"
          % (stats["blank_internet"],
             100.0 * stats["blank_internet"] / max(stats["rows"], 1)))
    print("square_ids out of range : %d chunks" % stats["out_of_range"])
    print("total processing time   : %.1f s (%.1f min)"
          % (stats["seconds"], stats["seconds"] / 60.0))
    print("grand total internet    : %.3f" % totals[1:].sum())
    print("output                  : %s" % TOTALS_OUT)

    busiest = list(frame.nlargest(3, "total_internet_traffic")["square_id"])
    print()
    print("three busiest areas     : %s" % busiest)
    if busiest != TOP_AREAS:
        print("WARNING: the busiest areas differ from the recorded %s"
              % TOP_AREAS)
    return 0


def expected_grid():
    """Build the expected ten-minute grid in local time and in milliseconds."""
    local = pd.date_range(PERIOD_START, PERIOD_END, freq="10min", tz=TZ)
    ms = local.tz_convert("UTC").as_unit("ms").astype("int64").to_numpy()
    assert ms[1] - ms[0] == 600000, "grid step is not 10 minutes in ms"
    return local, ms


def run_series(chunk_size):
    started = time.time()

    local_index, ms_index = expected_grid()
    n_slots = len(ms_index)
    print("PHASE 2: 10-minute series for the selected areas")
    print("observation window : %s .. %s (%s)"
          % (local_index[0], local_index[-1], TZ))
    print("expected slots     : %d  (62 days x 144)" % n_slots)
    print("squares            : %s" % SQUARES)
    print("chunk size         : %d rows" % chunk_size)
    print()

    row_of = {s: i for i, s in enumerate(SQUARES)}

    # Lookup tables are used because they are cheaper than np.isin per chunk.
    keep_lut = np.zeros(10001, dtype=bool)
    row_lut = np.zeros(10001, dtype=np.intp)
    for square, position in row_of.items():
        keep_lut[square] = True
        row_lut[square] = position

    sums = np.zeros((len(SQUARES), n_slots), dtype=np.float64)
    counts = np.zeros((len(SQUARES), n_slots), dtype=np.int32)
    blanks = np.zeros(len(SQUARES), dtype=np.int64)

    files = raw_files()
    kept_rows = 0
    scanned_rows = 0
    unmapped = 0

    for index, path in enumerate(files, 1):
        reader = pd.read_csv(path, sep="\t", header=None, names=COLS,
                             usecols=SERIES_USE, dtype=SERIES_DTYPES,
                             chunksize=chunk_size)
        for chunk in reader:
            scanned_rows += len(chunk)

            ids = chunk["square_id"].to_numpy()

            # Square ids are checked before they are used as lookup indices.
            if ids.size:
                lo, hi = int(ids.min()), int(ids.max())
                if lo < 1 or hi > MAX_SQUARE:
                    raise SystemExit(
                        "square_id outside the expected range 1..%d in %s "
                        "(found %d..%d); refusing to build the series file."
                        % (MAX_SQUARE, os.path.basename(path), lo, hi))

            keep = keep_lut[ids]
            if keep.any():
                sub_ids = ids[keep]
                sub_ts = chunk["time_interval"].to_numpy()[keep]
                sub_net = chunk["internet"].to_numpy()[keep]

                rows = row_lut[sub_ids]

                # Each timestamp is matched to its slot on the sorted grid.
                pos = np.searchsorted(ms_index, sub_ts)
                pos_clipped = np.minimum(pos, n_slots - 1)
                good = ms_index[pos_clipped] == sub_ts
                cols = pos_clipped
                unmapped += int((~good).sum())

                blank = np.isnan(sub_net)
                # A blank value adds nothing, but the slot is observed.
                weights = np.where(blank, 0.0, sub_net).astype(np.float64)

                np.add.at(sums, (rows[good], cols[good]), weights[good])
                np.add.at(counts, (rows[good], cols[good]), 1)
                np.add.at(blanks, rows[good][blank[good]], 1)

                kept_rows += int(good.sum())

            del chunk
        print("[%2d/%d] %s" % (index, len(files), os.path.basename(path)),
              flush=True)

    elapsed = time.time() - started

    # Observed slots keep their sum and slots with no row stay missing.
    values = np.where(counts > 0, sums, np.nan)

    frame = pd.DataFrame({
        "square_id": np.repeat(SQUARES, n_slots),
        "timestamp": np.tile(ms_index, len(SQUARES)),
        "datetime": np.tile(local_index.strftime("%Y-%m-%d %H:%M:%S%z"),
                            len(SQUARES)),
        "internet_traffic": values.reshape(-1),
    })
    frame = frame.sort_values(["square_id", "timestamp"]).reset_index(drop=True)

    # This is checked before writing, or the series file could hide gaps.
    incomplete = [square for square in SQUARES
                  if int((counts[row_of[square]] > 0).sum()) != n_slots]
    if incomplete:
        raise SystemExit(
            "Selected areas %s do not cover all %d expected time slots, so "
            "%s was not written. The raw input is incomplete."
            % (incomplete, n_slots, SERIES_OUT))
    print("all %d selected areas cover the full %d-slot grid"
          % (len(SQUARES), n_slots))

    os.makedirs(PROCESSED, exist_ok=True)
    frame.to_csv(SERIES_OUT, index=False)

    print()
    print("=" * 70)
    print("COMPLETENESS PER AREA")
    print("=" * 70)
    print("%-10s %10s %10s %10s %10s %12s"
          % ("square", "expected", "observed", "missing", "% complete",
             "blank rows"))
    for square in SQUARES:
        i = row_of[square]
        observed = int((counts[i] > 0).sum())
        missing = n_slots - observed
        print("%-10d %10d %10d %10d %9.2f%% %12d"
              % (square, n_slots, observed, missing,
                 100.0 * observed / n_slots, blanks[i]))

    print()
    print("rows scanned            : %d" % scanned_rows)
    print("rows kept (5 squares)   : %d (%.4f%%)"
          % (kept_rows, 100.0 * kept_rows / scanned_rows))
    print("timestamps outside grid : %d" % unmapped)
    print("output rows             : %d" % len(frame))
    print("output                  : %s" % SERIES_OUT)
    print("processing time         : %.1f s (%.1f min)"
          % (elapsed, elapsed / 60.0))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the processed data files from the raw dataset.")
    parser.add_argument("--phase", choices=["totals", "series", "all"],
                        default="all")
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not raw_files():
        raise SystemExit(
            "No raw files found in %s\n"
            "Run scripts/download_data.py first." % RAW)

    # Processed data is never built from an incomplete download.
    check_raw_complete()

    if args.phase in ("totals", "all"):
        run_totals(args.chunk_size, verbose=not args.quiet)
        print()
    if args.phase in ("series", "all"):
        run_series(args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
