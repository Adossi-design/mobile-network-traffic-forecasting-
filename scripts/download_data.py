"""Download and check the Milan telecom files from Harvard Dataverse."""

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit(
        "The 'requests' package is required.\n"
        "Install it with:\n\n    py -m pip install requests\n"
    )

BASE = "https://dataverse.harvard.edu"
# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The DOI is fixed because the guestbook handling is written for it.
DOI = "doi:10.7910/DVN/EGZHFV"
DEST_DIR = os.path.join(ROOT, "data", "raw", "telecom")

# The values come from the environment, so no personal detail is stored.
GUESTBOOK_RESPONSE = {
    "name": os.environ.get("DATAVERSE_NAME", ""),
    "email": os.environ.get("DATAVERSE_EMAIL", ""),
    "institution": os.environ.get("DATAVERSE_INSTITUTION", ""),
    "position": os.environ.get("DATAVERSE_POSITION", ""),
}

CHUNK_SIZE = 1024 * 1024
# A file is abandoned after this many attempts with no progress.
STALLED_LIMIT = 5
MAX_ATTEMPTS = 40
# The two values are the connect timeout and the read timeout in seconds.
TIMEOUT = (30, 180)
# Parallel mode prints a progress line at most this often, in seconds.
PARALLEL_REPORT_EVERY = 30.0

# A browser user agent is sent because Dataverse blocks the default one.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_print_lock = threading.Lock()
_thread_local = threading.local()


def emit(message):
    """Print a whole line at once so that workers do not interleave."""
    with _print_lock:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()


def human(n):
    """Format a byte count for humans."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return "%.2f %s" % (n, unit)
        n /= step
    return "%.2f PB" % n


def clock(seconds):
    """Format a duration as H:MM:SS."""
    seconds = int(seconds)
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60,
                             seconds % 60)


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def thread_session():
    """Return one requests session per thread, as sessions are not shared."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = make_session()
        _thread_local.session = session
    return session


def list_files(session):
    """Ask the Dataverse API for every file in the dataset."""
    url = BASE + "/api/datasets/:persistentId/"
    response = session.get(url, params={"persistentId": DOI}, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()["data"]["latestVersion"]["files"]

    files = []
    for entry in payload:
        data_file = entry["dataFile"]
        checksum = data_file.get("checksum") or {}
        files.append({
            "id": data_file["id"],
            # Keep the name exactly as Dataverse reports it.
            "name": data_file.get("filename") or entry.get("label"),
            "size": data_file.get("filesize"),
            "md5": (checksum.get("value") if checksum.get("type") == "MD5"
                    else data_file.get("md5")),
        })
    files.sort(key=lambda f: f["name"])
    return files


def signed_url_for(session, file_id):
    """Submit the guestbook response and return the signed download URL."""
    url = BASE + "/api/access/datafile/%d" % file_id
    response = session.post(url, json={"guestbookResponse": GUESTBOOK_RESPONSE},
                            timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    signed = (payload.get("data") or {}).get("signedUrl")
    if not signed:
        raise IOError("Dataverse returned no signed URL: %s"
                      % str(payload)[:200])
    return signed


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def already_complete(path, expected_size):
    """Return True when the file exists and matches the expected size."""
    if not os.path.exists(path):
        return False
    if expected_size is None:
        # Dataverse reported no size, so an existing file counts as complete.
        return True
    return os.path.getsize(path) == expected_size


def part_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def stream_to_disk(session, file_id, part_path, expected_size, stats,
                   inline_progress, label):
    """Stream one file to disk and resume from any partial download."""
    resume_from = part_size(part_path)

    # A .part at or beyond the expected size is stale, so it starts over.
    if expected_size is not None and resume_from >= expected_size:
        os.remove(part_path)
        resume_from = 0

    headers = {}
    if resume_from:
        headers["Range"] = "bytes=%d-" % resume_from

    url = signed_url_for(session, file_id)

    with session.get(url, headers=headers, stream=True,
                     timeout=TIMEOUT) as response:
        stats["status_codes"].append(response.status_code)
        response.raise_for_status()

        if resume_from and response.status_code == 206:
            mode = "ab"  # The server honoured the resume request.
        else:
            mode = "wb"  # The server ignored the range and restarted the file.
            resume_from = 0

        downloaded = resume_from
        started_at = time.time()
        started_bytes = downloaded
        last_report = started_at

        with open(part_path, mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                elapsed = now - started_at
                rate = (downloaded - started_bytes) / elapsed if elapsed else 0

                if inline_progress and now - last_report >= 1.0:
                    last_report = now
                    if expected_size:
                        pct = 100.0 * downloaded / expected_size
                        eta = ((expected_size - downloaded) / rate) if rate else 0
                        line = "    %s / %s (%.1f%%)  %.2f MB/s  ETA %s" % (
                            human(downloaded), human(expected_size), pct,
                            rate / 1048576.0, clock(eta))
                    else:
                        line = "    %s  %.2f MB/s" % (human(downloaded),
                                                      rate / 1048576.0)
                    sys.stdout.write("\r" + line.ljust(72))
                    sys.stdout.flush()
                elif not inline_progress and now - last_report >= PARALLEL_REPORT_EVERY:
                    last_report = now
                    pct = (100.0 * downloaded / expected_size) if expected_size else 0
                    emit("    %-38s %6.1f%%  %s  %.2f MB/s"
                         % (label, pct, human(downloaded), rate / 1048576.0))

    if inline_progress:
        sys.stdout.write("\r" + " " * 74 + "\r")
        sys.stdout.flush()

    # This counts only the time spent moving bytes, used for the speed table.
    stats["transfer_seconds"] += time.time() - started_at
    stats["transfer_bytes"] += downloaded - started_bytes
    return downloaded


def download_one(session, meta, verify_md5, inline_progress=True):
    """Download one file and return its status, message and statistics."""
    name = meta["name"]
    final_path = os.path.join(DEST_DIR, name)
    part_path = final_path + ".part"
    expected_size = meta["size"]

    stats = {
        "name": name,
        "expected_size": expected_size,
        "transfer_bytes": 0,
        "transfer_seconds": 0.0,
        "wall_seconds": 0.0,
        "attempts": 0,
        "errors": [],
        "status_codes": [],
        "size_ok": None,
        "md5_ok": None,
    }
    wall_start = time.time()

    if already_complete(final_path, expected_size):
        # A matching size is not proof, so the file is hashed as well.
        if verify_md5 and meta["md5"]:
            actual = md5_of(final_path)
            if actual.lower() == meta["md5"].lower():
                stats["size_ok"] = True
                stats["md5_ok"] = True
                stats["wall_seconds"] = time.time() - wall_start
                return ("skipped",
                        "already present, MD5 verified (%s)"
                        % human(os.path.getsize(final_path)), stats)
            stats["md5_ok"] = False
            stats["errors"].append(
                "existing file failed MD5: got %s, expected %s"
                % (actual, meta["md5"]))
            os.remove(final_path)
        else:
            stats["size_ok"] = True
            stats["wall_seconds"] = time.time() - wall_start
            return ("skipped",
                    "already present (%s)" % human(os.path.getsize(final_path)),
                    stats)

    # A file with the wrong size is moved aside so its bytes can be reused.
    if os.path.exists(final_path):
        os.replace(final_path, part_path)

    last_error = None
    stalled = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        stats["attempts"] = attempt
        before = part_size(part_path)
        try:
            downloaded = stream_to_disk(session, meta["id"], part_path,
                                        expected_size, stats,
                                        inline_progress, name)

            if expected_size is not None and downloaded != expected_size:
                raise IOError("incomplete: got %d of %d bytes"
                              % (downloaded, expected_size))

            if verify_md5 and meta["md5"]:
                actual = md5_of(part_path)
                if actual.lower() == meta["md5"].lower():
                    stats["md5_ok"] = True
                else:
                    os.remove(part_path)
                    raise IOError("MD5 mismatch: got %s, expected %s"
                                  % (actual, meta["md5"]))

            # The file is renamed only once it is known to be complete.
            os.replace(part_path, final_path)
            stats["size_ok"] = (expected_size is None
                                or os.path.getsize(final_path) == expected_size)
            stats["wall_seconds"] = time.time() - wall_start
            return "downloaded", human(downloaded), stats

        except Exception as exc:
            last_error = exc
            stats["errors"].append(str(exc))
            after = part_size(part_path)

            # Progress means the link expired mid transfer, not a real failure.
            if after > before:
                stalled = 0
                emit("    resuming %s at %s (%s)" % (name, human(after), exc))
            else:
                stalled += 1
                if stalled >= STALLED_LIMIT:
                    break
                wait = 5 * stalled
                emit("    %s attempt %d failed: %s -- retrying in %ds"
                     % (name, attempt, exc, wait))
                time.sleep(wait)

    stats["size_ok"] = False
    stats["wall_seconds"] = time.time() - wall_start
    return "failed", str(last_error), stats


def speed_table(stats_list, wall_seconds, files, workers):
    """Print the per-file and combined throughput figures for a run."""
    moved = [s for s in stats_list if s["transfer_bytes"] > 0]
    if not moved:
        return

    print()
    print("-" * 66)
    print("THROUGHPUT (%d worker%s)" % (workers, "" if workers == 1 else "s"))
    print("-" * 66)
    print("%-38s %10s %8s %9s" % ("file", "size", "secs", "MB/s"))
    for s in moved:
        rate = s["transfer_bytes"] / s["transfer_seconds"] if s["transfer_seconds"] else 0
        print("%-38s %10s %8.1f %9.2f"
              % (s["name"], human(s["expected_size"] or s["transfer_bytes"]),
                 s["transfer_seconds"], rate / 1048576.0))

    total_bytes = sum(s["transfer_bytes"] for s in moved)
    combined = total_bytes / wall_seconds if wall_seconds else 0
    sum_individual = sum(
        (s["transfer_bytes"] / s["transfer_seconds"]) if s["transfer_seconds"] else 0
        for s in moved)

    print("-" * 66)
    print("Files transferred this run:   %d" % len(moved))
    print("Bytes transferred this run:   %s (%d bytes)"
          % (human(total_bytes), total_bytes))
    print("Wall-clock for the run:       %s (%.1f s)"
          % (clock(wall_seconds), wall_seconds))
    print("Sum of individual speeds:     %.2f MB/s" % (sum_individual / 1048576.0))
    print("COMBINED AVERAGE SPEED:       %.2f MB/s" % (combined / 1048576.0))

    dataset_bytes = sum(f["size"] or 0 for f in files)
    if combined:
        print()
        print("Projection at this combined speed:")
        print("  whole dataset (%s): %s"
              % (human(dataset_bytes), clock(dataset_bytes / combined)))
        remaining = dataset_bytes - sum(
            os.path.getsize(os.path.join(DEST_DIR, f["name"]))
            for f in files
            if already_complete(os.path.join(DEST_DIR, f["name"]), f["size"]))
        print("  still missing (%s): %s"
              % (human(remaining), clock(remaining / combined)))


def report(files, results, failures, started_at, stats_list, workers):
    downloaded_now = [n for n, s in results.items() if s == "downloaded"]
    skipped = [n for n, s in results.items() if s == "skipped"]

    present, total_bytes = [], 0
    for meta in files:
        path = os.path.join(DEST_DIR, meta["name"])
        if already_complete(path, meta["size"]):
            present.append(meta["name"])
            total_bytes += os.path.getsize(path)
    present.sort()

    wall = time.time() - started_at
    speed_table(stats_list, wall, files, workers)

    print()
    print("=" * 66)
    print("DOWNLOAD REPORT")
    print("=" * 66)
    print("Number of files expected:                %d" % len(files))
    print("Number of files successfully downloaded: %d" % len(present))
    print("  (new this run: %d, already present: %d)"
          % (len(downloaded_now), len(skipped)))
    print("Number of failed files:                  %d" % len(failures))
    print("Total downloaded size:                   %s (%d bytes)"
          % (human(total_bytes), total_bytes))
    print("First filename:                          %s"
          % (present[0] if present else "-"))
    print("Last filename:                           %s"
          % (present[-1] if present else "-"))
    print("Destination folder:                      %s" % DEST_DIR)
    print("Elapsed:                                 %s" % clock(wall))

    checked = [s for s in stats_list if s["transfer_bytes"] > 0]
    if checked:
        passed = sum(1 for s in checked if s["size_ok"])
        print("Size verification:                       %d/%d passed"
              % (passed, len(checked)))

    problems = [(s["name"], s["errors"]) for s in stats_list if s["errors"]]
    print()
    if problems:
        print("HTTP errors / retries observed:")
        for name, errors in problems:
            for err in errors:
                print("  - %s: %s" % (name, err))
    else:
        print("HTTP errors / retries observed: none")

    missing = [f["name"] for f in files if f["name"] not in present]
    if failures:
        print()
        print("FAILED FILES (%d):" % len(failures))
        for name, reason in failures:
            print("  - %s  ->  %s" % (name, reason))
        print()
        print("Re-run this script to retry only the failed files.")
    elif missing:
        print()
        print("Not yet downloaded (%d): re-run without --limit to fetch them."
              % len(missing))
    else:
        print()
        print("All %d files downloaded successfully." % len(files))

    return 1 if failures else 0


def main():
    global DEST_DIR

    parser = argparse.ArgumentParser(
        description="Download the Telecommunications SMS/Call/Internet MI "
                    "dataset from Harvard Dataverse.")
    parser.add_argument("--list-only", action="store_true",
                        help="list the dataset files and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="only consider the first N files of the dataset")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of files to download simultaneously")
    parser.add_argument("--verify-md5", action="store_true",
                        help="verify each downloaded file against its MD5")
    parser.add_argument("--dest", default=None,
                        help="destination folder; a relative path is "
                             "resolved from the repository root "
                             "(default: data/raw/telecom)")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    # A limit of 0 must not start a download of the whole dataset.
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    # A signed URL needs an email address, but --list-only does not.
    if not args.list_only and not GUESTBOOK_RESPONSE["email"]:
        raise SystemExit(
            "DATAVERSE_EMAIL is not set. Harvard Dataverse requires an email "
            "address for the guestbook on this dataset. Set it first, for "
            'example:  DATAVERSE_EMAIL="your_email@example.com"')

    if args.dest:
        DEST_DIR = (args.dest if os.path.isabs(args.dest)
                    else os.path.join(ROOT, args.dest))

    # --list-only returns before anything is created on this machine.
    session = make_session()
    started_at = time.time()

    print("Querying Harvard Dataverse for %s ..." % DOI)
    try:
        files = list_files(session)
    except Exception as exc:
        sys.exit("Could not read the dataset file list: %s" % exc)

    total_size = sum(f["size"] or 0 for f in files)
    print("Dataverse found %d files." % len(files))
    print("Total dataset size: %s" % human(total_size))

    if args.list_only:
        print()
        for i, meta in enumerate(files, 1):
            print("%3d. %-40s %12s" % (i, meta["name"], human(meta["size"] or 0)))
        return 0

    # The run will really download now, so the destination is checked.
    os.makedirs(DEST_DIR, exist_ok=True)
    print("Saving to: %s" % DEST_DIR)
    print("Concurrency: %d simultaneous download%s"
          % (args.workers, "" if args.workers == 1 else "s"))

    free = shutil.disk_usage(DEST_DIR).free
    if total_size and free < total_size:
        print("\nWARNING: only %s free on this drive, dataset needs %s."
              % (human(free), human(total_size)))

    selected = files[:args.limit] if args.limit is not None else files
    results, failures, stats_list = {}, [], []

    # The timer starts here so it measures transfer time and not the API call.
    run_started = time.time()
    total = len(selected)

    def handle(index, meta, status, detail, stats):
        results[meta["name"]] = status
        stats_list.append(stats)
        if status == "skipped":
            emit("[%d/%d] Skipped: %s (%s)" % (index, total, meta["name"], detail))
        elif status == "downloaded":
            rate = (stats["transfer_bytes"] / stats["transfer_seconds"]
                    if stats["transfer_seconds"] else 0)
            emit("[%d/%d] Completed: %s (%s, %.2f MB/s)"
                 % (index, total, meta["name"], detail, rate / 1048576.0))
        else:
            emit("[%d/%d] FAILED: %s -- %s" % (index, total, meta["name"], detail))
            failures.append((meta["name"], detail))

    if args.workers == 1:
        for index, meta in enumerate(selected, 1):
            emit("")
            emit("[%d/%d] Downloading: %s ..." % (index, total, meta["name"]))
            status, detail, stats = download_one(session, meta, args.verify_md5,
                                                 inline_progress=True)
            handle(index, meta, status, detail, stats)
    else:
        def worker(index, meta):
            emit("[%d/%d] Downloading: %s (%s) ..."
                 % (index, total, meta["name"], human(meta["size"] or 0)))
            return download_one(thread_session(), meta, args.verify_md5,
                                inline_progress=False)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, i, m): (i, m)
                       for i, m in enumerate(selected, 1)}
            # Each file is reported as soon as it lands, not in order.
            for future in as_completed(futures):
                index, meta = futures[future]
                status, detail, stats = future.result()
                handle(index, meta, status, detail, stats)

    run_wall = time.time() - run_started
    code = report(files, results, failures, started_at, stats_list, args.workers)
    print("Transfer phase wall-clock:               %s" % clock(run_wall))
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the script to resume where it stopped.")
        sys.exit(130)
