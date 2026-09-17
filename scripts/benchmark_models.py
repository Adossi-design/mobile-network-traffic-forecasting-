"""Measure the training and inference time of the three frozen models."""

import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# The sibling modules live in scripts/, which Python already has on sys.path.
import ar_model
import lstm_model
import tcn_model

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TIMING_CSV = os.path.join(RESULTS, "final_timing.csv")

AREA = 5161
AR_LAGS = (list(range(1, 7)) + [143, 144, 145] + [287, 288, 289]
           + [1007, 1008, 1009])
LSTM_SEQ, LSTM_HIDDEN = 36, 64
TCN_SEQ, TCN_CHANNELS = 288, 32
SEED = 42
TRAIN_REPEATS_NN = 3
TRAIN_REPEATS_AR = 200
INFER_REPEATS = 5


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def summarise(times):
    return (statistics.median(times), min(times), max(times), len(times))


def main():

    rule("ENVIRONMENT")
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for key, value in info.items():
        print("  %-16s %s" % (key, value))

    data = ar_model.load_series()
    index = data[AREA][0]
    values = data[AREA][1]
    train_b = ar_model.window_bounds(index, *ar_model.TRAIN)
    valid_b = ar_model.window_bounds(index, *ar_model.VALID)
    test_b = ar_model.window_bounds(index, *ar_model.TEST)
    test_targets = np.arange(test_b[0], test_b[1] + 1)
    rows = []

    rule("AR TRAINING (frozen Selected-Lag design)")
    ar_train_targets = np.arange(train_b[0] + max(AR_LAGS), train_b[1] + 1)
    ar_fit = ar_model.SelectedLagAR(AR_LAGS)
    # The design matrix is built outside the timed section.
    X_train = ar_fit.design_matrix(values, ar_train_targets)
    y_train = values[ar_train_targets]

    times = []
    for _ in range(TRAIN_REPEATS_AR):
        start = time.perf_counter()
        np.linalg.lstsq(X_train, y_train, rcond=None)
        times.append(time.perf_counter() - start)
    med, lo, hi, n = summarise(times)
    print("  a single AR fit takes only a few milliseconds, so one")
    print("  measurement would be")
    print("  unreliable; %d repetitions are used to obtain a stable "
          "distribution." % n)
    print("  median %.6f s | min %.6f s | max %.6f s | repetitions %d"
          % (med, lo, hi, n))
    rows.append({"model": "AR", "phase": "training", "median_s": med,
                 "min_s": lo, "max_s": hi, "repetitions": n,
                 "note": "np.linalg.lstsq on the prebuilt design matrix; "
                         "many repeats because one fit takes only a few "
                         "milliseconds"})

    ar_fit.fit(values, ar_train_targets)

    rule("AR INFERENCE (1,008 individual one-step calls)")
    # The design matrix is built outside the timed section.
    ar_rows = ar_fit.design_matrix(values, test_targets)
    # Two warm-up passes run before the measured repetitions.
    for _ in range(2):
        for i in range(len(test_targets)):
            float(ar_rows[i] @ ar_fit.coefficients)
    times = []
    for _ in range(INFER_REPEATS):
        start = time.perf_counter()
        for i in range(len(test_targets)):
            float(ar_rows[i] @ ar_fit.coefficients)
        times.append(time.perf_counter() - start)
    med, lo, hi, n = summarise(times)
    print("  median for 1,008 predictions: %.6f s" % med)
    print("  median per prediction       : %.3f us" % (med / 1008 * 1e6))
    print("  repetitions                 : %d" % n)
    rows.append({"model": "AR", "phase": "inference_1008", "median_s": med,
                 "min_s": lo, "max_s": hi, "repetitions": n,
                 "note": "1,008 separate row-vector dot products; design "
                         "matrix built outside the timer"})

    for name, seq_len, builder in (
            ("LSTM", LSTM_SEQ, lambda: lstm_model.LSTMForecaster(LSTM_HIDDEN)),
            ("TCN", TCN_SEQ, lambda: tcn_model.TCNForecaster(TCN_CHANNELS))):
        rule("%s TRAINING (%d complete timed repetitions)"
             % (name, TRAIN_REPEATS_NN))

        # The inputs are prepared before the timed section begins.
        scaler = StandardScaler()
        scaler.fit(values[train_b[0]:train_b[1] + 1].reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).ravel()
        train_targets = np.arange(train_b[0] + seq_len, train_b[1] + 1)
        valid_targets = np.arange(valid_b[0], valid_b[1] + 1)
        maker = lstm_model.make_sequences if name == "LSTM" else tcn_model.make_sequences
        X_tr, y_tr = maker(scaled, train_targets, seq_len)
        X_va, y_va = maker(scaled, valid_targets, seq_len)
        X_te, _ = maker(scaled, test_targets, seq_len)

        times, epochs_seen = [], []
        for _ in range(TRAIN_REPEATS_NN):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64,
                                shuffle=True, generator=generator)

            start = time.perf_counter()
            model = builder()
            optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            best, best_state, since = float("inf"), None, 0
            for epoch in range(1, 101):
                model.train()
                for bx, by in loader:
                    optimiser.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimiser.step()
                model.eval()
                with torch.no_grad():
                    vloss = criterion(model(X_va), y_va).item()
                if vloss < best - 1e-9:
                    best, since = vloss, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    since += 1
                if since >= 10:
                    break
            model.load_state_dict(best_state)
            times.append(time.perf_counter() - start)
            epochs_seen.append(epoch)
            print("    repetition %d: %.2f s (%d epochs)"
                  % (len(times), times[-1], epoch), flush=True)

        med, lo, hi, n = summarise(times)
        print("  median %.2f s | min %.2f s | max %.2f s | repetitions %d"
              % (med, lo, hi, n))
        print("  epochs per repetition: %s" % epochs_seen)
        rows.append({"model": name, "phase": "training", "median_s": med,
                     "min_s": lo, "max_s": hi, "repetitions": n,
                     "note": "model creation + optimiser + training loop + "
                             "early stopping + best-weight restore; tensors "
                             "and loader prepared outside the timer"})

        rule("%s INFERENCE (1,008 individual one-step calls)" % name)
        model.eval()
        with torch.no_grad():
            # Twenty warm-up passes run before inference is measured.
            for i in range(20):
                model(X_te[i:i + 1])
        times = []
        for _ in range(INFER_REPEATS):
            start = time.perf_counter()
            with torch.no_grad():
                for i in range(len(test_targets)):
                    model(X_te[i:i + 1])
            times.append(time.perf_counter() - start)
        med, lo, hi, n = summarise(times)
        print("  median for 1,008 predictions: %.4f s" % med)
        print("  median per prediction       : %.3f ms" % (med / 1008 * 1e3))
        print("  repetitions                 : %d" % n)
        rows.append({"model": name, "phase": "inference_1008",
                     "median_s": med, "min_s": lo, "max_s": hi,
                     "repetitions": n,
                     "note": "1,008 separate batch-size-1 forward passes; "
                             "sequence tensors built outside the timer"})

    frame = pd.DataFrame(rows)
    for key, value in info.items():
        frame[key] = value
    frame["area"] = AREA
    frame.to_csv(TIMING_CSV, index=False)
    print()
    print("wrote %s" % TIMING_CSV)
    return 0


if __name__ == "__main__":
    # --help must exit without running anything that writes a result file.
    parser = argparse.ArgumentParser(
        description="Measure representative training and inference times for "
                    "the three frozen models on Square 5161. Takes no "
                    "options. Runs for roughly 25 minutes and overwrites "
                    "results/final_timing.csv.")
    parser.parse_args()
    raise SystemExit(main())
