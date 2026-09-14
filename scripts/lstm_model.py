"""Train and evaluate the LSTM using the frozen chronological split."""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
SERIES_CSV = os.path.join(PROC, "selected_area_timeseries.csv")
LOG_CSV = os.path.join(RESULTS, "experiment_log.csv")
CURVE_JSON = os.path.join(RESULTS, "lstm_learning_curves.json")

# The CLI refuses to log these experiment ids under any other settings.
SELECTED_SEQ_LEN = 36
SELECTED_HIDDEN = 64

AREAS = [5161, 5059, 5259]
TZ = "Europe/Rome"
TRAIN = ("2013-11-01 00:00", "2013-12-08 23:50")
VALID = ("2013-12-09 00:00", "2013-12-15 23:50")
TEST = ("2013-12-16 00:00", "2013-12-22 23:50")

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_series():
    frame = pd.read_csv(SERIES_CSV)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True
                                       ).dt.tz_convert(TZ)
    out = {}
    for area in AREAS:
        part = frame[frame["square_id"] == area].sort_values("timestamp")
        out[area] = (pd.DatetimeIndex(part["datetime"]),
                     part["internet_traffic"].to_numpy(dtype=np.float64))
    return out


def window_bounds(index, start, end):
    lo = pd.Timestamp(start, tz=TZ)
    hi = pd.Timestamp(end, tz=TZ)
    pos = np.where((index >= lo) & (index <= hi))[0]
    return int(pos[0]), int(pos[-1])


def metrics(actual, predicted):
    error = predicted - actual
    return (float(np.mean(np.abs(error))),
            float(np.sqrt(np.mean(error ** 2))),
            float(np.mean(np.abs(error / actual)) * 100.0))


def make_sequences(scaled, targets, length):
    """Build the input window for each target from the values before it."""
    targets = np.asarray(targets, dtype=np.intp)
    offsets = np.arange(-length, 0)
    rows = targets[:, None] + offsets[None, :]
    if rows.min() < 0:
        raise ValueError("insufficient history for sequence length")
    X = scaled[rows][:, :, None]
    y = scaled[targets]
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


class LSTMForecaster(nn.Module):
    """This model reads the sequence with one LSTM layer and a linear head."""

    def __init__(self, hidden_units, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_units,
                            num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden_units, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(values, bounds, length, hidden, verbose=False,
              evaluate_test=False):
    """Train the LSTM for one area and return its results."""
    (train_lo, train_hi), (valid_lo, valid_hi), (test_lo, test_hi) = bounds

    # The scaler is fitted on training values only so nothing leaks back.
    scaler = StandardScaler()
    scaler.fit(values[train_lo:train_hi + 1].reshape(-1, 1))
    # Scaling stops where this run may look, so tuning never sees the test set.
    scale_hi = test_hi if evaluate_test else valid_hi
    scaled = scaler.transform(values[:scale_hi + 1].reshape(-1, 1)).ravel()

    train_targets = np.arange(train_lo + length, train_hi + 1)
    valid_targets = np.arange(valid_lo, valid_hi + 1)

    X_train, y_train = make_sequences(scaled, train_targets, length)
    X_valid, y_valid = make_sequences(scaled, valid_targets, length)

    set_seed()
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=generator)

    model = LSTMForecaster(hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    train_curve, valid_curve = [], []
    since_improved = 0

    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(batch_x)
        train_loss = running / len(X_train)

        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid), y_valid).item()

        train_curve.append(train_loss)
        valid_curve.append(valid_loss)

        if valid_loss < best_loss - 1e-9:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print("      epoch %3d  train %.6f  valid %.6f"
                  % (epoch, train_loss, valid_loss), flush=True)

        if since_improved >= PATIENCE:
            break

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()

    def evaluate(X, targets):
        started_inf = time.perf_counter()
        with torch.no_grad():
            scaled_pred = model(X).numpy()
        elapsed = time.perf_counter() - started_inf
        predicted = scaler.inverse_transform(
            scaled_pred.reshape(-1, 1)).ravel()
        return predicted, metrics(values[targets], predicted), elapsed

    _, valid_metrics, valid_infer = evaluate(X_valid, valid_targets)

    result = {
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "parameters": model.n_parameters(),
        "train_samples": len(X_train),
        "train_seconds": train_seconds,
        "valid_metrics": valid_metrics,
        "valid_infer_seconds": valid_infer,
        "train_curve": train_curve,
        "valid_curve": valid_curve,
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0]),
    }

    # The test window is used only when the caller asks for it.
    if evaluate_test:
        test_targets = np.arange(test_lo, test_hi + 1)
        X_test, _ = make_sequences(scaled, test_targets, length)
        test_pred, test_metrics, test_infer = evaluate(X_test, test_targets)
        result["test_metrics"] = test_metrics
        result["test_infer_seconds"] = test_infer
        result["test_predictions"] = test_pred

    return result


def update_log(new_rows):
    """Merge this run's rows into the log, replacing its own earlier rows."""
    fresh = pd.DataFrame(new_rows)
    fresh.insert(0, "run_timestamp",
                 pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S%z"))

    if os.path.exists(LOG_CSV):
        existing = pd.read_csv(LOG_CSV)
        keys = set(zip(fresh["experiment_id"], fresh["area"].astype(str)))
        keep = [k not in keys for k in
                zip(existing["experiment_id"], existing["area"].astype(str))]
        replaced = len(existing) - sum(keep)
        merged = pd.concat([existing[keep], fresh], ignore_index=True)
    else:
        replaced = 0
        merged = fresh

    merged.to_csv(LOG_CSV, index=False)
    return len(fresh), replaced, len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seqlen", "hidden", "final"],
                        required=True)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    args = parser.parse_args()

    # This check runs before the data is loaded so a wrong run stops early.
    if args.stage == "hidden":
        if args.length is None:
            parser.error("--stage hidden requires --length")
        if args.length != SELECTED_SEQ_LEN:
            parser.error(
                "--stage hidden writes LSTM-B1 and LSTM-B2, which are recorded "
                "at sequence length %d; --length %d would overwrite them with "
                "a different configuration"
                % (SELECTED_SEQ_LEN, args.length))
    if args.stage == "final":
        absent = [flag for flag, value in (("--length", args.length),
                                           ("--hidden", args.hidden))
                  if value is None]
        if absent:
            parser.error("--stage final requires %s" % " and ".join(absent))
        wrong = [("--length", args.length, SELECTED_SEQ_LEN),
                 ("--hidden", args.hidden, SELECTED_HIDDEN)]
        bad = ["%s %d (frozen value is %d)" % (f, got, want)
               for f, got, want in wrong if got != want]
        if bad:
            parser.error(
                "--stage final writes LSTM-FINAL, the frozen model: %s"
                % "; ".join(bad))

    data = load_series()
    index = data[AREAS[0]][0]
    bounds = (window_bounds(index, *TRAIN),
              window_bounds(index, *VALID),
              window_bounds(index, *TEST))

    rows, curves = [], {}
    print("torch %s | threads %d | seed %d"
          % (torch.__version__, torch.get_num_threads(), SEED))
    print("periods: train %s..%s | valid %s..%s | test %s..%s"
          % (TRAIN + VALID + TEST))

    if args.stage == "seqlen":
        rule("EXPERIMENT LSTM-A: SEQUENCE LENGTH (everything else fixed)")
        print("fixed: 1 layer, 32 hidden, lr=1e-3, batch=64, MSE, "
              "max 100 epochs, patience 10, seed 42")
        configs = [("LSTM-A1", 36), ("LSTM-A2", 72), ("LSTM-A3", 144)]
        hidden = 32
    elif args.stage == "hidden":
        rule("EXPERIMENT LSTM-B: HIDDEN UNITS (sequence length fixed at %d)"
             % args.length)
        configs = [("LSTM-B1", args.length), ("LSTM-B2", args.length)]
        hidden = None
    else:
        rule("FINAL FROZEN LSTM -> TEST (16-22 Dec)")
        configs = [("LSTM-FINAL", args.length)]
        hidden = args.hidden

    for position, (exp_id, length) in enumerate(configs):
        units = hidden if hidden is not None else (32 if position == 0 else 64)
        print()
        print("--- %s  seq_len=%d  hidden=%d ---" % (exp_id, length, units))
        # Only the frozen final stage evaluates the test window.
        is_final = args.stage == "final"
        area_valid, area_test = [], []
        for area in AREAS:
            _, values = data[area]
            result = train_one(values, bounds, length, units,
                               verbose=(is_final and area == 5161),
                               evaluate_test=is_final)
            v_mae, v_rmse, v_mape = result["valid_metrics"]
            area_valid.append(result["valid_metrics"])

            if is_final:
                t_mae, t_rmse, t_mape = result["test_metrics"]
                area_test.append(result["test_metrics"])
                print("  area %-6d TEST  MAE %8.3f  RMSE %8.3f  MAPE %6.3f"
                      % (area, t_mae, t_rmse, t_mape))
            print("  area %-6d VAL   MAE %8.3f  RMSE %8.3f  MAPE %6.3f  "
                  "| epochs %3d (best %3d) | params %5d | train %.1fs"
                  % (area, v_mae, v_rmse, v_mape, result["epochs_run"],
                     result["best_epoch"], result["parameters"],
                     result["train_seconds"]), flush=True)

            if is_final and area == 5161:
                curves["5161"] = {"train": result["train_curve"],
                                  "valid": result["valid_curve"],
                                  "best_epoch": result["best_epoch"],
                                  "seq_len": length, "hidden": units}

            evaluated = "test" if is_final else "validation"
            reported = (result["test_metrics"] if is_final
                        else result["valid_metrics"])
            rows.append({
                "experiment_id": exp_id,
                "model": "LSTM (PyTorch)",
                "area": area,
                "history_or_lags": "seq_len=%d, hidden=%d, 1 layer" % (length, units),
                "train_period": "%s..%s" % TRAIN,
                "validation_period": "%s..%s" % VALID,
                "test_period": ("%s..%s" % TEST) if is_final
                               else "not evaluated at this step",
                "evaluated_on": evaluated,
                "parameters": result["parameters"],
                "MAE": reported[0], "RMSE": reported[1], "MAPE": reported[2],
                "training_time_seconds": result["train_seconds"],
                "inference_time_seconds": (result["test_infer_seconds"]
                                           if is_final
                                           else result["valid_infer_seconds"]),
                "observation": ("epochs=%d best_epoch=%d best_val_loss=%.6f "
                                "train_samples=%d"
                                % (result["epochs_run"], result["best_epoch"],
                                   result["best_valid_loss"],
                                   result["train_samples"])),
                "next_decision": "select on mean validation MAE"
                                 if args.stage != "final"
                                 else "frozen; compare with AR and baselines",
            })

        mean_valid = np.mean([m[0] for m in area_valid])
        print("  MEAN VALIDATION MAE across the three areas: %.3f" % mean_valid)
        if is_final:
            print("  MEAN TEST MAE across the three areas      : %.3f"
                  % np.mean([m[0] for m in area_test]))

    written, replaced, total = update_log(rows)
    print()
    print("experiment log: %d row(s) written, %d replaced, %d total -> %s"
          % (written, replaced, total, LOG_CSV))

    if curves:
        with open(CURVE_JSON, "w", encoding="utf-8") as handle:
            json.dump(curves, handle)
        print("learning curves saved: %s" % CURVE_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
