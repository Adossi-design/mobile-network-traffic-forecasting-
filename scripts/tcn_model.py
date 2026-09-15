"""Train and evaluate the TCN using causal convolutions."""

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
CURVE_JSON = os.path.join(RESULTS, "tcn_learning_curves.json")

# The CLI refuses to log these experiment ids under any other settings.
FINAL_SEQ_LEN = 288
FINAL_CHANNELS = 32
FINAL_DROPOUT = 0.0

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
KERNEL = 3
DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


class Chomp1d(nn.Module):
    """Remove the trailing padding so a convolution stays strictly causal."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size] if self.size > 0 else x


class TemporalBlock(nn.Module):
    """This block runs two dilated causal convolutions with a residual link."""

    def __init__(self, in_channels, out_channels, kernel, dilation,
                 dropout=0.0):
        super().__init__()
        padding = (kernel - 1) * dilation
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Conv1d(out_channels, out_channels, kernel,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNForecaster(nn.Module):
    def __init__(self, channels, kernel=KERNEL, dilations=DILATIONS,
                 dropout=0.0):
        super().__init__()
        self.kernel = kernel
        self.dilations = tuple(dilations)
        self.dropout = dropout
        blocks = []
        in_channels = 1
        for dilation in self.dilations:
            blocks.append(TemporalBlock(in_channels, channels, kernel,
                                        dilation, dropout))
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def features(self, x):
        """Return the temporal output with shape (batch, channels, time)."""
        return self.blocks(x)

    def forward(self, x):
        # Convolutions expect (batch, channels, time), not (batch, time, 1).
        features = self.features(x.transpose(1, 2))
        return self.head(features[:, :, -1]).squeeze(-1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def formula_receptive_field(self):
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)


def structural_receptive_field(channels, probe_length=700):
    """Measure how far back the output is wired to reach."""
    model = TCNForecaster(channels)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1 if parameter.dim() > 1 else 0.0)
    model.eval()

    base = torch.zeros(1, probe_length, 1)
    with torch.no_grad():
        reference = model(base).item()
        for position in range(probe_length):
            probe = base.clone()
            probe[0, position, 0] = 1.0
            if abs(model(probe).item() - reference) > 1e-9:
                return probe_length - position
    return 0


def causality_test(model, length=300, split=150):
    """Check that values after the split cannot change the output before it."""
    model.eval()
    torch.manual_seed(0)
    a = torch.randn(1, length, 1)
    b = a.clone()
    b[0, split + 1:, 0] = torch.randn(length - split - 1) * 50.0

    with torch.no_grad():
        fa = model.features(a.transpose(1, 2))
        fb = model.features(b.transpose(1, 2))

    at_split = float((fa[:, :, split] - fb[:, :, split]).abs().max())
    before = float((fa[:, :, :split + 1] - fb[:, :, :split + 1]).abs().max())
    after = float((fa[:, :, split + 1:] - fb[:, :, split + 1:]).abs().max())
    length_ok = fa.shape[2] == length
    return at_split, before, after, length_ok, fa.shape[2]


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
    targets = np.asarray(targets, dtype=np.intp)
    rows = targets[:, None] + np.arange(-length, 0)[None, :]
    if rows.min() < 0:
        raise ValueError("insufficient history")
    return (torch.tensor(scaled[rows][:, :, None], dtype=torch.float32),
            torch.tensor(scaled[targets], dtype=torch.float32))


def train_one(values, bounds, length, channels, dropout=0.0,
              evaluate_test=False):
    """Train the TCN for one area and return its results."""
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

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=generator)

    model = TCNForecaster(channels, dropout=dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_loss, best_epoch, best_state = float("inf"), 0, None
    train_curve, valid_curve, since = [], [], 0

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
            best_loss, best_epoch = valid_loss, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        if since >= PATIENCE:
            break

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()

    def evaluate(X, targets):
        start_inf = time.perf_counter()
        with torch.no_grad():
            scaled_pred = model(X).numpy()
        elapsed = time.perf_counter() - start_inf
        predicted = scaler.inverse_transform(
            scaled_pred.reshape(-1, 1)).ravel()
        return predicted, metrics(values[targets], predicted), elapsed

    valid_pred, valid_metrics, valid_infer = evaluate(X_valid, valid_targets)

    result = {
        "epochs_run": epoch, "best_epoch": best_epoch,
        "best_valid_loss": best_loss, "parameters": model.n_parameters(),
        "train_samples": len(X_train), "train_seconds": train_seconds,
        "valid_metrics": valid_metrics, "valid_infer_seconds": valid_infer,
        "train_curve": train_curve, "valid_curve": valid_curve,
        "receptive_field": model.formula_receptive_field(),
        "valid_predictions": valid_pred,
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
    parser.add_argument("--stage", required=True,
                        choices=["verify", "seqlen", "channels", "final"])
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    args = parser.parse_args()

    # This check runs before the data is loaded so a wrong run stops early.
    if args.stage == "channels":
        if args.length is None:
            parser.error("--stage channels requires --length")
        if args.length != FINAL_SEQ_LEN:
            parser.error(
                "--stage channels writes TCN-B2, which is recorded at "
                "sequence length %d; --length %d would overwrite it with a "
                "different configuration" % (FINAL_SEQ_LEN, args.length))
    if args.stage == "final":
        absent = [flag for flag, value in (("--length", args.length),
                                           ("--channels", args.channels))
                  if value is None]
        if absent:
            parser.error("--stage final requires %s" % " and ".join(absent))
        wrong = [("--length", args.length, FINAL_SEQ_LEN),
                 ("--channels", args.channels, FINAL_CHANNELS)]
        bad = ["%s %d (frozen value is %d)" % (f, got, want)
               for f, got, want in wrong if got != want]
        if bad:
            parser.error(
                "--stage final writes TCN-FINAL, the frozen model: %s"
                % "; ".join(bad))

    print("torch %s | threads %d | seed %d"
          % (torch.__version__, torch.get_num_threads(), SEED))

    if args.stage == "verify":
        rule("C/D. ARCHITECTURE, RECEPTIVE FIELD AND CAUSALITY VERIFICATION")
        verification_failures = []
        for channels in (16, 32):
            # The seed is set first so the numbers stay the same.
            torch.manual_seed(SEED)
            model = TCNForecaster(channels)
            formula = model.formula_receptive_field()
            measured = structural_receptive_field(channels)
            print()
            print("channels = %d" % channels)
            print("  kernel size            : %d" % KERNEL)
            print("  dilations              : %s" % (list(DILATIONS),))
            print("  residual blocks        : %d (2 convolutions each)"
                  % len(DILATIONS))
            print("  parameters             : %d" % model.n_parameters())
            print("  sum(dilations)         : %d" % sum(DILATIONS))
            print("  formula 1+2*(k-1)*sum(d): %d" % formula)
            print("  structural probe        : %d" % measured)
            print("  structural probe matches formula: %s"
                  % (formula == measured))
            print("  note: %d is the MAXIMUM structural receptive field of"
                  % formula)
            print("        this architecture. The frozen model is fed a %d"
                  % FINAL_SEQ_LEN)
            print("        step input window, so the history actually")
            print("        available to any prediction is %d observations."
                  % FINAL_SEQ_LEN)

            at_split, before, after, length_ok, out_len = causality_test(model)
            print("  causality test (sequences identical up to position 150,")
            print("                  different afterwards):")
            print("    max |difference| at position 150      : %.3e" % at_split)
            print("    max |difference| at positions <= 150  : %.3e" % before)
            print("    max |difference| at positions > 150   : %.3e" % after)
            print("    representation unchanged up to 150    : %s"
                  % (before < 1e-6))
            print("    representation DID change after 150   : %s"
                  % (after > 1e-6))
            print("    input length == output temporal length: %s (%d)"
                  % (length_ok, out_len))

            for label, ok in [
                    ("structural probe equals the formula", formula == measured),
                    ("representation unchanged up to the split", before < 1e-6),
                    ("representation changed after the split", after > 1e-6),
                    ("input length equals output temporal length", length_ok)]:
                if not ok:
                    verification_failures.append("%d channels: %s"
                                                 % (channels, label))

        if verification_failures:
            raise SystemExit("Architecture verification FAILED: %s"
                             % "; ".join(verification_failures))
        print()
        print("all architecture and causality checks passed")
        return 0

    data = load_series()
    index = data[AREAS[0]][0]
    bounds = (window_bounds(index, *TRAIN), window_bounds(index, *VALID),
              window_bounds(index, *TEST))

    if args.stage == "seqlen":
        rule("EXPERIMENT TCN-A: SEQUENCE LENGTH (everything else fixed)")
        configs = [("TCN-A1", 36, 16), ("TCN-A2", 144, 16),
                   ("TCN-A3", 288, 16)]
    elif args.stage == "channels":
        rule("EXPERIMENT TCN-B: CHANNEL WIDTH (seq_len fixed at %d)"
             % args.length)
        # The 16-channel arm is reused because it is exactly TCN-A3.
        configs = [("TCN-B2", args.length, 32)]
    else:
        rule("FINAL FROZEN TCN -> TEST (16-22 Dec)")
        configs = [("TCN-FINAL", args.length, args.channels)]

    rows, curves = [], {}
    is_final = args.stage == "final"
    for exp_id, length, channels in configs:
        print()
        print("--- %s  seq_len=%d  channels=%d ---" % (exp_id, length, channels))
        valid_maes, test_maes = [], []
        for area in AREAS:
            _, values = data[area]
            # Only the final stage looks at the test window.
            result = train_one(values, bounds, length, channels,
                               dropout=FINAL_DROPOUT,
                               evaluate_test=is_final)
            v_mae, v_rmse, v_mape = result["valid_metrics"]
            valid_maes.append(v_mae)

            if is_final:
                t_mae, t_rmse, t_mape = result["test_metrics"]
                test_maes.append(t_mae)
                print("  area %-6d TEST  MAE %8.3f  RMSE %8.3f  MAPE %6.3f"
                      % (area, t_mae, t_rmse, t_mape))
            print("  area %-6d VAL   MAE %8.3f  RMSE %8.3f  MAPE %6.3f  "
                  "| epochs %3d (best %3d) | params %6d | train %.1fs"
                  % (area, v_mae, v_rmse, v_mape, result["epochs_run"],
                     result["best_epoch"], result["parameters"],
                     result["train_seconds"]), flush=True)

            if area == 5161 and is_final:
                curves["5161"] = {"train": result["train_curve"],
                                  "valid": result["valid_curve"],
                                  "best_epoch": result["best_epoch"],
                                  "seq_len": length, "channels": channels}

            evaluated = "test" if is_final else "validation"
            reported = (result["test_metrics"] if is_final
                        else result["valid_metrics"])
            rows.append({
                "experiment_id": exp_id,
                "model": "TCN (PyTorch)",
                "area": area,
                "history_or_lags": ("seq_len=%d, channels=%d, kernel=%d, "
                                    "dilations=%s, receptive_field=%d, "
                                    "dropout=%.2f"
                                    % (length, channels, KERNEL,
                                       "-".join(str(d) for d in DILATIONS),
                                       result["receptive_field"],
                                       FINAL_DROPOUT)),
                "train_period": "%s..%s" % TRAIN,
                "validation_period": "%s..%s" % VALID,
                "test_period": ("%s..%s" % TEST) if is_final
                               else "not evaluated at this step",
                "evaluated_on": evaluated,
                "parameters": result["parameters"],
                "MAE": reported[0], "RMSE": reported[1],
                "MAPE": reported[2],
                "training_time_seconds": result["train_seconds"],
                "inference_time_seconds": (
                    result["test_infer_seconds"] if is_final
                    else result["valid_infer_seconds"]),
                "observation": ("epochs=%d best_epoch=%d "
                                "best_val_loss=%.6f train_samples=%d"
                                % (result["epochs_run"],
                                   result["best_epoch"],
                                   result["best_valid_loss"],
                                   result["train_samples"])),
                "next_decision": ("frozen; provisional comparison"
                                  if is_final
                                  else "select on mean validation MAE"),
            })

        print("  MEAN VALIDATION MAE across the three areas: %.3f"
              % np.mean(valid_maes))
        if is_final:
            print("  MEAN TEST MAE across the three areas      : %.3f"
                  % np.mean(test_maes))

    if rows:
        written, replaced, total = update_log(rows)
        print()
        print("experiment log: %d row(s) written, %d replaced, %d total -> %s"
              % (written, replaced, total, LOG_CSV))

    if curves and is_final:
        with open(CURVE_JSON, "w", encoding="utf-8") as handle:
            json.dump(curves, handle)
        print("learning curves saved: %s" % CURVE_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
