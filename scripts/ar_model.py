"""Train and evaluate the selected-lag AR model and the naive baselines."""

import argparse
import os
import time

import numpy as np
import pandas as pd

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
SERIES_CSV = os.path.join(PROC, "selected_area_timeseries.csv")
LOG_CSV = os.path.join(RESULTS, "experiment_log.csv")

AREAS = [5161, 5059, 5259]
TZ = "Europe/Rome"

TRAIN = ("2013-11-01 00:00", "2013-12-08 23:50")
VALID = ("2013-12-09 00:00", "2013-12-15 23:50")
TEST = ("2013-12-16 00:00", "2013-12-22 23:50")

PER_DAY = 144
PER_WEEK = 1008

# The four lag sets are compared against each other on validation.
EXPERIMENTS = [
    ("AR-1", "recent short-term lags (1 hour)",
     list(range(1, 7))),
    ("AR-2", "one complete 144-step daily window",
     list(range(1, 145))),
    ("AR-3", "AR-1 short-term lags + selected daily seasonal lags",
     list(range(1, 7)) + [143, 144, 145, 287, 288, 289]),
    ("AR-4", "AR-3 + weekly lags",
     list(range(1, 7)) + [143, 144, 145, 287, 288, 289,
                          1007, 1008, 1009]),
]


def rule(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def load_series():
    """Return the times and values of each of the three forecast areas."""
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
    """Return the first and last index of a window given by local times."""
    lo = pd.Timestamp(start, tz=TZ)
    hi = pd.Timestamp(end, tz=TZ)
    positions = np.where((index >= lo) & (index <= hi))[0]
    return int(positions[0]), int(positions[-1])


def metrics(actual, predicted):
    """Return the MAE, RMSE and ordinary MAPE for one set of predictions."""
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = float(np.mean(np.abs(error / actual)) * 100.0)
    return mae, rmse, mape


def timed(function, repeats=1):
    """Run the function and return its result with the best time in seconds."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return result, best, repeats


class SelectedLagAR:
    """This linear model is fitted by least squares on a chosen set of lags."""

    def __init__(self, lags):
        self.lags = sorted(int(lag) for lag in lags)
        self.coefficients = None
        self.rank = None
        self.condition_number = None

    @property
    def n_parameters(self):
        return len(self.lags) + 1  # The extra parameter is the intercept.

    def design_matrix(self, series, targets):
        """Build the design matrix with the intercept and each selected lag."""
        lags = np.asarray(self.lags, dtype=np.intp)
        rows = np.asarray(targets, dtype=np.intp)[:, None] - lags[None, :]
        if rows.min() < 0:
            raise ValueError("insufficient history for the requested lags")
        matrix = series[rows]
        return np.hstack([np.ones((len(targets), 1)), matrix])

    def fit(self, series, targets):
        X = self.design_matrix(series, targets)
        y = series[np.asarray(targets, dtype=np.intp)]
        beta, _, rank, singular = np.linalg.lstsq(X, y, rcond=None)
        self.coefficients = beta
        self.rank = int(rank)
        self.condition_number = float(singular.max() / singular.min())
        return self

    def predict(self, series, targets):
        return self.design_matrix(series, targets) @ self.coefficients


def persistence(series, targets):
    """Forecast each value with the previous observation."""
    return series[np.asarray(targets, dtype=np.intp) - 1]


def seasonal_naive(series, targets, period):
    """Forecast each value with the observation one period earlier."""
    return series[np.asarray(targets, dtype=np.intp) - period]


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
    rows_for_log = []
    started = time.perf_counter()
    os.makedirs(RESULTS, exist_ok=True)

    data = load_series()

    rule("A. OFFICIAL FORECASTING AREAS")
    for area in AREAS:
        index, values = data[area]
        print("  square %-6d observations=%d  missing=%d  first=%s  last=%s"
              % (area, len(values), int(np.isnan(values).sum()),
                 index[0], index[-1]))
        assert len(values) == 8928, "expected 8,928 observations"
        assert not np.isnan(values).any()
    print("  all three areas contain 8,928 complete observations")
    print("  (4159 and 4556 remain exploratory only; not forecast here)")

    rule("B. CHRONOLOGICAL EXPERIMENT PERIODS (Europe/Rome)")
    index = data[AREAS[0]][0]
    train_lo, train_hi = window_bounds(index, *TRAIN)
    valid_lo, valid_hi = window_bounds(index, *VALID)
    test_lo, test_hi = window_bounds(index, *TEST)

    for label, (lo, hi) in [("TRAINING", (train_lo, train_hi)),
                            ("VALIDATION", (valid_lo, valid_hi)),
                            ("FINAL TEST", (test_lo, test_hi))]:
        print("  %-11s idx %5d..%5d  n=%5d  %s .. %s"
              % (label, lo, hi, hi - lo + 1, index[lo], index[hi]))

    unused_lo = test_hi + 1
    print("  %-11s idx %5d..%5d  n=%5d  %s .. %s"
          % ("UNUSED", unused_lo, len(index) - 1, len(index) - unused_lo,
             index[unused_lo], index[-1]))
    print("              not used for forecasting-model fitting, validation "
          "selection,")
    print("              or the official test evaluation. Exploratory analysis "
          "of the")
    print("              full 62-day series does cover this period.")

    valid_targets = np.arange(valid_lo, valid_hi + 1)
    test_targets = np.arange(test_lo, test_hi + 1)
    assert len(test_targets) == 1008, "test must hold 1,008 targets"
    assert len(valid_targets) == 1008

    rule("C/D. ROLLING ONE-STEP-AHEAD SETUP AND LEAKAGE CHECKS")
    print("  Evaluation protocol: rolling one-step-ahead.")
    print("    To predict t+1 a model uses ACTUAL observations up to t.")
    print("    No prediction is ever fed back in. Not recursive multi-step.")
    print("    Every design-matrix column is an actual value at a strictly")
    print("    earlier index than its target, so all 1,008 test targets are")
    print("    predicted -- none are lost to a warm-up window.")
    print()

    max_lag = max(max(lags) for _, _, lags in EXPERIMENTS)
    # The rows are aligned so only the lag set differs between experiments.
    train_targets = np.arange(train_lo + max_lag, train_hi + 1)
    print("  common training targets: idx %d..%d  n=%d"
          % (train_targets[0], train_targets[-1], len(train_targets)))
    print("  (aligned to the largest lag used, %d, so experiments differ"
          % max_lag)
    print("   only by lag set and not by training-sample size)")
    print()

    checks = [
        ("training targets strictly before validation",
         train_targets.max() < valid_targets.min()),
        ("validation targets strictly before test",
         valid_targets.max() < test_targets.min()),
        ("no test index appears in training targets",
         not set(train_targets) & set(test_targets)),
        ("no validation index appears in training targets",
         not set(train_targets) & set(valid_targets)),
        ("training targets end on 2013-12-08 23:50",
         str(index[train_targets.max()]).startswith("2013-12-08 23:50")),
        ("test targets are exactly 1,008", len(test_targets) == 1008),
        ("chronological order preserved (no shuffling)",
         bool(np.all(np.diff(train_targets) == 1))),
    ]
    # A target must never appear inside its own input row.
    lag_ok = all(min(lags) >= 1 for _, _, lags in EXPERIMENTS)
    checks.append(("every lag >= 1 (target never inside its own inputs)",
                   lag_ok))
    # Test inputs use earlier actual values and never the test targets.
    for label, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    failed = [label for label, ok in checks if not ok]
    if failed:
        raise SystemExit(
            "Leakage check failed, so no experiment was run: %s"
            % "; ".join(failed))
    print()
    print("  note: scalers for the later neural models must be fitted on")
    print("  training values only; no scaler is fitted in this stage.")

    # The helpers are defined here so no test target is read before this point.
    baselines = [
        ("Persistence", "x(t-1)", lambda s, t: persistence(s, t)),
        ("Daily seasonal naive", "x(t-144)",
         lambda s, t: seasonal_naive(s, t, PER_DAY)),
        ("Weekly seasonal naive", "x(t-1008)",
         lambda s, t: seasonal_naive(s, t, PER_WEEK)),
    ]

    def zero_value_check(label, targets):
        """Confirm that no actual value is zero, because MAPE divides by it."""
        found = False
        for area in AREAS:
            _, values = data[area]
            actual = values[targets]
            zeros = int((actual == 0).sum())
            print("  square %-6d %-11s min=%10.4f  zeros=%d"
                  % (area, label, actual.min(), zeros))
            found = found or zeros > 0
        if found:
            raise SystemExit("Zero target values found - stopping before MAPE.")
        print("  no %s target value equals zero -> ordinary MAPE is used, "
              "with no epsilon added" % label)

    def run_baselines(period_label, targets):
        """Evaluate the three naive baselines over one period."""
        print()
        print("  %s" % period_label)
        print("  %-24s %-8s %10s %10s %9s %14s %16s"
              % ("baseline", "area", "MAE", "RMSE", "MAPE %",
                 "infer_total_s", "per_step_us"))
        for name, formula, function in baselines:
            for area in AREAS:
                _, values = data[area]
                predicted, elapsed, _ = timed(
                    lambda: function(values, targets), repeats=100)
                mae, rmse, mape = metrics(values[targets], predicted)
                print("  %-24s %-8d %10.3f %10.3f %9.3f %14.6f %16.4f"
                      % (name, area, mae, rmse, mape, elapsed,
                         elapsed / len(targets) * 1e6))
                rows_for_log.append({
                    "experiment_id": "BASE-%s-%s" % (
                        name.split()[0][:4].upper(), period_label[:3]),
                    "model": name,
                    "area": area,
                    "history_or_lags": formula,
                    "train_period": "n/a (no fitting)",
                    "validation_period": "%s..%s" % VALID,
                    # Only the test pass may name the test window.
                    "test_period": ("%s..%s" % TEST
                                    if period_label == "TEST"
                                    else "not evaluated at this step"),
                    "evaluated_on": period_label.lower(),
                    "parameters": 0,
                    "MAE": mae, "RMSE": rmse, "MAPE": mape,
                    "training_time_seconds": 0.0,
                    "inference_time_seconds": elapsed,
                    "observation": "no fitted parameters; %s" % formula,
                    "next_decision": "retain as reference for all models",
                })

    rule("E1. MAPE ZERO-VALUE CHECK (VALIDATION TARGETS)")
    zero_value_check("validation", valid_targets)

    rule("E2. BASELINE RESULTS (VALIDATION)")
    run_baselines("VALIDATION", valid_targets)

    rule("F. SELECTED-LAG AUTOREGRESSIVE MODEL - VALIDATION EXPERIMENTS")
    print("  fitting method: numpy.linalg.lstsq (SVD least squares);")
    print("  inv(X.T @ X) is never formed.")
    print()
    print("  %-7s %-8s %6s %7s %10s %10s %9s %11s %13s %12s"
          % ("exp", "area", "lags", "params", "val MAE", "val RMSE",
             "val MAPE", "fit_s", "infer_1008_s", "cond(X)"))

    validation_summary = {}
    for exp_id, description, lags in EXPERIMENTS:
        for area in AREAS:
            _, values = data[area]
            model = SelectedLagAR(lags)

            _, fit_seconds, _ = timed(
                lambda: model.fit(values, train_targets), repeats=5)
            predicted, infer_seconds, _ = timed(
                lambda: model.predict(values, valid_targets), repeats=20)
            mae, rmse, mape = metrics(values[valid_targets], predicted)

            validation_summary[(exp_id, area)] = (mae, rmse, mape)
            print("  %-7s %-8d %6d %7d %10.3f %10.3f %9.3f %11.5f %13.6f %12.3e"
                  % (exp_id, area, len(lags), model.n_parameters, mae, rmse,
                     mape, fit_seconds, infer_seconds,
                     model.condition_number))

            rows_for_log.append({
                "experiment_id": exp_id,
                "model": "Selected-Lag Autoregressive Model",
                "area": area,
                "history_or_lags": description + " | lags=" + (
                    "1-%d" % max(lags) if lags == list(range(1, max(lags) + 1))
                    else ",".join(str(l) for l in lags)),
                "train_period": "%s..%s" % TRAIN,
                "validation_period": "%s..%s" % VALID,
                "test_period": "not evaluated at this step",
                "evaluated_on": "validation",
                "parameters": model.n_parameters,
                "MAE": mae, "RMSE": rmse, "MAPE": mape,
                "training_time_seconds": fit_seconds,
                "inference_time_seconds": infer_seconds,
                "observation": "cond(X)=%.3e rank=%d"
                               % (model.condition_number, model.rank),
                "next_decision": "compare candidates by mean validation MAE",
            })

    print()
    print("  mean validation MAE across the three areas:")
    for exp_id, description, lags in EXPERIMENTS:
        mean_mae = np.mean([validation_summary[(exp_id, a)][0] for a in AREAS])
        mean_rmse = np.mean([validation_summary[(exp_id, a)][1] for a in AREAS])
        mean_mape = np.mean([validation_summary[(exp_id, a)][2] for a in AREAS])
        print("    %-7s MAE %9.3f  RMSE %9.3f  MAPE %7.3f   (%s)"
              % (exp_id, mean_mae, mean_rmse, mean_mape, description))

    best_exp = min(EXPERIMENTS,
                   key=lambda e: np.mean(
                       [validation_summary[(e[0], a)][0] for a in AREAS]))
    print()
    print("  SELECTED BY VALIDATION MAE ONLY: %s (%s)"
          % (best_exp[0], best_exp[1]))

    rule("G. INDEPENDENT VERIFICATION OF THE NUMPY IMPLEMENTATION")
    from statsmodels.regression.linear_model import OLS
    import scipy.linalg as sla

    print("  reference 1: statsmodels OLS on the identical design matrix")
    print("  reference 2: normal equations solved by Cholesky (scipy), which")
    print("               is a different algorithm from the SVD in lstsq")
    print()
    for exp_id, _, lags in EXPERIMENTS:
        for area in AREAS:
            _, values = data[area]
            model = SelectedLagAR(lags).fit(values, train_targets)
            X = model.design_matrix(values, train_targets)
            y = values[train_targets]

            ols = OLS(y, X).fit()
            gram = X.T @ X
            factor = sla.cho_factor(gram)
            beta_chol = sla.cho_solve(factor, X.T @ y)

            coef_diff_ols = float(np.abs(model.coefficients
                                         - ols.params).max())
            coef_diff_chol = float(np.abs(model.coefficients
                                          - beta_chol).max())
            pred_mine = model.predict(values, valid_targets)
            pred_ols = model.design_matrix(values, valid_targets) @ ols.params
            pred_diff = float(np.abs(pred_mine - pred_ols).max())

            print("  %-7s area %-6d max|coef diff| vs OLS %.3e | vs Cholesky "
                  "%.3e | max|pred diff| %.3e"
                  % (exp_id, area, coef_diff_ols, coef_diff_chol, pred_diff))

    print()
    print("  note: AR coefficients are estimated by least squares and are NOT")
    print("  the ACF values. The ACF only informed which lags to offer the")
    print("  model; the fitted weights are a separate estimation problem.")

    # The configuration is frozen, so the test targets are read only now.
    rule("H1. MAPE ZERO-VALUE CHECK (TEST TARGETS)")
    zero_value_check("test", test_targets)

    rule("H2. BASELINE RESULTS (TEST)")
    run_baselines("TEST", test_targets)

    rule("K. FROZEN AR CONFIGURATION -> FINAL TEST (16-22 Dec)")
    final_id, final_description, final_lags = best_exp
    print("  frozen configuration : %s (%s)" % (final_id, final_description))
    print("  lags                 : %s"
          % ("1-%d" % max(final_lags)
             if final_lags == list(range(1, max(final_lags) + 1))
             else ",".join(str(l) for l in final_lags)))
    print("  parameters           : %d" % (len(final_lags) + 1))
    print("  selected by mean validation MAE; final AR test metrics do not")
    print("  enter the selection.")
    print()
    print("  %-8s %10s %10s %9s %11s %14s %16s"
          % ("area", "MAE", "RMSE", "MAPE %", "fit_s", "infer_1008_s",
             "per_step_us"))
    for area in AREAS:
        _, values = data[area]
        model = SelectedLagAR(final_lags)
        _, fit_seconds, _ = timed(
            lambda: model.fit(values, train_targets), repeats=5)
        predicted, infer_seconds, _ = timed(
            lambda: model.predict(values, test_targets), repeats=20)
        mae, rmse, mape = metrics(values[test_targets], predicted)
        print("  %-8d %10.3f %10.3f %9.3f %11.5f %14.6f %16.4f"
              % (area, mae, rmse, mape, fit_seconds, infer_seconds,
                 infer_seconds / len(test_targets) * 1e6))
        rows_for_log.append({
            "experiment_id": "AR-FINAL",
            "model": "Selected-Lag Autoregressive Model",
            "area": area,
            "history_or_lags": "frozen %s | %s" % (final_id, final_description),
            "train_period": "%s..%s" % TRAIN,
            "validation_period": "%s..%s" % VALID,
            "test_period": "%s..%s" % TEST,
            "evaluated_on": "test",
            "parameters": len(final_lags) + 1,
            "MAE": mae, "RMSE": rmse, "MAPE": mape,
            "training_time_seconds": fit_seconds,
            "inference_time_seconds": infer_seconds,
            "observation": "selected by mean validation MAE; final test "
                           "metrics not used for selection",
            "next_decision": "compare against LSTM and TCN in a later stage",
        })

    rule("J. EXPERIMENT LOG")
    written, replaced, total = update_log(rows_for_log)
    print("  rows written this run : %d" % written)
    print("  existing rows replaced: %d" % replaced)
    print("  total rows in log     : %d" % total)
    print("  path                  : %s" % LOG_CSV)

    print()
    print("total wall-clock: %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    # --help must exit without running anything that writes a result file.
    parser = argparse.ArgumentParser(
        description="Run the naive baselines and the Selected-Lag "
                    "autoregressive experiments, then evaluate the frozen "
                    "AR-4 configuration on the test week. Takes no options. "
                    "Updates its rows in results/experiment_log.csv, "
                    "replacing any existing rows with the same "
                    "experiment_id and area rather than appending "
                    "duplicates.")
    parser.parse_args()
    raise SystemExit(main())
