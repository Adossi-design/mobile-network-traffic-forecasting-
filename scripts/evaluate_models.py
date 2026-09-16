"""Reproduce the frozen predictions and build the final evaluation files."""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The sibling modules live in scripts/, which Python already has on sys.path.
import ar_model
import lstm_model
import tcn_model

# This script lives in scripts/, so the repository root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
FORECAST_DIR = os.path.join(FIGS, "final_forecasts")

PRED_CSV = os.path.join(RESULTS, "final_predictions.csv")
METRICS_CSV = os.path.join(RESULTS, "final_model_metrics.csv")
FAILURE_CSV = os.path.join(RESULTS, "failure_analysis.csv")
LOG_CSV = os.path.join(RESULTS, "experiment_log.csv")

AREAS = [5161, 5059, 5259]
TZ = "Europe/Rome"

# The window is the first two weeks, without 1 November as a weekday.
FIRST_TWO_WEEKS_END = "2013-11-15"
EXCLUDED_DATE = "2013-11-01"

AR_LAGS = (list(range(1, 7)) + [143, 144, 145] + [287, 288, 289]
           + [1007, 1008, 1009])
LSTM_SEQ, LSTM_HIDDEN = 36, 64
TCN_SEQ, TCN_CHANNELS = 288, 32

# Above this value the reproduction fails and no result file is written.
REPRO_TOLERANCE = 1e-9

MODELS = [("ar_prediction", "AR"), ("lstm_prediction", "LSTM"),
          ("tcn_prediction", "TCN")]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRIDCOL = "#e5e4e0"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDCOL)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRIDCOL, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def area_characteristics(area):
    """Return the statistics the cross-area summary needs for one area."""
    index, values = ar_model.load_series()[area]
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(),
            "coef_variation": values.std(ddof=1) / values.mean(),
            "peak_to_mean": values.max() / values.mean()}


def weekend_weekday_ratio(area):
    """Calculate the weekend to weekday ratio over the first two weeks."""
    index, values = ar_model.load_series()[area]
    frame = pd.DataFrame({"dt": index, "value": values})
    first = frame[frame["dt"] < pd.Timestamp(FIRST_TWO_WEEKS_END, tz=TZ)]
    weekend = first[first["dt"].dt.dayofweek >= 5]["value"]
    weekday = first[(first["dt"].dt.dayofweek < 5)
                    & (first["dt"].dt.normalize()
                       != pd.Timestamp(EXCLUDED_DATE, tz=TZ))]["value"]
    return weekend.mean() / weekday.mean()


def run_predictions():
    started = time.perf_counter()

    data = ar_model.load_series()
    index = data[AREAS[0]][0]
    train_b = ar_model.window_bounds(index, *ar_model.TRAIN)
    valid_b = ar_model.window_bounds(index, *ar_model.VALID)
    test_b = ar_model.window_bounds(index, *ar_model.TEST)
    bounds = (train_b, valid_b, test_b)

    train_targets = np.arange(train_b[0] + max(AR_LAGS), train_b[1] + 1)
    test_targets = np.arange(test_b[0], test_b[1] + 1)
    assert len(test_targets) == 1008

    rule("AR LAG-INDEX VERIFICATION")
    print("For hand-picked targets, confirm each design-matrix column really")
    print("is series[target - lag] for that lag.")
    model_probe = ar_model.SelectedLagAR(AR_LAGS)
    values_probe = data[5161][1]
    probe_targets = np.array([test_targets[0], test_targets[100],
                              test_targets[500], test_targets[-1]])
    X_probe = model_probe.design_matrix(values_probe, probe_targets)
    lag_position = {lag: i + 1 for i, lag in enumerate(model_probe.lags)}
    all_ok = True
    for row, target in enumerate(probe_targets):
        parts = []
        for lag in (1, 144, 1008):
            if lag in lag_position:
                got = X_probe[row, lag_position[lag]]
                want = values_probe[target - lag]
                ok = got == want
                all_ok = all_ok and ok
                parts.append("lag %-4d %s (design %.6f vs series[t-%d] %.6f)"
                             % (lag, "OK" if ok else "MISMATCH", got, lag, want))
        print("  target idx %d (%s)" % (target, index[target]))
        for part in parts:
            print("      %s" % part)
    print("  intercept column is all ones: %s"
          % bool(np.all(X_probe[:, 0] == 1.0)))
    # Every lag is checked here, not only the three spot checks.
    exhaustive = all(
        np.array_equal(X_probe[:, lag_position[lag]],
                       values_probe[probe_targets - lag])
        for lag in model_probe.lags)
    print("  ALL %d lag columns match series[target-lag]: %s"
          % (len(model_probe.lags), exhaustive))
    print("  spot checks (lags 1, 144, 1008) all correct: %s" % all_ok)

    intercept_ok = bool(np.all(X_probe[:, 0] == 1.0))
    if not (all_ok and exhaustive and intercept_ok):
        raise SystemExit(
            "AR design-matrix verification FAILED (spot checks %s, exhaustive "
            "%s, intercept %s); prediction regeneration stopped and no result "
            "file was written."
            % (all_ok, exhaustive, intercept_ok))

    rule("REGENERATING FROZEN-MODEL PREDICTIONS FOR 16-22 DECEMBER")
    frames = []
    for area in AREAS:
        _, values = data[area]
        print()
        print("  area %d" % area, flush=True)

        persistence = ar_model.persistence(values, test_targets)
        daily = ar_model.seasonal_naive(values, test_targets, 144)
        weekly = ar_model.seasonal_naive(values, test_targets, 1008)

        # The fit is named ar_fit so it does not hide the imported module.
        ar_fit = ar_model.SelectedLagAR(AR_LAGS).fit(values, train_targets)
        ar_pred = ar_fit.predict(values, test_targets)
        print("    AR   refit and predicted", flush=True)

        # The configuration is frozen, so test evaluation is asked for here.
        lstm = lstm_model.train_one(values, bounds, LSTM_SEQ, LSTM_HIDDEN,
                                    evaluate_test=True)
        print("    LSTM retrained (%d epochs, best %d, %.1fs)"
              % (lstm["epochs_run"], lstm["best_epoch"],
                 lstm["train_seconds"]), flush=True)

        tcn = tcn_model.train_one(values, bounds, TCN_SEQ, TCN_CHANNELS,
                                  evaluate_test=True)
        print("    TCN  retrained (%d epochs, best %d, %.1fs)"
              % (tcn["epochs_run"], tcn["best_epoch"],
                 tcn["train_seconds"]), flush=True)

        frames.append(pd.DataFrame({
            "timestamp": index[test_targets].strftime("%Y-%m-%d %H:%M:%S%z"),
            "square_id": area,
            "actual": values[test_targets],
            "persistence": persistence,
            "daily_seasonal_naive": daily,
            "weekly_seasonal_naive": weekly,
            "ar_prediction": ar_pred,
            "lstm_prediction": lstm["test_predictions"],
            "tcn_prediction": tcn["test_predictions"],
        }))

    predictions = pd.concat(frames, ignore_index=True)

    rule("REPRODUCIBILITY CHECK vs RECORDED FROZEN TEST METRICS")
    log = pd.read_csv(LOG_CSV)
    recorded = log[log.evaluated_on == "test"]
    key = {"ar_prediction": "AR-FINAL", "lstm_prediction": "LSTM-FINAL",
           "tcn_prediction": "TCN-FINAL",
           "persistence": "BASE-PERS-TES",
           "daily_seasonal_naive": "BASE-DAIL-TES",
           "weekly_seasonal_naive": "BASE-WEEK-TES"}

    # Every expected row must be present before the comparison is believed.
    missing = [(exp_id, area) for exp_id in key.values() for area in AREAS
               if recorded[(recorded.experiment_id == exp_id)
                           & (recorded.area == area)].empty]
    if missing:
        raise SystemExit(
            "Reproduction check cannot run: %d expected test row(s) missing "
            "from %s: %s. Existing result files left untouched."
            % (len(missing), LOG_CSV, missing))
    print("  all %d expected (experiment_id, area) test rows found"
          % (len(key) * len(AREAS)))
    print()

    print("  %-22s %-7s %10s %10s %11s %11s %11s"
          % ("model", "area", "MAE now", "MAE rec.", "dMAE", "dRMSE", "dMAPE"))
    worst = 0.0
    for column, exp_id in key.items():
        for area in AREAS:
            part = predictions[predictions.square_id == area]
            mae, rmse, mape = ar_model.metrics(part["actual"].to_numpy(),
                                               part[column].to_numpy())
            ref = recorded[(recorded.experiment_id == exp_id)
                           & (recorded.area == area)].iloc[0]
            d_mae, d_rmse = mae - ref.MAE, rmse - ref.RMSE
            d_mape = mape - ref.MAPE
            worst = max(worst, abs(d_mae), abs(d_rmse), abs(d_mape))
            print("  %-22s %-7d %10.4f %10.4f %11.2e %11.2e %11.2e"
                  % (column, area, mae, ref.MAE, d_mae, d_rmse, d_mape))
    print()
    print("  worst absolute difference across every model/area/metric: %.3e"
          % worst)
    print("  declared tolerance: %.1e" % REPRO_TOLERANCE)

    if worst > REPRO_TOLERANCE:
        raise SystemExit(
            "REPRODUCTION FAILED: worst difference %.3e exceeds the declared "
            "tolerance %.1e. The frozen configurations did not reproduce the "
            "recorded test metrics, so %s and %s were NOT written and the "
            "existing files are unchanged."
            % (worst, REPRO_TOLERANCE, PRED_CSV, METRICS_CSV))
    print("  reproduction within tolerance -> safe to write result files")

    label = {"ar_prediction": "AR (Selected-Lag)",
             "lstm_prediction": "LSTM", "tcn_prediction": "TCN"}
    main_rows = []
    for column, name in label.items():
        for area in AREAS:
            part = predictions[predictions.square_id == area]
            mae, rmse, mape = ar_model.metrics(part["actual"].to_numpy(),
                                               part[column].to_numpy())
            main_rows.append({"square_id": area, "model": name, "MAE": mae,
                              "RMSE": rmse, "MAPE": mape})

    # Results are written only now, once the reproduction check has passed.
    os.makedirs(RESULTS, exist_ok=True)
    predictions.to_csv(PRED_CSV, index=False)
    print()
    print("  wrote %d rows -> %s" % (len(predictions), PRED_CSV))
    pd.DataFrame(main_rows).to_csv(METRICS_CSV, index=False)
    print("  wrote %s" % METRICS_CSV)

    rule("THREE REPORT-READY TABLES (selected models only)")
    frame = pd.DataFrame(main_rows)
    for area in AREAS:
        print()
        print("  Square %d" % area)
        print("  %-20s %10s %10s %9s" % ("Model", "MAE", "RMSE", "MAPE %"))
        part = frame[frame.square_id == area].sort_values("MAE")
        for _, r in part.iterrows():
            print("  %-20s %10.3f %10.3f %9.3f"
                  % (r.model, r.MAE, r.RMSE, r.MAPE))

    print()
    print("prediction phase wall-clock: %.1f s" % (time.perf_counter() - started))
    return 0


def run_analysis():
    os.makedirs(FORECAST_DIR, exist_ok=True)
    predictions = pd.read_csv(PRED_CSV)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"],
                                              utc=True).dt.tz_convert(TZ)

    rule("NINE REQUIRED FORECAST FIGURES")
    created = []
    for area in AREAS:
        part = predictions[predictions.square_id == area].sort_values(
            "timestamp")
        # The three models of one area share the same y-limits.
        stack = np.concatenate([part["actual"].to_numpy()]
                               + [part[c].to_numpy() for c, _ in MODELS])
        pad = 0.04 * (stack.max() - stack.min())
        ylim = (stack.min() - pad, stack.max() + pad)

        # The time zone is dropped because matplotlib would draw labels in UTC.
        local_time = part["timestamp"].dt.tz_localize(None)
        for column, name in MODELS:
            fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
            style_axes(ax)
            ax.plot(local_time, part["actual"], color=INK_2,
                    linewidth=1.5, label="Actual", zorder=3)
            ax.plot(local_time, part[column], color=BLUE,
                    linewidth=1.2, label="%s prediction" % name, zorder=4)
            ax.set_ylim(*ylim)
            ax.set_title("%s Forecast vs Actual - Square %d, 16-22 December "
                         "2013 (one-step-ahead)" % (name, area),
                         fontsize=12, color=INK, pad=10)
            ax.set_xlabel("Date and time (Europe/Rome local time)",
                          fontsize=9.5, color=INK_2)
            ax.set_ylabel("Internet traffic\nper 10-minute interval",
                          fontsize=9.5, color=INK_2)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(12,)))
            ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2,
                      loc="upper right")
            fig.tight_layout()
            path = os.path.join(FORECAST_DIR,
                                "%s_square_%d.png" % (name.lower(), area))
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)
            created.append(path)
            print("  %s  (y-limits %.0f..%.0f shared within the area)"
                  % (os.path.basename(path), ylim[0], ylim[1]))

    rule("FAILURE ANALYSIS")
    records = []
    for area in AREAS:
        part = predictions[predictions.square_id == area].sort_values(
            "timestamp").reset_index(drop=True)
        actual = part["actual"].to_numpy()
        # Using the persistence column keeps the change for the first interval.
        change = np.abs(actual - part["persistence"].to_numpy())
        for column, name in MODELS:
            predicted = part[column].to_numpy()
            frame = pd.DataFrame({
                "timestamp": part["timestamp"],
                "square_id": area,
                "model": name,
                "actual": actual,
                "prediction": predicted,
                "abs_error": np.abs(predicted - actual),
                "abs_pct_error": np.abs((predicted - actual) / actual) * 100,
                "abs_change": change,
                "hour": part["timestamp"].dt.hour,
                "is_weekend": part["timestamp"].dt.dayofweek >= 5,
            })
            records.append(frame)
    diagnostics = pd.concat(records, ignore_index=True)

    # The column starts as object dtype because pandas 3 rejects strings.
    diagnostics["traffic_quartile"] = pd.Series("", index=diagnostics.index,
                                                dtype="object")
    diagnostics["change_quartile"] = pd.Series("", index=diagnostics.index,
                                               dtype="object")
    for area in AREAS:
        mask = diagnostics.square_id == area
        base = diagnostics[mask & (diagnostics.model == "AR")]
        t_edges = np.quantile(base["actual"], [0, .25, .5, .75, 1.0])
        c_edges = np.quantile(base["abs_change"], [0, .25, .5, .75, 1.0])
        diagnostics.loc[mask, "traffic_quartile"] = pd.cut(
            diagnostics.loc[mask, "actual"], bins=t_edges,
            labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
            include_lowest=True).astype(str)
        diagnostics.loc[mask, "change_quartile"] = pd.cut(
            diagnostics.loc[mask, "abs_change"], bins=c_edges,
            labels=["C1 smallest", "C2", "C3", "C4 largest"],
            include_lowest=True).astype(str)

    diagnostics.to_csv(FAILURE_CSV, index=False)
    print("  wrote %d diagnostic rows -> %s" % (len(diagnostics), FAILURE_CSV))

    print()
    print("  MAE and MAPE BY ACTUAL-TRAFFIC QUARTILE")
    print("  %-7s %-6s %-11s %9s %9s %11s"
          % ("area", "model", "quartile", "MAE", "MAPE %", "mean actual"))
    for area in AREAS:
        for _, name in MODELS:
            sub = diagnostics[(diagnostics.square_id == area)
                              & (diagnostics.model == name)]
            for q in ["Q1 lowest", "Q2", "Q3", "Q4 highest"]:
                block = sub[sub.traffic_quartile == q]
                print("  %-7d %-6s %-11s %9.2f %9.3f %11.1f"
                      % (area, name, q, block.abs_error.mean(),
                         block.abs_pct_error.mean(), block.actual.mean()))
        print()

    print("  MAE BY QUARTILE OF ABSOLUTE CHANGE FROM PREVIOUS OBSERVATION")
    print("  %-7s %-6s %-13s %9s %9s %13s"
          % ("area", "model", "change q.", "MAE", "MAPE %", "mean |change|"))
    for area in AREAS:
        for _, name in MODELS:
            sub = diagnostics[(diagnostics.square_id == area)
                              & (diagnostics.model == name)]
            for q in ["C1 smallest", "C2", "C3", "C4 largest"]:
                block = sub[sub.change_quartile == q]
                print("  %-7d %-6s %-13s %9.2f %9.3f %13.1f"
                      % (area, name, q, block.abs_error.mean(),
                         block.abs_pct_error.mean(), block.abs_change.mean()))
        print()

    print("  WEEKDAY vs WEEKEND MAE")
    print("  %-7s %-6s %11s %11s %9s"
          % ("area", "model", "weekday MAE", "weekend MAE", "ratio"))
    for area in AREAS:
        for _, name in MODELS:
            sub = diagnostics[(diagnostics.square_id == area)
                              & (diagnostics.model == name)]
            wd = sub[~sub.is_weekend].abs_error.mean()
            we = sub[sub.is_weekend].abs_error.mean()
            print("  %-7d %-6s %11.2f %11.2f %9.3f"
                  % (area, name, wd, we, we / wd))
        print()

    print("  LARGEST ABSOLUTE ERRORS (top 3 per model and area)")
    print("  %-7s %-6s %-22s %10s %10s %10s %9s"
          % ("area", "model", "timestamp", "actual", "prediction",
             "abs error", "pct err"))
    for area in AREAS:
        for _, name in MODELS:
            sub = diagnostics[(diagnostics.square_id == area)
                              & (diagnostics.model == name)]
            for _, r in sub.nlargest(3, "abs_error").iterrows():
                print("  %-7d %-6s %-22s %10.1f %10.1f %10.1f %9.2f"
                      % (area, name,
                         r.timestamp.strftime("%Y-%m-%d %H:%M"),
                         r.actual, r.prediction, r.abs_error,
                         r.abs_pct_error))
        print()

    rule("HYPOTHESIS 1: does the TCN produce disproportionately large "
         "PERCENTAGE errors at LOW traffic on 5161?")
    sub = diagnostics[diagnostics.square_id == 5161]
    print("  %-6s %-11s %9s %9s %11s"
          % ("model", "quartile", "MAE", "MAPE %", "mean actual"))
    for _, name in MODELS:
        block = sub[sub.model == name]
        for q in ["Q1 lowest", "Q2", "Q3", "Q4 highest"]:
            b = block[block.traffic_quartile == q]
            print("  %-6s %-11s %9.2f %9.3f %11.1f"
                  % (name, q, b.abs_error.mean(), b.abs_pct_error.mean(),
                     b.actual.mean()))
        print()
    print("  TCN MAPE in Q1 relative to the other models in Q1:")
    q1 = sub[sub.traffic_quartile == "Q1 lowest"]
    for _, name in MODELS:
        print("    %-6s Q1 MAPE %.3f%%  Q1 MAE %.2f"
              % (name, q1[q1.model == name].abs_pct_error.mean(),
                 q1[q1.model == name].abs_error.mean()))
    tcn_q1 = q1[q1.model == "TCN"]
    tcn_all = sub[sub.model == "TCN"]
    share = 100 * tcn_q1.abs_pct_error.sum() / tcn_all.abs_pct_error.sum()
    print("    share of the TCN's total percentage error coming from Q1: "
          "%.1f%% (Q1 is 25%% of the intervals)" % share)
    # This checks whether the TCN forecasts too high when traffic is low.
    bias = (tcn_q1.prediction - tcn_q1.actual).mean()
    print("    TCN mean signed error in Q1: %+.2f (positive = over-forecast)"
          % bias)
    for _, name in MODELS:
        b = q1[q1.model == name]
        print("    %-6s mean signed error in Q1: %+.2f"
              % (name, (b.prediction - b.actual).mean()))

    rule("HYPOTHESIS 2: where is AR's extra error on 5059 concentrated?")
    sub = diagnostics[diagnostics.square_id == 5059]
    ar = sub[sub.model == "AR"].reset_index(drop=True)
    lstm = sub[sub.model == "LSTM"].reset_index(drop=True)
    gap = ar.abs_error - lstm.abs_error
    print("  AR MAE %.2f vs LSTM MAE %.2f -> AR is %.2f worse on average"
          % (ar.abs_error.mean(), lstm.abs_error.mean(), gap.mean()))
    print()
    print("  extra AR error by ACTUAL-TRAFFIC quartile:")
    for q in ["Q1 lowest", "Q2", "Q3", "Q4 highest"]:
        m = ar.traffic_quartile == q
        print("    %-11s AR %8.2f  LSTM %8.2f  gap %+8.2f  (n=%d)"
              % (q, ar[m].abs_error.mean(), lstm[m].abs_error.mean(),
                 gap[m].mean(), m.sum()))
    print()
    print("  extra AR error by ABSOLUTE-CHANGE quartile:")
    for q in ["C1 smallest", "C2", "C3", "C4 largest"]:
        m = ar.change_quartile == q
        print("    %-13s AR %8.2f  LSTM %8.2f  gap %+8.2f  (n=%d)"
              % (q, ar[m].abs_error.mean(), lstm[m].abs_error.mean(),
                 gap[m].mean(), m.sum()))
    print()
    print("  extra AR error, weekday vs weekend:")
    for label, m in [("weekday", ~ar.is_weekend), ("weekend", ar.is_weekend)]:
        print("    %-8s AR %8.2f  LSTM %8.2f  gap %+8.2f  (n=%d)"
              % (label, ar[m].abs_error.mean(), lstm[m].abs_error.mean(),
                 gap[m].mean(), m.sum()))
    print()
    print("  extra AR error by hour of day (largest six gaps):")
    by_hour = gap.groupby(ar.hour).mean().sort_values(ascending=False)
    for hour, value in by_hour.head(6).items():
        print("    %02d:00  gap %+8.2f" % (hour, value))
    print()
    print("  AR mean signed error on 5059: %+.2f (positive = over-forecast)"
          % (ar.prediction - ar.actual).mean())
    print("  LSTM mean signed error on 5059: %+.2f"
          % (lstm.prediction - lstm.actual).mean())

    rule("SELECTING THE FAILURE CASE (by measured error, not appearance)")
    diagnostics["day"] = diagnostics.timestamp.dt.date
    combined = (diagnostics.groupby(["square_id", "day"])["abs_error"]
                .mean().reset_index()
                .sort_values("abs_error", ascending=False))
    print("  worst days by mean absolute error across all three models:")
    for _, r in combined.head(6).iterrows():
        print("    square %d  %s  mean |error| %.2f"
              % (r.square_id, r.day, r.abs_error))

    hourly = (diagnostics.groupby(
        ["square_id", diagnostics.timestamp.dt.floor("1h")])["abs_error"]
        .mean().reset_index(name="mean_abs_error")
        .sort_values("mean_abs_error", ascending=False))
    hourly.columns = ["square_id", "hour_start", "mean_abs_error"]
    print()
    print("  worst single hours (mean across the three models):")
    for _, r in hourly.head(8).iterrows():
        print("    square %d  %s  mean |error| %.2f"
              % (r.square_id, r.hour_start.strftime("%Y-%m-%d %H:%M"),
                 r.mean_abs_error))

    worst = hourly.iloc[0]
    case_area = int(worst.square_id)
    centre = worst.hour_start
    lo = centre - pd.Timedelta(hours=5)
    hi = centre + pd.Timedelta(hours=6)

    part = predictions[(predictions.square_id == case_area)
                       & (predictions.timestamp >= lo)
                       & (predictions.timestamp <= hi)].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURFACE)
    style_axes(ax)
    case_time = part["timestamp"].dt.tz_localize(None)
    centre_naive = centre.tz_localize(None)
    ax.plot(case_time, part["actual"], color=INK_2, linewidth=2.4,
            label="Actual", zorder=5)
    for (column, name), colour in zip(MODELS, (BLUE, ORANGE, AQUA)):
        ax.plot(case_time, part[column], color=colour, linewidth=1.8,
                label=name, zorder=4)
    ax.axvspan(centre_naive, centre_naive + pd.Timedelta(hours=1),
               color=GRIDCOL, alpha=0.75, linewidth=0, zorder=0)
    ax.annotate("worst hour", xy=(centre_naive + pd.Timedelta(minutes=30),
                                  ax.get_ylim()[1] * 0.96),
                ha="center", va="top", fontsize=9, color=INK)
    ax.set_title("Failure case - Square %d, %s (all three models, "
                 "one-step-ahead)"
                 % (case_area, centre.strftime("%d %B %Y")),
                 fontsize=12, color=INK, pad=10)
    ax.set_xlabel("Time (Europe/Rome local time)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Internet traffic\nper 10-minute interval", fontsize=9.5,
                  color=INK_2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=4,
              loc="upper left")
    fig.tight_layout()
    case_path = os.path.join(FIGS, "failure_case.png")
    fig.savefig(case_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    created.append(case_path)

    print()
    print("  SELECTED CASE: square %d, hour beginning %s"
          % (case_area, centre))
    window = diagnostics[(diagnostics.square_id == case_area)
                         & (diagnostics.timestamp >= centre)
                         & (diagnostics.timestamp < centre
                            + pd.Timedelta(hours=1))]
    print("  %-6s %10s %10s %10s %9s"
          % ("model", "MAE", "max |err|", "mean act.", "MAPE %"))
    for _, name in MODELS:
        b = window[window.model == name]
        print("  %-6s %10.2f %10.2f %10.1f %9.2f"
              % (name, b.abs_error.mean(), b.abs_error.max(),
                 b.actual.mean(), b.abs_pct_error.mean()))
    print()
    print("  interval detail in that hour:")
    act = predictions[(predictions.square_id == case_area)
                      & (predictions.timestamp >= centre)
                      & (predictions.timestamp < centre
                         + pd.Timedelta(hours=1))].sort_values("timestamp")
    print("  %-17s %10s %10s %10s %10s"
          % ("time", "actual", "AR", "LSTM", "TCN"))
    for _, r in act.iterrows():
        print("  %-17s %10.1f %10.1f %10.1f %10.1f"
              % (r.timestamp.strftime("%Y-%m-%d %H:%M"), r.actual,
                 r.ar_prediction, r.lstm_prediction, r.tcn_prediction))
    prev = predictions[(predictions.square_id == case_area)
                       & (predictions.timestamp < centre)].sort_values(
                           "timestamp").tail(3)
    print("  preceding intervals (actual): %s"
          % ", ".join("%s=%.1f" % (r.timestamp.strftime("%H:%M"), r.actual)
                      for _, r in prev.iterrows()))

    rule("CROSS-AREA COMPARISON (descriptive, three areas only)")
    print("  %-7s %9s %8s %9s %10s %8s %8s %8s"
          % ("area", "mean", "CoV", "peak/mean", "we/wd*", "AR MAE",
             "LSTM MAE", "TCN MAE"))
    for area in AREAS:
        row = area_characteristics(area)
        ratio = weekend_weekday_ratio(area)
        maes = {}
        for _, name in MODELS:
            b = diagnostics[(diagnostics.square_id == area)
                            & (diagnostics.model == name)]
            maes[name] = b.abs_error.mean()
        print("  %-7d %9.1f %8.3f %9.3f %10.2f %8.2f %8.2f %8.2f"
              % (area, row["mean"], row["coef_variation"],
                 row["peak_to_mean"], ratio, maes["AR"], maes["LSTM"],
                 maes["TCN"]))
    print("  * first-two-week weekend/weekday mean-traffic ratio,")
    print("    1 November excluded from the weekday side")

    print()
    print("figures created:")
    for path in created:
        print("  %s" % path)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation of the three frozen models.")
    parser.add_argument("--phase", choices=["predict", "analyse", "all"],
                        default="all")
    args = parser.parse_args()

    if args.phase in ("predict", "all"):
        run_predictions()
    if args.phase in ("analyse", "all"):
        if not os.path.exists(PRED_CSV):
            raise SystemExit(
                "%s not found. Run --phase predict first." % PRED_CSV)
        run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
