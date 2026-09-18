# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Video presentation

[Watch my project presentation](https://drive.google.com/file/d/1ouYuPzJgWGZpmIapXXSweACf98cm1Vri/view?usp=sharing)

## 1. Overview

In this project I compare three models that predict mobile Internet traffic in Milan ten minutes ahead. The first is a selected-lag autoregressive model, which is a linear formula built from chosen past values, the second is an LSTM, which keeps its own memory of the sequence, and the third is a Temporal Convolutional Network, which reads a long window of past values with filters that never look forward. All three use the same training, validation and test periods. They do not receive the same amount of history, because each one was given the window that performed best for it on validation. The data is a public Telecom Italia dataset from Harvard Dataverse. It records mobile activity every ten minutes from 1 November 2013 to 1 January 2014, across a grid of 10,000 small areas. The forecasting work covers the three areas with the highest total Internet traffic, which are squares 5161, 5059 and 5259.

Mobile traffic rises and falls through the day, and weekends often look different from weekdays. The pattern also changes from one part of the city to another, so two areas can reach their peak at different hours. An operator who can estimate the next few minutes of demand is able to prepare capacity in advance, while an operator who reacts only after congestion appears is always one step behind. This is why short-term forecasting is treated here as a practical engineering problem rather than an academic exercise.

Each model predicts only the next ten-minute interval, and it always works from real observations up to the current interval. No prediction is ever fed back as an input. An error made at one step therefore cannot build up into the next one.

I split the data by date and never at random. Training covers 1 November to 8 December, validation covers the week after it, and the test week is 16 to 22 December 2013. I chose every model setting on the validation week alone, and the test metrics were never used to select or change a configuration. One point has to be stated openly, because my exploratory analysis in the notebook used the full 62 days, and that includes the test week, so I do not describe that week as an untouched holdout.

I also measure three simple forecasts under the same rules, and the strongest of them simply repeats the last observed value. A complex model is only worth its cost when it beats a forecast that needs no training at all.

My main question is not only which model has the lowest average error. It is whether the best model changes with the kind of area being forecast. I also wanted to know whether the neural networks earn back the extra time they need to train and to run. This is why the results are reported for each area separately and with three different error measures. It is also why every model is timed on the same machine, and why I look closely at the moments where the models fail.

The LSTM performed best on average across the three areas, but it did not win everywhere. The autoregressive model had the lowest MAE and RMSE on Square 5259, and on Square 5161 each error measure pointed to a different winner.

The cost of the three models is very different, because the LSTM took about 72 seconds to train on Square 5161 and the TCN took about 393 seconds, while the autoregressive model was fitted in a few milliseconds.

I left the raw dataset out of this repository, because it takes about 19.38 GiB. The processed files and every saved result are included. You can therefore run the notebook without downloading the raw data and without training any model again. To rebuild the processed files yourself, download the raw data and follow the steps in section 8.

## 2. Research question

> How do different sequential models compare for one-step-ahead mobile network
> traffic forecasting, and how does their performance vary across geographical
> areas with different traffic characteristics?

## 3. Dataset

The analysis uses one public dataset from Harvard Dataverse, and a second dataset (the Milano Grid) is listed below as well, because it shows where the area identifiers come from, even though no step of the workflow needs it.

**Telecommunications - SMS, Call, Internet - MI**
Telecom Italia, 2015, Harvard Dataverse, V1.3.
DOI: [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)
62 daily files, 19.38 GiB, 319,896,289 rows, 10 minute resolution,
1 November 2013 to 1 January 2014.

**Milano Grid** (companion dataset, not required here)
Telecom Italia, 2015, Harvard Dataverse, V1.3.
DOI: [10.7910/DVN/QJWLFU](https://doi.org/10.7910/DVN/QJWLFU)
It is a single GeoJSON file with 10,000 grid cells. It gives the shape of every square identifier used in this project. No script or figure in this repository reads it.

The dataset is described in:

G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi,
F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri, "A multi-source dataset
of urban life in the city of Milan and the Province of Trentino,"
*Scientific Data*, vol. 2, Art. no. 150055, 2015.
DOI: [10.1038/sdata.2015.55](https://doi.org/10.1038/sdata.2015.55)

The telecommunications dataset is protected by a Dataverse guestbook, so the download script sends the required response and then uses the signed URL that Dataverse returns, reading the email address from the `DATAVERSE_EMAIL` environment variable. The script downloads this dataset only and never fetches the companion grid.

## 4. Models compared

| Model | Type | Configuration | Parameters |
|---|---|---|---|
| Selected-Lag AR | Linear statistical | Lags 1-6, 143-145, 287-289, 1007-1009 | 16 |
| LSTM | Recurrent neural | 1 layer, 64 hidden units, sequence length 36 | 17,217 |
| TCN | Causal convolutional | 32 channels, kernel 3, dilations 1-64, sequence length 288 | 40,577 |

The baselines are persistence, a daily seasonal naive forecast at lag 144, and a weekly seasonal naive forecast at lag 1008.

The forecasts cover squares 5161, 5059 and 5259, because those are the three areas with the highest total Internet traffic. Squares 4159 and 4556 appear only in the exploratory analysis.

## 5. Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── mobile_network_traffic_forecasting.ipynb   the main analysis document
│
├── scripts/
│   ├── download_data.py       download and verify the raw dataset
│   ├── prepare_data.py        aggregate 319.9M rows, build the area series
│   ├── ar_model.py            baselines and the autoregressive model
│   ├── lstm_model.py          LSTM experiments and the frozen LSTM
│   ├── tcn_model.py           TCN experiments and the frozen TCN
│   ├── evaluate_models.py     final predictions, figures, failure analysis
│   ├── benchmark_models.py    representative timing benchmark
│   └── memory_benchmark.py    dataframe memory for one raw daily file
│
├── data/
│   ├── raw/                   not in Git, about 19.38 GiB
│   └── processed/             total_traffic_by_square.csv
│                              selected_area_timeseries.csv
│
└── results/
    ├── experiment_log.csv     documented model-selection and evaluation
    │                          experiments
    ├── final_model_metrics.csv
    ├── final_predictions.csv
    ├── final_timing.csv
    ├── failure_analysis.csv
    ├── lstm_learning_curves.json   per-epoch training history
    ├── tcn_learning_curves.json    per-epoch training history
    └── figures/
        ├── final_forecasts/   the nine actual-versus-predicted plots
        └── failure_case.png
```

The notebook computes the exploratory analysis directly from the processed CSV files. That covers the autocorrelation, the seasonal decomposition and the stationarity test. It also draws the learning curves from the saved JSON training histories. The scripts handle the work that is too heavy or too slow for a notebook, which is reading the raw dataset, training the models and timing them.

The `results/figures/` folder holds only the figures that a script produces. These are the nine forecast plots and the failure case, and all of them are written by `scripts/evaluate_models.py`. Every other figure in the notebook is drawn inline from the processed files and the saved results.

## 6. Installation

The project was tested with Python 3.13.1 and every dependency is pinned to an exact version in `requirements.txt`.

```
git clone https://github.com/Adossi-design/mobile-network-traffic-forecasting-.git
cd mobile-network-traffic-forecasting-
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS or Linux
pip install -r requirements.txt
```

On Windows the `py` launcher works when `python` is not registered on your PATH. The commands then become `py -m venv .venv` and `py scripts/ar_model.py`.

PyTorch is pinned to the CPU build (`torch==2.14.0+cpu`). That wheel is not on PyPI, so `requirements.txt` includes the PyTorch CPU index:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

Running `pip install -r requirements.txt` is therefore enough on its own. No GPU is needed, because the whole project was developed and timed on a CPU. If you already have a CUDA build of PyTorch and want to keep it, install every line except the `torch` one.

## 7. Quick way to explore the project

This is the recommended way to start, because it does **not** download the 19.38 GiB dataset and does **not** retrain any neural network.

```
pip install -r requirements.txt
jupyter notebook notebooks/mobile_network_traffic_forecasting.ipynb
```

After that, run all the cells. The notebook reads only the small processed files and the saved result files, so on the reference machine it usually finishes in tens of seconds. It walks through the whole investigation, from the structure of the raw data to the failure analysis.

## 8. Full reproduction workflow

This workflow rebuilds everything from the raw telecommunications dataset, which covers the processed traffic files, the documented model experiments, the frozen predictions, the evaluation outputs and the timing results. It does not generate the bibliography, which was checked by hand, and it depends on the dataset remaining available from Harvard Dataverse. The workflow is **not** fast. Expect several hours in total, and most of that time goes on the download and on training the TCN.

**Step 1 - Download the raw data:** The download is about 19.38 GiB. It took roughly 3.5 hours on the connection used here. The script is safe to stop and start again, because finished files are skipped and partial files resume from where they stopped.

Harvard Dataverse needs an email address for this dataset's guestbook. The script reads it from the environment, so no personal identity is stored in the source:

Set it in your shell first. For PowerShell:

```
$env:DATAVERSE_EMAIL = "your_email@example.com"
```

For macOS, Linux or Git Bash:

```
export DATAVERSE_EMAIL="your_email@example.com"
```

Then run the downloader:

```
python scripts/download_data.py --workers 1 --verify-md5
```

The `--list-only` option prints the file list only. It does not need the variable to be set.

With `--verify-md5`, every file is checked against the MD5 that Dataverse reports. This includes files already sitting on disk from an earlier run. A file whose size matches is still hashed, and it is accepted only when the checksum matches as well. A file that fails the check is removed and downloaded again, so a corrupted download can never be reused without anyone noticing.

**Step 2 - Prepare the processed data:** This step reads all 319,896,289 rows in chunks and writes the small CSV files. It takes about 9 minutes and stays inside roughly 91 MiB of memory.

```
python scripts/prepare_data.py
```

The preparation stops before writing anything unless all 62 expected daily files are present, covering 1 November 2013 to 1 January 2014. It checks the exact filenames. An incomplete download or an unexpected substitution is therefore refused, instead of being turned into incomplete processed data. Each selected area must also fill all 8,928 expected time slots before the series file is written.

**Step 3 - Baselines and the autoregressive experiments:** This takes about 5 seconds.

```
python scripts/ar_model.py
```

**Step 4 - LSTM experiments:** Each stage trains one model for each area.

```
python scripts/lstm_model.py --stage seqlen
python scripts/lstm_model.py --stage hidden --length 36
python scripts/lstm_model.py --stage final --length 36 --hidden 64
```

**Step 5 - TCN experiments:** This is the slowest part of the whole workflow. It takes roughly 1.5 hours in total on the reference machine.

```
python scripts/tcn_model.py --stage verify
python scripts/tcn_model.py --stage seqlen
python scripts/tcn_model.py --stage channels --length 288
python scripts/tcn_model.py --stage final --length 288 --channels 32
```

**Step 6 - Final evaluation:** Run the two phases explicitly. The `predict` phase is the expensive one, because it retrains all three frozen configurations and checks the regenerated metrics against the recorded ones. It writes `results/final_predictions.csv` and `results/final_model_metrics.csv` only when that check passes, and it takes about 37 minutes. The `analyse` phase is fast, because it only reads the saved predictions. It writes `results/failure_analysis.csv` and every figure in `results/figures/`, so I can run it again on its own without retraining anything.

```
python scripts/evaluate_models.py --phase predict
python scripts/evaluate_models.py --phase analyse
```

**Step 7 - Timing benchmark:** This is kept separate, so that it never runs by accident during an ordinary evaluation.

```
python scripts/benchmark_models.py
```

**Step 8 - Memory benchmark, optional:** This reproduces the memory table in section 3 of the notebook. It reads one raw daily file, so it needs step 1. It measures the memory held by the dataframe itself, which is a different quantity from the peak working set of the process that `prepare_data.py` reports.

```
python scripts/memory_benchmark.py
```

Steps 1, 2 and the optional step 8 need the raw dataset. Steps 3 to 7 work from the small processed files alone.

## 9. Main results

The test period is 16 to 22 December 2013. It holds 1,008 one-step-ahead predictions for each area.

Mean results across the three forecast areas:

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| LSTM | 74.67 | 108.38 | 8.09 |
| TCN | 78.36 | 111.98 | 8.89 |
| Selected-Lag AR | 81.77 | 116.31 | 8.65 |
| Persistence (benchmark) | 83.43 | 119.61 | 8.42 |

MAE for each area:

| Area | AR | LSTM | TCN |
|---|---|---|---|
| 5161 | 87.94 | **86.38** | 88.58 |
| 5059 | 91.56 | **69.80** | 79.61 |
| 5259 | **65.80** | 67.85 | 66.89 |

The LSTM has the best mean performance on all three metrics, but no model wins everywhere: Square 5259 goes to the AR model on MAE and RMSE. On Square 5161 the winner changes with the metric: the LSTM on MAE, the TCN on RMSE and the AR model on MAPE.

By MAE, the LSTM and the TCN beat persistence on all three areas, while the AR model loses to persistence on Square 5059. By MAPE, the LSTM and the TCN are worse than persistence on Square 5161, and the AR model is worse than persistence on Square 5059.

The representative cost on Square 5161 using median values on a CPU only:

| Model | Training | 1,008 predictions | Parameters |
|---|---|---|---|
| AR | 0.002665 s | 0.007481 s | 16 |
| LSTM | 72.26 s | 1.6394 s | 17,217 |
| TCN | 393.02 s | 14.0445 s | 40,577 |

## 10. Reproducibility notes

- **The splits follow the calendar and are never random:** Training runs from 1 November to 8 December with 5,472 observations per area, while validation runs from 9 to 15 December and test from 16 to 22 December with 1,008 observations each. Data after 22 December was never used for fitting a model, for choosing a configuration on validation, or for the official test evaluation.
- **Model selection used validation only:** The configurations were selected on validation metrics. The final test metrics were never used to choose or change them. The tuning stages of `lstm_model.py` and `tcn_model.py` do not build test sequences at all. Only the frozen final stage and the final evaluation ask for test results.
- **The three forecast areas were chosen from the whole dataset:** Squares 5161, 5059 and 5259 are the busiest by total traffic over all 62 days. The step of choosing the areas is therefore not an untouched holdout. The model selection that follows for each area still uses validation metrics only.
- **The exploratory analysis used the full 62-day series:** It helped shape the candidate lag and history window choices, so it included the official evaluation week and the later unused stretch. Selection still followed the rule above, on validation metrics alone. The week of 16 to 22 December should therefore be read as the official evaluation window of the assignment, rather than a completely untouched holdout. No model was fitted on test data.
- **Scalers were fitted on training values only** and then applied unchanged to validation and test.
- **A fixed seed of 42** was used for both neural models.
- **The frozen models reproduce:** Retraining from scratch and recomputing the metrics gave a worst difference of 5.684e-14 against the recorded values.
- **The experiment log holds one record for each documented experiment and area:** `results/experiment_log.csv` has 66 rows. Running a stage again replaces its own matching `(experiment_id, area)` rows instead of adding duplicates. Following the workflow above on a fresh clone therefore never doubles the log.

The reference machine was an Intel Core i7-1255U with 10 physical cores and 15.65 GiB of usable RAM. It ran Windows 11 Pro build 26200, Python 3.13.1 and PyTorch 2.14.0+cpu on 10 threads. No CUDA GPU was used, so every run was CPU-only.

## 11. Data storage note

The raw telecommunications dataset is about **19.38 GiB** across 62 files. It is deliberately kept out of Git, and `data/raw/` is listed in `.gitignore`.

Downloading it from Harvard Dataverse is step 1 of the full workflow. The download script always checks each file against the size that Dataverse reports. With `--verify-md5`, as used in the workflow above, it also verifies the MD5 checksums that Dataverse provides. A partial or corrupted download is therefore caught instead of being silently accepted.

The processed files in `data/processed/` are small, about 3 MB in total. They are included on purpose, so that the notebook runs without the raw data.

## 12. References

The sources used in this project are cited in section 18.3 of the analysis notebook. Their DOIs were verified against Crossref, arXiv or DataCite rather than written from memory.