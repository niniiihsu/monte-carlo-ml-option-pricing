"""
evaluate.py

Loads the trained model and the held-out test split (both saved by
train.py), then reports the metrics this section owns:
    - MAE, RMSE, R^2 (accuracy vs the Monte Carlo labels)
    - inference latency (ms per single prediction, averaged)
    - parameter count and model size on disk

Writes one row into results.csv using the shared benchmarking schema
from the README so Ishika can drop it straight into the comparison
table:
    method,mae,rmse,r2,latency_ms,throughput,data_movement_bytes,energy_estimate

data_movement_bytes and energy_estimate are left blank here since those
come from Raya's IMC mapping, not the plain NN inference measurement.
"""

import os
import time
import csv

import numpy as np

from data_utils import load_dataset, prepare_arrays, Standardizer
from model import OptionPriceNN


def compute_metrics(y_pred, y_true):
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return mae, rmse, r2


def measure_inference_latency(model, X, num_repeats=100):
    """
    Times single-sample inference, averaged over many repeats, since one
    call is too fast to measure reliably on its own.
    """
    sample = X[0:1]

    # warm-up call, first call sometimes has extra overhead
    model.predict(sample)

    start = time.perf_counter()
    for _ in range(num_repeats):
        model.predict(sample)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / num_repeats) * 1000.0
    return avg_latency_ms


def evaluate(model_path="best_model.npz", preprocessing_path="preprocessing_state.npz",
             test_data_path="test_split.csv", results_path="results.csv"):

    model = OptionPriceNN()
    model.load(model_path)

    prep_state = np.load(preprocessing_path)
    standardizer = Standardizer()
    standardizer.means = prep_state["means"]
    standardizer.stds = prep_state["stds"]

    test_df = load_dataset(test_data_path)
    X_test, y_test, _ = prepare_arrays(test_df, standardizer=standardizer)

    y_pred = model.predict(X_test)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)

    latency_ms = measure_inference_latency(model, X_test)
    throughput = 1000.0 / latency_ms  # predictions per second, single-sample

    param_count = model.param_count()
    model_size_bytes = os.path.getsize(model_path)

    print("=== Neural Network Surrogate: Evaluation ===")
    print(f"Test set size:        {len(test_df)}")
    print(f"MAE:                  {mae:.4f}")
    print(f"RMSE:                 {rmse:.4f}")
    print(f"R^2:                  {r2:.4f}")
    print(f"Inference latency:    {latency_ms:.4f} ms/sample")
    print(f"Throughput:           {throughput:.1f} samples/sec")
    print(f"Parameter count:      {param_count}")
    print(f"Model size on disk:   {model_size_bytes} bytes")

    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "method", "mae", "rmse", "r2", "latency_ms",
                "throughput", "data_movement_bytes", "energy_estimate",
            ])
        writer.writerow([
            "neural_network_cpu", f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}",
            f"{latency_ms:.4f}", f"{throughput:.1f}", "", "",
        ])

    print(f"\nAppended results row to {results_path}")

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "latency_ms": latency_ms, "throughput": throughput,
        "param_count": param_count, "model_size_bytes": model_size_bytes,
    }


if __name__ == "__main__":
    evaluate()
