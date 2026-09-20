"""
imc_model.py

IMC (in-memory computing) mapping for the OptionPriceNN surrogate
(5 -> 64 -> 64 -> 1, see neural_network/model.py).

What this does
---------------
1. Maps each dense layer to a matrix-vector operation y = Wx + b and
   counts MACs (multiply-accumulates) and memory accesses.
2. Estimates data movement (bytes moved between compute and memory) for
   two architectures:
     - "conventional": a standard von Neumann setup where weights live
       in DRAM/off-chip memory and must be fetched into the compute
       unit for every inference.
     - "imc": weights are stored once in a memory array (e.g. a
       resistive/SRAM crossbar) and MACs are performed in place, so
       weights never move. Only activations move between layers.
3. Converts MACs + data movement into energy and latency estimates
   using published, commonly-cited order-of-magnitude numbers for
   compute and memory-access energy (see ASSUMPTIONS below).
4. Appends result rows to results/results.csv using the project's
   shared schema:
       method,mae,rmse,r2,latency_ms,throughput,data_movement_bytes,energy_estimate

This is a software/simulation-level model, not a physical chip
characterization -- consistent with the project's MVP scope ("no
physical chip required"). Every number that isn't a direct count
(MACs, elements, bytes) is a documented estimate, flagged as such in
the printed report and in imc_mapping/ASSUMPTIONS.md.

Usage:
    python imc_model.py
    python imc_model.py --model ../neural_network/best_model.npz
"""

import argparse
import csv
import os
import time

import numpy as np

# make the neural_network module importable when run from imc_mapping/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neural_network"))

BYTES_PER_FLOAT32 = 4

# ---------------------------------------------------------------------
# ASSUMPTIONS (documented, order-of-magnitude energy numbers commonly
# used in energy-efficient-computing / IMC literature, e.g. Horowitz,
# "Computing's Energy Problem", ISSCC 2014, and follow-on accelerator
# papers such as Eyeriss). These are estimates, not measurements from
# real silicon, and are meant to show the *shape* of the conventional
# vs. IMC tradeoff, not an exact chip's numbers.
# ---------------------------------------------------------------------
ASSUMPTIONS = {
    # energy per 32-bit MAC done by a conventional digital ALU
    # (~3.7 pJ multiply + ~0.9 pJ add for 32-bit float)
    "e_mac_conventional_pJ": 4.6,
    # energy per 32-bit MAC performed in an analog/mixed-signal
    # compute-in-memory array (net of ADC/DAC overhead). Reported CIM
    # papers vary widely (roughly 5x-50x better than digital MAC+fetch
    # combined); we use a conservative ~4.6x reduction here.
    "e_mac_imc_pJ": 1.0,
    # energy to move one byte from off-chip DRAM to the compute unit
    # (~640 pJ per 32-bit word / 4 bytes = 160 pJ/byte)
    "e_dram_byte_pJ": 160.0,
    # energy to move one byte through on-chip SRAM / local buffers
    # (~5 pJ per 32-bit word / 4 bytes = 1.25 pJ/byte). Used for
    # activation movement in BOTH architectures, since even IMC still
    # has to shuttle activations between layer tiles.
    "e_sram_byte_pJ": 1.25,
    # off-chip memory bandwidth used to turn "conventional" weight
    # fetch bytes into a latency estimate (single DDR4 channel, ~GB/s)
    "dram_bandwidth_GBps": 25.6,
    # on-chip bandwidth used for activation movement, both architectures
    "sram_bandwidth_GBps": 200.0,
    # IMC array throughput: MACs per second the crossbar can retire
    # once weights are resident (order-of-magnitude estimate for a
    # small array at modest precision)
    "imc_macs_per_second": 5.0e9,
}


def load_layer_shapes(model_path=None):
    """
    Returns the list of (in_dim, out_dim) for each dense layer.
    If a trained model file exists, loads the real weight shapes from
    it; otherwise falls back to the architecture documented in the
    README / model.py (5 -> 64 -> 64 -> 1).
    """
    if model_path and os.path.exists(model_path):
        data = np.load(model_path)
        shapes = [data["W1"].shape, data["W2"].shape, data["W3"].shape]
        source = f"loaded from {model_path}"
    else:
        from model import OptionPriceNN
        m = OptionPriceNN(input_size=5, hidden_size=64, seed=42)
        shapes = [m.W1.shape, m.W2.shape, m.W3.shape]
        source = "default architecture (5 -> 64 -> 64 -> 1), no trained model file found"
    return shapes, source


def analyze_layers(shapes):
    """
    For each dense layer y = Wx + b, computes:
      - MACs (= in_dim * out_dim, one multiply-add per weight element
        for a single sample / batch size 1)
      - weight elements (+ 1 bias per output unit)
      - activation elements flowing out of the layer
    """
    layers = []
    for i, (in_dim, out_dim) in enumerate(shapes, start=1):
        macs = in_dim * out_dim
        weight_elems = in_dim * out_dim
        bias_elems = out_dim
        layers.append({
            "layer": i,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "macs": macs,
            "weight_elems": weight_elems,
            "bias_elems": bias_elems,
            "activation_out_elems": out_dim,
        })
    return layers


def data_movement(layers, input_dim):
    """
    Estimates per-inference (batch size = 1) data movement in bytes for
    the conventional and IMC architectures.

    Conventional: every inference re-fetches all weights + biases from
    off-chip memory (worst-case / edge-streaming assumption: no weight
    caching across inferences -- see ASSUMPTIONS.md for the alternative
    "weights cached on-chip" case). Activations move between the
    compute unit and local buffers at every layer boundary (write then
    read).

    IMC: weights are resident in the memory array and never move.
    Activations still have to move between layer tiles (write + read),
    same as conventional, since inter-layer values leave one array and
    enter the next.
    """
    weight_bytes_total = sum(
        (l["weight_elems"] + l["bias_elems"]) * BYTES_PER_FLOAT32 for l in layers
    )

    # activation traffic: input in, then write+read at each hidden
    # layer boundary, then final output out
    activation_bytes = input_dim * BYTES_PER_FLOAT32  # read input
    for l in layers[:-1]:
        # write this layer's output to a buffer, then read it back in
        # as the next layer's input
        activation_bytes += 2 * l["activation_out_elems"] * BYTES_PER_FLOAT32
    activation_bytes += layers[-1]["activation_out_elems"] * BYTES_PER_FLOAT32  # write final output

    conventional_bytes = weight_bytes_total + activation_bytes
    imc_bytes = activation_bytes  # weights never move

    return {
        "weight_bytes_total": weight_bytes_total,
        "activation_bytes": activation_bytes,
        "conventional_bytes": conventional_bytes,
        "imc_bytes": imc_bytes,
    }


def energy_estimate(total_macs, movement, assumptions):
    """
    Energy (in picojoules) for one single-sample inference under both
    architectures, split into compute energy and data-movement energy
    so the two contributions can be reported separately.
    """
    conv_compute_pJ = total_macs * assumptions["e_mac_conventional_pJ"]
    conv_weight_move_pJ = movement["weight_bytes_total"] * assumptions["e_dram_byte_pJ"]
    conv_activation_move_pJ = movement["activation_bytes"] * assumptions["e_sram_byte_pJ"]
    conv_total_pJ = conv_compute_pJ + conv_weight_move_pJ + conv_activation_move_pJ

    imc_compute_pJ = total_macs * assumptions["e_mac_imc_pJ"]
    imc_activation_move_pJ = movement["activation_bytes"] * assumptions["e_sram_byte_pJ"]
    imc_total_pJ = imc_compute_pJ + imc_activation_move_pJ

    return {
        "conventional": {
            "compute_pJ": conv_compute_pJ,
            "weight_movement_pJ": conv_weight_move_pJ,
            "activation_movement_pJ": conv_activation_move_pJ,
            "total_pJ": conv_total_pJ,
        },
        "imc": {
            "compute_pJ": imc_compute_pJ,
            "activation_movement_pJ": imc_activation_move_pJ,
            "total_pJ": imc_total_pJ,
        },
    }


def latency_estimate(total_macs, movement, assumptions, measured_cpu_latency_ms=None):
    """
    Analytical latency estimate (ms) for both architectures, modeled as
    the slower of (data movement / bandwidth) and (compute / throughput)
    -- i.e. whichever resource is the bottleneck, not the sum of both,
    since real hardware overlaps compute with the *next* transfer.

    Also reports the actual measured wall-clock latency of the NumPy
    forward pass for reference: since the model is tiny and NumPy's
    BLAS keeps the weights cache-resident, that measurement is closer
    to a "weights already cached on-chip" scenario than to the
    cold-DRAM-fetch scenario the conventional estimate below models.
    """
    dram_Bps = assumptions["dram_bandwidth_GBps"] * 1e9
    sram_Bps = assumptions["sram_bandwidth_GBps"] * 1e9
    imc_macs_per_s = assumptions["imc_macs_per_second"]

    conv_weight_fetch_s = movement["weight_bytes_total"] / dram_Bps
    conv_activation_s = movement["activation_bytes"] / sram_Bps
    conv_compute_s = total_macs / (2.0e9)  # ~2 GMAC/s for unvectorized scalar compute
    conv_latency_s = max(conv_weight_fetch_s, conv_compute_s) + conv_activation_s

    imc_compute_s = total_macs / imc_macs_per_s
    imc_activation_s = movement["activation_bytes"] / sram_Bps
    imc_latency_s = imc_compute_s + imc_activation_s  # weights resident, no fetch

    result = {
        "conventional_latency_ms": conv_latency_s * 1000.0,
        "imc_latency_ms": imc_latency_s * 1000.0,
    }
    if measured_cpu_latency_ms is not None:
        result["measured_numpy_cpu_latency_ms"] = measured_cpu_latency_ms
    return result


def measure_numpy_latency(model_path=None, repeats=2000):
    from model import OptionPriceNN
    m = OptionPriceNN(input_size=5, hidden_size=64, seed=42)
    if model_path and os.path.exists(model_path):
        m.load(model_path)
    X = np.zeros((1, 5))
    m.predict(X)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        m.predict(X)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def write_results_row(results_path, method, latency_ms, throughput, data_movement_bytes, energy_estimate_nJ):
    file_exists = os.path.isfile(results_path)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "method", "mae", "rmse", "r2", "latency_ms",
                "throughput", "data_movement_bytes", "energy_estimate",
            ])
        # mae/rmse/r2 are the NN accuracy team's fields, not ours -> blank
        writer.writerow([
            method, "", "", "",
            f"{latency_ms:.6f}", f"{throughput:.1f}",
            f"{data_movement_bytes:.0f}", f"{energy_estimate_nJ:.4f}",
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../neural_network/best_model.npz",
                         help="Path to a trained OptionPriceNN .npz file (optional)")
    parser.add_argument("--results", type=str, default="../results/results.csv",
                         help="Shared results CSV to append rows to")
    parser.add_argument("--input_dim", type=int, default=5)
    args = parser.parse_args()

    shapes, source = load_layer_shapes(args.model)
    layers = analyze_layers(shapes)
    total_macs = sum(l["macs"] for l in layers)
    total_weight_elems = sum(l["weight_elems"] + l["bias_elems"] for l in layers)

    movement = data_movement(layers, args.input_dim)
    energy = energy_estimate(total_macs, movement, ASSUMPTIONS)

    measured_latency_ms = measure_numpy_latency(args.model if os.path.exists(args.model) else None)
    latency = latency_estimate(total_macs, movement, ASSUMPTIONS, measured_latency_ms)

    # ---- report ----
    print("=== IMC Mapping: OptionPriceNN (single-sample inference) ===")
    print(f"Architecture source: {source}")
    print()
    print("Per-layer MAC / element counts (y = Wx + b):")
    for l in layers:
        print(f"  Layer {l['layer']}: {l['in_dim']:>3} -> {l['out_dim']:<3}  "
              f"MACs={l['macs']:>6}  weight_elems={l['weight_elems']:>6}  bias_elems={l['bias_elems']}")
    print(f"  TOTAL MACs: {total_macs}")
    print(f"  TOTAL weight+bias elements: {total_weight_elems}")
    print()
    print("Data movement per inference (bytes, FP32):")
    print(f"  Weight+bias bytes (must be fetched every inference, conventional): {movement['weight_bytes_total']:.0f}")
    print(f"  Activation bytes (both architectures):                             {movement['activation_bytes']:.0f}")
    print(f"  Conventional TOTAL:                                                {movement['conventional_bytes']:.0f}")
    print(f"  IMC TOTAL (weights resident, never moved):                        {movement['imc_bytes']:.0f}")
    reduction = 1 - movement["imc_bytes"] / movement["conventional_bytes"]
    print(f"  -> data movement reduction from IMC: {reduction*100:.1f}%")
    print()
    print("Energy per inference (documented order-of-magnitude estimates, see ASSUMPTIONS.md):")
    print(f"  Conventional: compute={energy['conventional']['compute_pJ']:.1f} pJ  "
          f"weight_movement={energy['conventional']['weight_movement_pJ']:.1f} pJ  "
          f"activation_movement={energy['conventional']['activation_movement_pJ']:.1f} pJ  "
          f"TOTAL={energy['conventional']['total_pJ']:.1f} pJ "
          f"({energy['conventional']['total_pJ']/1000:.2f} nJ)")
    print(f"  IMC:          compute={energy['imc']['compute_pJ']:.1f} pJ  "
          f"activation_movement={energy['imc']['activation_movement_pJ']:.1f} pJ  "
          f"TOTAL={energy['imc']['total_pJ']:.1f} pJ "
          f"({energy['imc']['total_pJ']/1000:.2f} nJ)")
    energy_reduction = 1 - energy["imc"]["total_pJ"] / energy["conventional"]["total_pJ"]
    print(f"  -> energy reduction from IMC: {energy_reduction*100:.1f}%")
    print()
    print("Latency per inference:")
    print(f"  Measured NumPy forward pass (weights cache-resident on this machine): {latency['measured_numpy_cpu_latency_ms']:.5f} ms")
    print(f"  Modeled conventional (cold DRAM weight fetch, worst case):            {latency['conventional_latency_ms']:.5f} ms")
    print(f"  Modeled IMC (weights resident in array):                             {latency['imc_latency_ms']:.5f} ms")
    print()
    print("NOTE: all numbers above except the raw MAC/element/byte counts and the")
    print("measured NumPy latency are simulated estimates based on documented,")
    print("literature-derived assumptions (see ASSUMPTIONS.md), not measurements")
    print("from physical silicon. This is a software-level IMC proof of concept,")
    print("consistent with the project's MVP scope.")

    # ---- write rows to shared results.csv ----
    conv_latency_ms = latency["conventional_latency_ms"]
    conv_throughput = 1000.0 / conv_latency_ms
    write_results_row(
        args.results, "neural_network_conventional_datamovement",
        conv_latency_ms, conv_throughput,
        movement["conventional_bytes"], energy["conventional"]["total_pJ"] / 1000.0,
    )

    imc_latency_ms = latency["imc_latency_ms"]
    imc_throughput = 1000.0 / imc_latency_ms
    write_results_row(
        args.results, "neural_network_imc",
        imc_latency_ms, imc_throughput,
        movement["imc_bytes"], energy["imc"]["total_pJ"] / 1000.0,
    )
    print(f"\nAppended 2 rows to {args.results}")


if __name__ == "__main__":
    main()
