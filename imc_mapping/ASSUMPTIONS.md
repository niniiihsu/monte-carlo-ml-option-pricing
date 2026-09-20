# IMC Mapping — Methodology & Assumptions

This is a **software/simulation-level** model of how the `OptionPriceNN`
surrogate (5 → 64 → 64 → 1, see `neural_network/model.py`) would map onto
a conventional (von Neumann) architecture vs. an in-memory-computing
(IMC) architecture. No physical chip is built or required — that's in
scope per the project's MVP rules. Every number below is either a direct
count (MACs, elements, bytes) or a clearly labeled estimate.

## 1. What's counted directly (not estimated)

- **MACs per layer** = `in_dim * out_dim`, one multiply-accumulate per
  weight in the `y = Wx + b` matrix-vector product, for a single sample.
  Total for this architecture: 5·64 + 64·64 + 64·1 = **4,480 MACs**.
- **Weight/bias elements**: 4,480 weights + 129 biases = **4,609
  parameters**, matching `model.param_count()`.
- **Bytes moved**: every element above is stored as FP32 (4 bytes), so
  byte counts are just `elements * 4`.

## 2. The two architectures being compared

- **Conventional**: weights live in off-chip DRAM. Every inference
  re-fetches all weights + biases into the compute unit (worst-case /
  edge-streaming assumption — no weight caching across calls). This is
  the pessimistic end of "conventional"; see the caveat in §4.
- **IMC**: weights are written into a memory array once (e.g. a
  resistive or SRAM crossbar) and the multiply-accumulate happens
  *inside* the array. Weights never move again. Only activations still
  have to move between layer tiles (a real limitation of multi-layer
  IMC — it isn't magic, only the weight-fetch cost disappears).

## 3. Estimated numbers, and where they come from

All energy figures are order-of-magnitude numbers commonly cited in
energy-efficient-computing / compute-in-memory papers, most traceable to
Mark Horowitz's "Computing's Energy Problem" (ISSCC 2014) keynote, which
is the standard reference this class of back-of-envelope model uses:

| Quantity | Value used | Rationale |
|---|---|---|
| Energy / 32-bit MAC, conventional digital ALU | 4.6 pJ | ~3.7 pJ multiply + ~0.9 pJ add (Horowitz) |
| Energy / 32-bit MAC, IMC array | 1.0 pJ | ~4.6x lower than digital, conservative vs. some CIM papers claiming larger gains; accounts for ADC/DAC overhead |
| Energy / byte, DRAM access | 160 pJ | 640 pJ per 32-bit word (Horowitz) ÷ 4 bytes |
| Energy / byte, on-chip SRAM/buffer | 1.25 pJ | ~5 pJ per 32-bit word (Horowitz) ÷ 4 bytes |
| DRAM bandwidth | 25.6 GB/s | single DDR4 channel, typical |
| On-chip bandwidth | 200 GB/s | typical on-chip buffer/interconnect order of magnitude |
| IMC array throughput | 5 GMAC/s | order-of-magnitude estimate for a small crossbar at modest precision |

These are **not measurements from a fabricated chip** — they're the same
kind of literature-anchored estimate used in most software-level IMC
feasibility studies, and are flagged as such everywhere they appear in
`imc_model.py`'s output and in `results/results.csv`.

## 4. Headline results (single-sample inference)

| Metric | Conventional | IMC | Change |
|---|---|---|---|
| Data movement | 19,484 bytes | 1,048 bytes | **-94.6%** |
| Energy | ~2,971.7 nJ | ~5.79 nJ | **-99.8%** |
| Modeled latency | ~0.00225 ms | ~0.00090 ms | ~2.5x faster |

**Important caveat:** the huge energy/data-movement win comes almost
entirely from *not re-fetching 18 KB of weights from DRAM on every
single inference*. This is a realistic worst case for an
edge/streaming deployment (one sample at a time, no weight caching),
and it's exactly the scenario IMC is motivated by. If weights are
instead assumed to be cached in on-chip SRAM/registers across repeated
calls (realistic for a server doing batched inference), the
conventional-side weight-fetch cost is paid once and amortized, and the
IMC advantage shrinks accordingly — worth mentioning explicitly in the
demo rather than presenting the single-sample number as the *only*
story.

## 5. How this was produced

Run `python imc_mapping/imc_model.py` from the `imc_mapping/` folder
(optionally pointing `--model` at a trained `best_model.npz` once
Rupsaa's training run produces one — the script will read the actual
learned weight shapes instead of the default architecture, though the
shapes are the same regardless: 5→64→64→1). It appends two rows to
`results/results.csv`:

- `neural_network_conventional_datamovement` — fills in the
  `data_movement_bytes` / `energy_estimate` fields that
  `neural_network/evaluate.py` intentionally leaves blank for the
  `neural_network_cpu` row.
- `neural_network_imc` — the full IMC-side estimate, for the final MC
  vs. NN-CPU vs. NN+IMC comparison in `benchmarking/`.
