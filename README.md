# Monte Carlo & ML Option Pricing

Part of **Core 4**'s Chips & AI Hackathon project: *In-Memory Acceleration of Monte Carlo and
ML-Based Financial Workloads* (Theme 4 — In-Memory Computing and Circuits).

This repo is the Monte Carlo pricing and dataset-generation component, owned by **Nini Hsu**.
It produces the synthetic training dataset that feeds the neural-network surrogate (Rupsaa) and,
downstream, the IMC mapping analysis (Raya).

## What's in here

| File | Description |
|---|---|
| `monte_carlo.py` | Reusable module: single-scenario pricer + batched dataset generator (docstrings included) |
| `monte_carlo.ipynb` | Notebook that runs the generator and prints a validation check |
| `core4_mc_dataset_10000.csv` | Generated dataset: 10,000 synthetic European call scenarios with Monte Carlo prices |
| `requirements.txt` | Dependencies |

## Method

Terminal stock price simulated under risk-neutral geometric Brownian motion:

```
S_T = S · exp[(r − ½σ²)T + σ√T · Z],   Z ~ N(0,1)
```

European call payoff and Monte Carlo price estimate:

```
Payoff = max(S_T − K, 0)
C_MC = exp(−rT) · average[Payoff]
```

## Validation

Test case: **S=100, K=100, T=1, σ=0.20, r=0.05**, priced with 200,000 Monte Carlo paths.

| | Price |
|---|---|
| Black-Scholes closed form | $10.4506 |
| Monte Carlo (this implementation) | $10.4634 |
| Relative error | ~0.12% |

This is within expected Monte Carlo sampling noise at 200k paths and confirms the simulation is
implemented correctly before generating the full dataset.

## Dataset

`generate_mc_dataset()` produces `core4_mc_dataset_10000.csv`: 10,000 scenarios, each priced with
20,000 Monte Carlo paths, generated in ~4 seconds on a single machine (vectorized/batched in
groups of 100 scenarios at a time).

**Sampling:**

```
K          ~ Uniform(50, 200)
moneyness  ~ Uniform(0.7, 1.3)      # S/K, so scenarios cover OTM, ATM, and ITM
S          = K * moneyness
T          ~ Uniform(0.05, 2.0)     # years
sigma      ~ Uniform(0.10, 0.60)
r          ~ Uniform(0.00, 0.08)
```

Note: S is *derived* from K and moneyness rather than sampled directly, so the actual S range
ends up wider than a flat $50–$200 — in the generated dataset it's about **$36–$258**. This was a
deliberate tradeoff to guarantee even coverage of moneyness (in-the-money vs out-of-the-money)
rather than a fixed S range.

**Output CSV columns:**

```
scenario_id,S,K,T,sigma,r,moneyness,num_paths,mc_runtime_ms,mc_price
```

**Actual ranges in the generated 10,000-row dataset:**

| Column | Min | Max |
|---|---|---|
| S | 36.05 | 258.01 |
| K | 50.00 | 199.94 |
| moneyness | 0.700 | 1.300 |
| T | 0.050 | 2.000 |
| sigma | 0.100 | 0.600 |
| r | 0.00001 | 0.080 |
| mc_price | 0.00 | 108.48 |

`mc_runtime_ms` is the *average* per-scenario cost within its batch (batch wall-clock time ÷
batch size — here batches of 100 scenarios average **~0.36 ms/scenario** at 20,000 paths each),
not an individually measured time per scenario. Batches are priced as one vectorized NumPy
operation, so true per-scenario timing isn't meaningful at that granularity — the batch average
is the fair number to use for throughput comparisons downstream.

## Reproducibility

Both the scenario sampling and the Monte Carlo path simulation use fixed seeds
(`scenario_seed=20260909`, `mc_seed=12345` for the dataset; `seed=42` for the single-scenario
validation test), so re-running `generate_mc_dataset()` reproduces the exact same CSV.

## Usage

```bash
pip install -r requirements.txt

# Regenerate the dataset + run the validation check
python monte_carlo.py

# Or import the functions directly
python -c "
from monte_carlo import monte_carlo_call_price
print(monte_carlo_call_price(S=100, K=100, T=1, sigma=0.20, r=0.05, num_paths=200_000))
"
```

The notebook (`monte_carlo.ipynb`) runs the same code interactively.

## Handoff

`core4_mc_dataset_10000.csv` is the agreed dataset format passed to the neural-network surrogate
for training (`S, K, T, sigma, r` as inputs, `mc_price` as the training target), and ultimately
feeds the IMC mapping and final benchmarking/comparison stages of the Core 4 pipeline.

## Requirements

```
numpy
pandas
jupyter
```
