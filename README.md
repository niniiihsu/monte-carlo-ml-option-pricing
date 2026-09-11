# Core 4 — In-Memory Acceleration of Monte Carlo and ML-Based Financial Workloads

**Team:** Nini Hsu · Rupsaa Goswami · Raya Chauhan · Ishika Pandurangam (collaborative — no task is
exclusive to one person; the roles below are starting points, not fixed assignments)
**Theme:** 4 — In-Memory Computing and Circuits
**Final demo video due:** Sep 19, 2026

## Problem & Motivation

We want to make computationally intensive financial models more efficient in data movement,
latency, energy consumption, and (possibly) throughput — while maintaining pricing accuracy.
We use European call option pricing as the example workload, compare a traditional Monte Carlo
approach against a small neural-network surrogate, then model how the neural network's
matrix-vector computations could map onto an in-memory (IMC) / near-memory computing
architecture.

**Note:** this project does *not* predict real stock prices (e.g. Apple, NVIDIA). The dataset is
synthetic, hypothetical European call-option scenarios, chosen to keep the focus on computational
workload acceleration rather than stock forecasting.

## Pipeline

```
Synthetic Option Scenarios → Monte Carlo Pricing → Training Dataset
    → Neural-Network Surrogate → IMC Mapping → Performance Comparison
```

## Current Status (as of Sep 11, 2026)

| Component | Status |
|---|---|
| Monte Carlo pricer + dataset | Done — 10,000-scenario dataset generated, validated against Black-Scholes (~0.12% error) |
| Neural-network surrogate | In progress |
| IMC mapping | In progress |
| Benchmarking & comparison | Schema in progress; final numbers arrive as MC/NN/IMC results land |

## Repo Structure

```
core4-project/
├── monte_carlo/       # MC pricer, Black-Scholes validation, dataset generation
├── neural_network/     # NN preprocessing, training, evaluation
├── imc_mapping/        # IMC modeling, MAC counts, latency/energy estimates
├── benchmarking/        # benchmark schema, comparison tables, plots
├── data/               # Shared CSV datasets (dummy + final)
├── results/             # Final tables, figures, frozen results
└── docs/                # Project plan, proposal, schedule, completion plan
```

## Dataset Spec

Each row = one hypothetical European call option.

| Column | Meaning | Initial Range | Role |
|---|---|---|---|
| S | Current underlying price | $50–$200 | NN input |
| K | Strike price | covers moneyness 0.7–1.3 | NN input |
| T | Time to expiration (yrs) | 0.05–2.0 | NN input |
| sigma | Annualized volatility | 0.10–0.60 | NN input |
| r | Risk-free rate | 0.00–0.08 | NN input |
| mc_price | Monte Carlo call price | calculated | target |

Metadata columns (not NN inputs): `moneyness`, `num_paths`, `mc_runtime_ms`.

CSV header:
```
S,K,T,sigma,r,mc_price,moneyness,num_paths,mc_runtime_ms
```

Dev dataset: 500–1,000 rows. Final target: ~20,000–100,000 scenarios.

## Monte Carlo Reference Model

GBM terminal price:
```
S_T = S * exp[(r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z],   Z ~ N(0,1)
```
Payoff: `max(S_T - K, 0)`
Price estimate: `C_MC = exp(-r*T) * average(payoff)`

**Validation case:** S=100, K=100, T=1, sigma=0.20, r=0.05 → should approach Black-Scholes ≈ $10.45.

## Common Result Table Schema

```
method,mae,rmse,r2,latency_ms,throughput,data_movement_bytes,energy_estimate
```

## Open Tasks (not assigned to one person — pick up whatever's open)

- **Monte Carlo** — pricer + BS validation + dataset generation: **done**, open to refinement (e.g. expanding dataset, edge-case validation)
- **Neural network** — pipeline (start 5→64→64→1), training, MAE/RMSE/R-squared/latency/size metrics
- **IMC mapping** — modeling approach, MAC counts, data movement, latency/energy estimates
- **Benchmarking** — schema, results aggregation, plots, final comparison + demo coordination

See `docs/core4_completion_plan.pdf` for the full milestone-based plan and a detailed,
non-exclusive repo blueprint.

## Key Dependencies

```
MC + dataset ───────────────→ NN training
Agreed NN architecture ───→ IMC mapping
MC + NN + IMC results ─────→ Final comparison
```

## Hard Deadlines

| Date | Milestone |
|---|---|
| Sep 13 | End-to-end pipeline works |
| Sep 16 | Results frozen |
| Sep 18 | Demo video finished |
| Sep 19 | Submit |

## Scope Rules (MVP)

- European call options only
- Synthetic scenarios, not historical stock prediction
- One small NN before trying larger architectures
- Software/simulation-based IMC — no physical chip required
- No new features after Sep 16 unless critical
- Real stock example is optional/demo-only, not the main dataset

See `docs/` for the full project plan, one-page proposal, materials checklist, and completion plan.
