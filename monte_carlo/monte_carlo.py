"""
Monte Carlo European call option pricing and synthetic dataset generation.

Core 4 - Chips & AI Hackathon
Contributors: Nini Hsu (Monte Carlo pricer, dataset generation),
Ishika Pandurangam (module refactor, documentation)

This module implements:
    1. A single-scenario Monte Carlo European call pricer
       (`monte_carlo_call_price`), used for validation against Black-Scholes.
    2. A vectorized, batched dataset generator (`generate_mc_dataset`) that
       produces the synthetic training dataset consumed by the neural-network
       surrogate (Rupsaa) and, downstream, the IMC mapping analysis (Raya).

Pricing model
-------------
Terminal stock price under risk-neutral geometric Brownian motion:

    S_T = S * exp[(r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z],   Z ~ N(0, 1)

European call payoff and Monte Carlo price estimate:

    payoff = max(S_T - K, 0)
    C_MC   = exp(-r * T) * mean(payoff)
"""

from pathlib import Path
import csv
import time

import numpy as np

# ---------- Project paths ----------
# Resolved relative to this file so the repo works on any machine/checkout,
# rather than the hardcoded local path used during initial development.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = PROJECT_DIR / "core4_mc_dataset_10000.csv"


def monte_carlo_call_price(S, K, T, sigma, r, num_paths=100_000, seed=42):
    """
    Price a single European call option via Monte Carlo simulation.

    Simulates `num_paths` terminal stock prices under geometric Brownian
    motion, computes the call payoff for each, and returns the discounted
    average payoff.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration, in years.
    sigma : float
        Annualized volatility.
    r : float
        Risk-free rate.
    num_paths : int, default 100_000
        Number of simulated price paths.
    seed : int, default 42
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    float
        Monte Carlo estimate of the European call option price.

    Notes
    -----
    Validation case: S=100, K=100, T=1, sigma=0.20, r=0.05 should converge
    to the Black-Scholes closed-form price of ~$10.4506 as num_paths grows.
    With num_paths=200_000 this implementation returns ~$10.4634
    (~0.12% relative error) - see README for the full validation writeup.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(num_paths)

    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)

    return float(np.exp(-r * T) * payoffs.mean())


def generate_mc_dataset(
    output_csv=DEFAULT_CSV_PATH,
    n_scenarios=10_000,
    num_paths=20_000,
    scenario_seed=20260909,
    mc_seed=12345,
    batch_size=100,
):
    """
    Generate a synthetic dataset of European call option scenarios, each
    priced via Monte Carlo simulation, and write it to CSV.

    Sampling
    --------
        K          ~ Uniform(50, 200)
        moneyness  ~ Uniform(0.7, 1.3)     (S/K, covers OTM/ATM/ITM)
        S          =  K * moneyness
        T          ~ Uniform(0.05, 2.0)    (years)
        sigma      ~ Uniform(0.10, 0.60)
        r          ~ Uniform(0.00, 0.08)

    Note this samples K directly and derives S from moneyness, so the
    resulting S values range roughly $35-$260 rather than a flat
    Uniform(50, 200) on S itself - moneyness coverage was prioritized over
    a fixed S range.

    Pricing is vectorized and batched: within each batch of `batch_size`
    scenarios, all Monte Carlo paths for all scenarios in the batch are
    simulated as a single (batch_size x num_paths) array, which is far
    faster than pricing one scenario at a time in Python.

    Parameters
    ----------
    output_csv : str or Path, default DEFAULT_CSV_PATH
        Destination CSV path.
    n_scenarios : int, default 10_000
        Number of option scenarios to generate.
    num_paths : int, default 20_000
        Number of Monte Carlo paths used to price each scenario.
    scenario_seed : int, default 20260909
        Seed for sampling scenario parameters (S, K, T, sigma, r).
    mc_seed : int, default 12345
        Seed for the Monte Carlo path simulation.
    batch_size : int, default 100
        Number of scenarios priced together per vectorized batch.

    Returns
    -------
    dict
        Summary with keys: `file`, `rows`, `num_paths_per_scenario`,
        `runtime_seconds`.

    Output CSV columns
    -------------------
    scenario_id, S, K, T, sigma, r, moneyness, num_paths, mc_runtime_ms, mc_price

    `mc_runtime_ms` is the batch's total pricing time divided by the batch
    size (an average per-scenario cost), not each scenario's individually
    measured runtime - batching makes true per-scenario timing meaningless
    at this granularity, but the average is a fair figure for benchmarking
    throughput.
    """
    scenario_rng = np.random.default_rng(scenario_seed)
    mc_rng = np.random.default_rng(mc_seed)

    K_all = scenario_rng.uniform(50.0, 200.0, n_scenarios)
    m_all = scenario_rng.uniform(0.7, 1.3, n_scenarios)
    S_all = K_all * m_all
    T_all = scenario_rng.uniform(0.05, 2.0, n_scenarios)
    sigma_all = scenario_rng.uniform(0.10, 0.60, n_scenarios)
    r_all = scenario_rng.uniform(0.00, 0.08, n_scenarios)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    rows_written = 0

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "S", "K", "T", "sigma", "r",
            "moneyness", "num_paths", "mc_runtime_ms", "mc_price",
        ])

        for start_idx in range(0, n_scenarios, batch_size):
            end_idx = min(start_idx + batch_size, n_scenarios)
            b = end_idx - start_idx

            S = S_all[start_idx:end_idx]
            K = K_all[start_idx:end_idx]
            T = T_all[start_idx:end_idx]
            sigma = sigma_all[start_idx:end_idx]
            r = r_all[start_idx:end_idx]
            m = m_all[start_idx:end_idx]

            batch_t0 = time.perf_counter()

            # Each row = one option scenario, each column = one MC path
            Z = mc_rng.standard_normal((b, num_paths))
            ST = S[:, None] * np.exp(
                (r[:, None] - 0.5 * sigma[:, None] ** 2) * T[:, None]
                + sigma[:, None] * np.sqrt(T[:, None]) * Z
            )
            payoffs = np.maximum(ST - K[:, None], 0.0)
            prices = np.exp(-r * T) * payoffs.mean(axis=1)

            batch_ms = (time.perf_counter() - batch_t0) * 1000
            approx_per_scenario_ms = batch_ms / b

            for j in range(b):
                idx = start_idx + j
                writer.writerow([
                    idx + 1,
                    round(float(S[j]), 6),
                    round(float(K[j]), 6),
                    round(float(T[j]), 6),
                    round(float(sigma[j]), 6),
                    round(float(r[j]), 6),
                    round(float(m[j]), 6),
                    num_paths,
                    round(float(approx_per_scenario_ms), 6),
                    round(float(prices[j]), 6),
                ])

            rows_written += b

    total_sec = time.perf_counter() - start

    return {
        "file": str(output_csv),
        "rows": rows_written,
        "num_paths_per_scenario": num_paths,
        "runtime_seconds": total_sec,
    }


if __name__ == "__main__":
    summary = generate_mc_dataset(
        DEFAULT_CSV_PATH, n_scenarios=10_000, num_paths=20_000, batch_size=100,
    )

    test_price = monte_carlo_call_price(
        S=100, K=100, T=1, sigma=0.20, r=0.05, num_paths=200_000, seed=42,
    )

    print(summary)
    print("Test price:", test_price)
    print("CSV saved to:", DEFAULT_CSV_PATH)
