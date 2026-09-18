"""
generate_dummy_data.py

Creates a dummy dataset with the same columns Nini's real Monte Carlo
dataset will use. This lets the NN pipeline get built and tested without
waiting on the real data.

Columns: S,K,T,sigma,r,mc_price,moneyness,num_paths,mc_runtime_ms

The "mc_price" here is NOT from a real Monte Carlo simulation. It is a
rough Black-Scholes-style approximation with noise added, just so the NN
has something realistic-shaped to fit while we wait for the real labels.
"""

import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm


def black_scholes_call(S, K, T, sigma, r):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return price


def generate_dummy_dataset(num_rows, seed=42):
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(num_rows):
        S = rng.uniform(50, 200)
        moneyness = rng.uniform(0.7, 1.3)
        K = S / moneyness
        T = rng.uniform(0.05, 2.0)
        sigma = rng.uniform(0.10, 0.60)
        r = rng.uniform(0.00, 0.08)

        true_price = black_scholes_call(S, K, T, sigma, r)
        noise = rng.normal(0, 0.02 * max(true_price, 0.5))
        mc_price = max(true_price + noise, 0.0)

        num_paths = 10000
        mc_runtime_ms = rng.uniform(5, 50)

        rows.append([S, K, T, sigma, r, mc_price, moneyness, num_paths, mc_runtime_ms])

    columns = ["S", "K", "T", "sigma", "r", "mc_price", "moneyness", "num_paths", "mc_runtime_ms"]
    df = pd.DataFrame(rows, columns=columns)
    return df


if __name__ == "__main__":
    df = generate_dummy_dataset(1000)
    out_path = "dummy_option_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())
