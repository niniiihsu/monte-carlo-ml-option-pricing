"""
data_utils.py

Loads the option dataset, splits it into train/validation/test sets, and
standardizes the NN input columns (S, K, T, sigma, r).

The five inputs are on very different scales (S is in the hundreds, sigma
and r are small decimals), so standardizing each column to mean 0 / std 1
matters a lot for training a small feedforward net.
"""

import numpy as np
import pandas as pd

INPUT_COLUMNS = ["S", "K", "T", "sigma", "r"]
TARGET_COLUMN = "mc_price"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    missing = [col for col in INPUT_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def split_dataset(df, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Shuffles the dataframe and splits it into train/val/test.
    train_frac + val_frac + (leftover as test) should sum to 1.0.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val:]

    return train_df, val_df, test_df


class Standardizer:
    """
    Stores the mean/std of each input column from the training set, and
    applies the same transform to any other split. Fitting only on the
    training set (not val/test) avoids leaking information.
    """

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        # guard against a zero std for a constant column
        for i in range(len(self.stds)):
            if self.stds[i] == 0:
                self.stds[i] = 1.0

    def transform(self, X):
        return (X - self.means) / self.stds


def prepare_arrays(df, standardizer=None, fit=False):
    """
    Pulls out the input/target arrays from a dataframe split and applies
    standardization to the inputs. If fit=True, the standardizer learns
    its mean/std from this split (only do this on the training split).
    """
    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float64).reshape(-1, 1)

    if standardizer is None:
        standardizer = Standardizer()

    if fit:
        standardizer.fit(X)

    X_scaled = standardizer.transform(X)
    return X_scaled, y, standardizer
