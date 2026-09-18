"""
train.py

Trains the OptionPriceNN surrogate on the option pricing dataset.

Usage:
    python train.py --data dummy_option_data.csv --epochs 200

Once Nini hands off the real Monte Carlo dataset, just point --data at
that CSV instead. Nothing else in the pipeline needs to change, since
this script only expects the agreed column names.
"""

import argparse
import time

import numpy as np

from data_utils import load_dataset, split_dataset, prepare_arrays
from model import OptionPriceNN


def iterate_batches(X, y, batch_size, rng):
    n = X.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def mse(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


def train_model(data_path, epochs=200, batch_size=32, learning_rate=0.001, seed=42):
    df = load_dataset(data_path)
    train_df, val_df, test_df = split_dataset(df, seed=seed)

    X_train, y_train, standardizer = prepare_arrays(train_df, fit=True)
    X_val, y_val, _ = prepare_arrays(val_df, standardizer=standardizer)
    X_test, y_test, _ = prepare_arrays(test_df, standardizer=standardizer)

    model = OptionPriceNN(input_size=X_train.shape[1], hidden_size=64, seed=seed)
    rng = np.random.default_rng(seed)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        for X_batch, y_batch in iterate_batches(X_train, y_train, batch_size, rng):
            _, cache = model.forward(X_batch)
            model.backward(cache, y_batch, learning_rate)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_loss = mse(train_pred, y_train)
        val_loss = mse(val_pred, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save("best_model.npz")

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

    training_time_s = time.time() - start_time
    print(f"\nTraining finished in {training_time_s:.1f}s. Best val MSE: {best_val_loss:.4f}")

    # save the standardizer stats and test split so evaluate.py uses the
    # exact same scaling and the exact same held-out test set
    np.savez(
        "preprocessing_state.npz",
        means=standardizer.means,
        stds=standardizer.stds,
    )
    test_df.to_csv("test_split.csv", index=False)

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dummy_option_data.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
