"""From-scratch MNIST activation comparison: Sigmoid, ReLU, Leaky ReLU, Mish,
and hybrid ReLU (epochs 1-5) -> Mish (later epochs).

Research-based implementation, not the paper authors' original source.
Input CSV: header with `label` and 784 pixel columns, or 785 columns with
label first and no header. Install: pip install numpy pandas
Run: python activation_functions_mnist.py --csv mnist_train.csv --epochs 20
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0, x)


def activate(z, name, leak=0.01):
    if name == "sigmoid":
        a = sigmoid(z)
        return a, a * (1 - a)
    if name == "relu":
        return np.maximum(z, 0), (z > 0).astype(z.dtype)
    if name == "leaky_relu":
        return np.where(z > 0, z, leak * z), np.where(z > 0, 1., leak)
    if name == "mish":
        sp = softplus(z)
        t = np.tanh(sp)
        a = z * t
        # d/dx [x tanh(softplus(x))] = tanh(sp) + x sigmoid(x) sech^2(sp)
        return a, t + z * sigmoid(z) * (1 - t * t)
    raise ValueError(name)


def softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def read_mnist_csv(path):
    with open(path, newline="") as f:
        first = next(csv.reader(f))
    header = any(not item.strip().replace(".", "", 1).isdigit() for item in first)
    data = np.loadtxt(path, delimiter=",", skiprows=1 if header else 0, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 785:
        raise ValueError(f"Expected 785 columns (label + 784 pixels), got {data.shape}")
    y = data[:, 0].astype(np.int64)
    if not np.all((0 <= y) & (y <= 9)):
        raise ValueError("MNIST labels should be integers 0..9")
    x = data[:, 1:] / 255.0
    return x, y


def split_data(x, y, seed=42):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(x))
    # 70/15/15 shuffle split; exact paper proportions were not specified.
    n_train, n_val = int(0.70 * len(x)), int(0.15 * len(x))
    train, val, test = ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]
    return (x[train], y[train]), (x[val], y[val]), (x[test], y[test])


class TwoLayerNet:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-0.5, 0.5, (784, 10))
        self.b1 = np.zeros(10)
        self.w2 = rng.uniform(-0.5, 0.5, (10, 10))
        self.b2 = np.zeros(10)

    def forward(self, x, activation, leak=0.01):
        z1 = x @ self.w1 + self.b1
        a1, deriv = activate(z1, activation, leak)
        probabilities = softmax(a1 @ self.w2 + self.b2)
        return a1, deriv, probabilities

    def train_batch(self, x, y, activation, lr, leak):
        a1, derivative, probs = self.forward(x, activation, leak)
        batch = len(y)
        dz2 = probs.copy()
        dz2[np.arange(batch), y] -= 1
        dz2 /= batch
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.w2.T) * derivative
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

    def evaluate(self, x, y, activation, leak):
        _, _, probabilities = self.forward(x, activation, leak)
        accuracy = float((np.argmax(probabilities, axis=1) == y).mean())
        loss = float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean())
        return {"accuracy": accuracy, "cross_entropy": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="MNIST CSV (label + 784 pixels)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--leak", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("activation_results.json"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    x, y = read_mnist_csv(args.csv)
    (xtr, ytr), (xval, yval), (xte, yte) = split_data(x, y, args.seed)
    results = {"settings": vars(args).copy(), "models": {}}
    results["settings"]["csv"] = str(args.csv)
    results["settings"]["output"] = str(args.output)
    for name in ("sigmoid", "relu", "leaky_relu", "mish", "hybrid_relu_mish"):
        net = TwoLayerNet(args.seed)  # identical initialization for fair comparison
        rng = np.random.default_rng(args.seed + 1)
        history = []
        for epoch in range(1, args.epochs + 1):
            act = ("relu" if epoch <= 5 else "mish") if name == "hybrid_relu_mish" else name
            for ids in np.array_split(rng.permutation(len(xtr)),
                                      max(1, int(np.ceil(len(xtr) / args.batch_size)))):
                net.train_batch(xtr[ids], ytr[ids], act, args.learning_rate, args.leak)
            metrics = net.evaluate(xval, yval, act, args.leak)
            history.append({"epoch": epoch, "activation": act, "validation": metrics})
            print(f"{name:20} epoch {epoch:2}/{args.epochs}: val_acc={metrics['accuracy']:.4%}")
        final_act = "mish" if name == "hybrid_relu_mish" and args.epochs > 5 else (
            "relu" if name == "hybrid_relu_mish" else name)
        test = net.evaluate(xte, yte, final_act, args.leak)
        results["models"][name] = {"validation_history": history, "test": test}
        print(f"{name:20} TEST accuracy={test['accuracy']:.4%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}. Observed results may differ from paper's reported values.")


if __name__ == "__main__":
    main()
