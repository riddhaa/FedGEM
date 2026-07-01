"""Validate adaptive split-refine-merge as a centralized clustering algorithm.

Labels are used only for the train/test split and evaluation; the preprocessing
algorithm receives feature vectors only.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptive_preprocessing import AdaptiveSplitConfig, adaptive_split_client


ROOT = Path(__file__).resolve().parent


def _load_npy(dataset):
    directory = ROOT / dataset
    if dataset == "MNIST":
        x = np.concatenate([
            np.load(directory / "embeddings_vae_train.npy"),
            np.load(directory / "embeddings_vae_test.npy"),
        ])
        y = np.concatenate([
            np.load(directory / "labels_train.npy"),
            np.load(directory / "labels_test.npy"),
        ])
    else:
        x = np.load(directory / "x_data.npy")
        y = np.load(directory / "y_data.npy")
    return np.asarray(x, dtype=float), np.asarray(y).reshape(-1)


def _load_frogs(dataset):
    import pandas as pd

    dataframe = pd.read_csv(ROOT / dataset / "Frogs_MFCCs.csv")
    raw_features = dataframe.iloc[:, :-3].values
    x = raw_features[:, 1:-1].astype(float)
    label_column = -2 if dataset == "FrogA" else -3
    _, y = np.unique(dataframe.iloc[:, label_column].values, return_inverse=True)
    return x, y


def _load_uci(dataset):
    from ucimlrepo import fetch_ucirepo

    if dataset == "Abalone":
        fetched = fetch_ucirepo(id=1)
        x = np.asarray(fetched.data.features)[:, 1:].astype(float)
        raw = np.asarray(fetched.data.targets)[:, 0] - 1
        y = np.zeros(raw.shape, dtype=int)
        y[raw == 6] = 1
        y[raw == 7] = 2
        y[raw == 8] = 3
        y[raw == 9] = 4
        y[raw == 10] = 5
        y[(raw == 11) | (raw == 12)] = 6
        y[raw > 12] = 7
        return x, y

    fetched = fetch_ucirepo(id=107)
    return (
        np.asarray(fetched.data.features, dtype=float),
        np.asarray(fetched.data.targets)[:, 0],
    )


def load_dataset(dataset):
    if dataset == "Synthetic":
        return _load_synthetic()
    if dataset in {"MNIST", "FMNIST", "EMNIST", "CIFAR-10"}:
        return _load_npy(dataset)
    if dataset in {"FrogA", "FrogB"}:
        return _load_frogs(dataset)
    return _load_uci(dataset)


def _load_synthetic():
    """Small smoke-test dataset with three obvious Gaussian clusters."""
    rng = np.random.default_rng(7)
    centers = np.array([[-20.0, 0.0], [0.0, 16.0], [20.0, 0.0]])
    x = np.concatenate([
        rng.normal(loc=center, scale=0.5, size=(120, 2))
        for center in centers
    ])
    y = np.repeat(np.arange(len(centers)), 120)
    return x, y


def predict_components(x, means, weights):
    squared_distance = np.sum((x[:, None, :] - means[None, :, :]) ** 2, axis=2)
    log_joint = -0.5 * squared_distance + np.log(np.maximum(weights, 1e-300))
    return np.argmax(log_joint - logsumexp(log_joint, axis=1)[:, None], axis=1)


def matched_centroid_report(x_train, y_train, means):
    """Match estimated centroids to labeled class means for interpretation only."""
    labels = np.unique(y_train)
    class_means = np.stack([x_train[y_train == label].mean(axis=0) for label in labels])
    distances = np.linalg.norm(means[:, None, :] - class_means[None, :, :], axis=2)
    estimated_indices, class_indices = linear_sum_assignment(distances)
    return [
        {
            "estimated_component": int(estimated),
            "true_label": str(labels[target]),
            "centroid_distance": float(distances[estimated, target]),
        }
        for estimated, target in zip(estimated_indices, class_indices)
    ]


def evaluate(dataset, args):
    x, y = load_dataset(dataset)
    if len(x) != len(y):
        raise ValueError("Feature and label arrays have different lengths")
    if not np.all(np.isfinite(x)):
        raise ValueError("Dataset contains non-finite feature values")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=args.train_fraction,
        random_state=args.seed,
        stratify=y,
    )
    scaler = None
    if args.standardize:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    config = AdaptiveSplitConfig(
        perturbation=args.delta,
        min_child_fraction=args.tau,
        overlap_radius=args.overlap_radius,
        split_em_steps=args.split_em_steps,
        refine_em_steps=args.refine_em_steps,
        max_components=args.max_components,
    )
    started = time.perf_counter()
    means, weights = adaptive_split_client(x_train, config)
    runtime = time.perf_counter() - started
    predicted = predict_components(x_test, means, weights)

    silhouette = None
    if 1 < len(np.unique(predicted)) < len(x_test):
        silhouette = float(silhouette_score(x_test, predicted))
    true_k = int(len(np.unique(y)))
    result = {
        "dataset": dataset,
        "command": " ".join(sys.argv),
        "config": {
            "delta": float(args.delta),
            "tau": float(args.tau),
            "overlap_radius": float(args.overlap_radius),
            "split_em_steps": int(args.split_em_steps),
            "refine_em_steps": int(args.refine_em_steps),
            "max_components": (
                int(args.max_components) if args.max_components is not None else None
            ),
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
        },
        "n_samples": int(len(x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x.shape[1]),
        "true_k": true_k,
        "estimated_k": int(len(means)),
        "cluster_count_error": int(len(means) - true_k),
        "adjusted_rand_index": float(adjusted_rand_score(y_test, predicted)),
        "silhouette_score": silhouette,
        "runtime_seconds": float(runtime),
        "standardized": bool(args.standardize),
        "mixture_weights": weights.tolist(),
        "centroid_matches": matched_centroid_report(x_train, y_train, means),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset.lower().replace("-", "_")
    centroids_original_scale = scaler.inverse_transform(means) if scaler else means
    np.savez(
        output_dir / f"{stem}_standalone_centroids.npz",
        centroids=centroids_original_scale,
        weights=weights,
        estimated_k=len(means),
        true_k=true_k,
    )
    (output_dir / f"{stem}_standalone_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "Synthetic",
            "MNIST",
            "FMNIST",
            "EMNIST",
            "CIFAR-10",
            "Abalone",
            "FrogA",
            "FrogB",
            "Waveform",
            "all",
        ],
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--overlap-radius", type=float, default=1.0)
    parser.add_argument("--split-em-steps", type=int, default=20)
    parser.add_argument("--refine-em-steps", type=int, default=20)
    parser.add_argument("--max-components", type=int)
    parser.add_argument("--output-dir", default=str(ROOT / "standalone_results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must lie in (0, 1)")
    datasets = (
        ["MNIST", "FMNIST", "EMNIST", "CIFAR-10", "Abalone", "FrogA", "FrogB", "Waveform"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset in datasets:
        print("\n===", dataset, "===")
        evaluate(dataset, args)


if __name__ == "__main__":
    main()
