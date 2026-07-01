"""Client-local adaptive split-refine-merge preprocessing for FedGEM.

This implements Algorithms 1 and 2 from the accompanying thesis.  It deliberately
uses only a client's local observations and returns dictionaries in the format
expected by the original FedClem/FedGEM implementation.
"""

from dataclasses import dataclass
import os

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class AdaptiveSplitConfig:
    perturbation: float = 1.0
    min_child_fraction: float = 0.05
    overlap_radius: float = 1.0
    split_em_steps: int = 20
    refine_em_steps: int = 20
    max_components: int | None = None
    tolerance: float = 1e-6

    @classmethod
    def from_environment(cls):
        """Allow long experiment batches to tune settings without editing code."""
        max_components = os.getenv("FEDGEM_SPLIT_MAX_COMPONENTS")
        return cls(
            perturbation=float(os.getenv("FEDGEM_SPLIT_DELTA", "1.0")),
            min_child_fraction=float(os.getenv("FEDGEM_SPLIT_TAU", "0.05")),
            overlap_radius=float(os.getenv("FEDGEM_OVERLAP_RADIUS", "1.0")),
            split_em_steps=int(os.getenv("FEDGEM_SPLIT_EM_STEPS", "20")),
            refine_em_steps=int(os.getenv("FEDGEM_REFINE_EM_STEPS", "20")),
            max_components=int(max_components) if max_components else None,
        )


def _log_normal_identity(x, means):
    d = x.shape[1]
    squared_distance = np.sum((x[:, None, :] - means[None, :, :]) ** 2, axis=2)
    return -0.5 * (d * np.log(2.0 * np.pi) + squared_distance)


def _responsibilities(x, means, weights):
    log_joint = _log_normal_identity(x, means) + np.log(np.maximum(weights, 1e-300))
    log_norm = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - log_norm[:, None]), float(np.sum(log_norm))


def _fixed_covariance_em(x, means, weights, steps, tolerance):
    means = np.asarray(means, dtype=float).copy()
    weights = np.asarray(weights, dtype=float).copy()
    previous = -np.inf
    responsibilities = None
    for _ in range(steps):
        responsibilities, likelihood = _responsibilities(x, means, weights)
        mass = responsibilities.sum(axis=0)
        valid = mass > 1e-12
        if not np.all(valid):
            means = means[valid]
            responsibilities = responsibilities[:, valid]
            mass = mass[valid]
        means = (responsibilities.T @ x) / mass[:, None]
        weights = mass / mass.sum()
        if likelihood - previous <= tolerance:
            break
        previous = likelihood
    responsibilities, likelihood = _responsibilities(x, means, weights)
    return means, weights, responsibilities, likelihood


def _weighted_two_component_em(x, sample_weights, initial_means, config):
    means = np.asarray(initial_means, dtype=float).copy()
    mixing = np.array([0.5, 0.5], dtype=float)
    previous = -np.inf
    for _ in range(config.split_em_steps):
        log_joint = _log_normal_identity(x, means) + np.log(mixing)[None, :]
        log_norm = logsumexp(log_joint, axis=1)
        conditional = np.exp(log_joint - log_norm[:, None])
        weighted_resp = sample_weights[:, None] * conditional
        mass = weighted_resp.sum(axis=0)
        if np.any(mass <= 1e-12):
            return None
        means = (weighted_resp.T @ x) / mass[:, None]
        mixing = mass / mass.sum()
        likelihood = float(sample_weights @ log_norm)
        if likelihood - previous <= config.tolerance:
            break
        previous = likelihood
    return means, mixing, mass, likelihood


def _try_split(x, means, weights, responsibilities, component, config):
    parent_resp = responsibilities[:, component]
    effective_n = float(parent_resp.sum())
    if effective_n < 2.0 * config.min_child_fraction * len(x):
        return None

    parent_mean = (parent_resp @ x) / effective_n
    centered = x - parent_mean
    covariance = (centered * parent_resp[:, None]).T @ centered / effective_n
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    initial = np.stack(
        [parent_mean + config.perturbation * direction,
         parent_mean - config.perturbation * direction]
    )

    candidate = _weighted_two_component_em(x, parent_resp, initial, config)
    if candidate is None:
        return None
    child_means, child_conditional_weights, child_mass, child_likelihood = candidate

    parent_likelihood = float(parent_resp @ _log_normal_identity(x, parent_mean[None, :])[:, 0])
    bic_gain = child_likelihood - parent_likelihood
    bic_gain -= 0.5 * (x.shape[1] + 1) * np.log(max(effective_n, 1.0))
    if bic_gain <= 0.0 or np.min(child_mass) < config.min_child_fraction * len(x):
        return None

    new_means = np.delete(means, component, axis=0)
    new_weights = np.delete(weights, component)
    parent_weight = weights[component]
    new_means = np.concatenate([new_means, child_means], axis=0)
    new_weights = np.concatenate(
        [new_weights, parent_weight * child_conditional_weights], axis=0
    )
    return _fixed_covariance_em(
        x, new_means, new_weights, config.refine_em_steps, config.tolerance
    )[:2]


def _merge_overlaps(x, means, weights, config):
    means, weights, _, _ = _fixed_covariance_em(
        x, means, weights, config.refine_em_steps, config.tolerance
    )
    while len(means) > 1:
        pair = None
        for i in range(len(means) - 1):
            for j in range(i + 1, len(means)):
                if np.linalg.norm(means[i] - means[j]) <= 2.0 * config.overlap_radius:
                    pair = (i, j)
                    break
            if pair is not None:
                break
        if pair is None:
            break
        i, j = pair
        combined_weight = weights[i] + weights[j]
        combined_mean = (weights[i] * means[i] + weights[j] * means[j]) / combined_weight
        keep = np.ones(len(means), dtype=bool)
        keep[[i, j]] = False
        means = np.concatenate([means[keep], combined_mean[None, :]], axis=0)
        weights = np.concatenate([weights[keep], [combined_weight]])
        means, weights, _, _ = _fixed_covariance_em(
            x, means, weights, config.refine_em_steps, config.tolerance
        )
    return means, weights


def adaptive_split_client(x, config=None):
    """Recover one client's local component set, starting from K_g = 1."""
    config = config or AdaptiveSplitConfig.from_environment()
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Client data must be a non-empty, finite 2-D array")
    if not 0.0 < config.min_child_fraction <= 0.5:
        raise ValueError("min_child_fraction must lie in (0, 0.5]")

    means = np.mean(x, axis=0, keepdims=True)
    weights = np.ones(1)
    natural_limit = max(1, int(np.floor(1.0 / config.min_child_fraction)))
    component_limit = config.max_components or natural_limit

    split_accepted = True
    while split_accepted and len(means) < component_limit:
        split_accepted = False
        means, weights, responsibilities, _ = _fixed_covariance_em(
            x, means, weights, config.refine_em_steps, config.tolerance
        )
        for component in range(len(means)):
            split = _try_split(x, means, weights, responsibilities, component, config)
            if split is not None:
                means, weights = split
                split_accepted = True
                break  # Recompute all responsibilities after every accepted split.

    return _merge_overlaps(x, means, weights, config)


def preprocess_clients(train_clients, config=None):
    """Return FedGEM initial means, weights, and recovered local cardinalities."""
    config = config or AdaptiveSplitConfig.from_environment()
    theta_dict, pi_dict = {}, {}
    recovered = np.zeros(len(train_clients), dtype=int)
    for g in range(len(train_clients)):
        key = "client_" + str(g)
        means, weights = adaptive_split_client(train_clients[key], config)
        theta_dict["Theta_" + str(g)] = means
        pi_dict["Pi_" + str(g)] = weights
        recovered[g] = len(means)
        print("Adaptive preprocessing client", g, "recovered K_g =", recovered[g])
    return theta_dict, pi_dict, recovered
