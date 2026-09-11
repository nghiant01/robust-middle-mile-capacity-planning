"""Plotting functions only."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_policy_cost_comparison(policy_metrics, path):
    metrics = ["mean_total_cost", "p95_total_cost", "worst_case_cost"]
    labels = ["Mean", "95th percentile", "Worst case"]
    x = np.arange(len(policy_metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1) * width, policy_metrics[metric], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(policy_metrics["policy"])
    ax.set_ylabel("Held-out total cost")
    ax.set_title("Out-of-sample policy cost comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_service_level_comparison(policy_metrics, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = 100.0 * policy_metrics["service_level"].to_numpy()
    ax.bar(policy_metrics["policy"], values)
    ax.set_ylabel("Demand-weighted service level (%)")
    ax.set_title("Held-out service level")
    ax.set_ylim(max(0.0, values.min() - 3.0), min(100.0, values.max() + 1.0))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_algorithm_convergence(algorithm_results, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in algorithm_results.items():
        if result["residual_history"]:
            ax.plot(
                result["iteration_history"],
                result["residual_history"],
                label=label,
                linewidth=1.7,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Natural residual $\|z-P(z-F(z))\|$")
    ax.set_title("Robust saddle-point algorithm convergence")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_capacity_comparison(policy_solutions, data, path):
    """Compare total weekly reserved capacity by lane."""
    num_days = data["num_days"]
    num_lanes = data["num_lanes"]
    lane_labels = data["lanes"]["lane_id"].to_numpy()
    x_locations = np.arange(num_lanes)
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (policy, x) in enumerate(policy_solutions.items()):
        lane_totals = np.asarray(x).reshape(num_days, num_lanes).sum(axis=0)
        ax.bar(x_locations + (i - 1) * width, lane_totals, width, label=policy)

    ax.set_xticks(x_locations)
    ax.set_xticklabels(lane_labels, rotation=45)
    ax.set_ylabel("Weekly reserved capacity")
    ax.set_title("Capacity allocation by transportation lane")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_uncertainty_stress_test(stress_df, path, metric="worst_case_cost"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in stress_df["policy"].unique():
        subset = stress_df[stress_df["policy"] == policy]
        ax.plot(
            subset["uncertainty_level"],
            subset[metric],
            marker="o",
            label=policy,
        )

    ylabel = "Held-out worst-case total cost" if metric == "worst_case_cost" else metric
    ax.set_xlabel("Demand uncertainty level")
    ax.set_ylabel(ylabel)
    ax.set_title("Uncertainty stress test")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)
