"""Held-out evaluation and result-table helpers."""

import numpy as np
import pandas as pd

from src.problem import all_scenario_costs


def evaluate_solution(x, data):
    """Evaluate one fixed capacity plan on held-out test scenarios only."""
    demands = data["test_demands"]
    x = np.asarray(x)

    costs = all_scenario_costs(x, demands, data)
    shortage = np.maximum(demands - x[None, :], 0.0)
    unused = np.maximum(x[None, :] - demands, 0.0)

    scenario_shortage = np.sum(shortage, axis=1)
    scenario_unused = np.sum(unused, axis=1)

    # Demand-weighted service level across all lane-day observations.
    service_level = 1.0 - np.sum(shortage) / np.sum(demands)

    return {
        "reservation_cost": float(np.dot(data["reservation_cost"], x)),
        "mean_total_cost": float(np.mean(costs)),
        "median_total_cost": float(np.median(costs)),
        "p95_total_cost": float(np.percentile(costs, 95)),
        "worst_case_cost": float(np.max(costs)),
        "mean_shortage": float(np.mean(scenario_shortage)),
        "p95_shortage": float(np.percentile(scenario_shortage, 95)),
        "worst_shortage": float(np.max(scenario_shortage)),
        "mean_unused_capacity": float(np.mean(scenario_unused)),
        "service_level": float(service_level),
    }


def build_policy_metrics(policy_solutions, data):
    rows = []
    for policy_name, x in policy_solutions.items():
        row = {"policy": policy_name}
        row.update(evaluate_solution(x, data))
        row["total_reserved_capacity"] = float(np.sum(x))
        rows.append(row)
    return pd.DataFrame(rows)


def build_algorithm_metrics(algorithm_results, data):
    rows = []
    for name, result in algorithm_results.items():
        train_robust_objective = float(
            np.max(all_scenario_costs(result["x"], data["train_demands"], data))
        )
        final_residual = (
            float(result["residual_history"][-1])
            if result["residual_history"]
            else np.nan
        )
        rows.append(
            {
                "algorithm": name,
                "train_robust_objective": train_robust_objective,
                "final_natural_residual": final_residual,
                "iterations": result["iterations"],
                "runtime_seconds": result["runtime"],
                "status": result["status"],
            }
        )
    return pd.DataFrame(rows)
