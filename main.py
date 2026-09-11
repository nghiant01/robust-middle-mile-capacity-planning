"""Main end-to-end experiment for the GitHub portfolio project."""

from pathlib import Path
import numpy as np

from experiments.stress_test import run_stress_test
from src.data import generate_data
from src.evaluation import build_algorithm_metrics, build_policy_metrics
from src.plots import (
    plot_algorithm_convergence,
    plot_capacity_comparison,
    plot_policy_cost_comparison,
    plot_service_level_comparison,
)
from src.problem import finite_difference_gradient_check, project_capacity, project_simplex
from src.solvers import run_solver, solve_expected, solve_gda, solve_nominal, solve_scipy_robust


# ============================================================
# Key numerical choices are intentionally visible in one place.
# ============================================================
SEED = 42
UNCERTAINTY_LEVEL = 0.20

CONFIG = {
    "step_size": 0.010,          # Extragradient step size.
    "gda_step_size": 0.004,      # Smaller baseline step to avoid obvious divergence.
    "max_iterations": 8000,
    "tolerance": 1e-5,
    "record_every": 20,
    "verbose": False,
    "scipy_max_iterations": 1000,
    "scipy_robust_max_iterations": 1000,
    "scipy_robust_ftol": 1e-9,
}

# Change only this string to use another robust solver through the shared API.
ALGORITHM = "extragradient"
RUN_STRESS_TEST = True


def run_basic_checks(data):
    gradient_check = finite_difference_gradient_check(data)
    print("Gradient check:", gradient_check)
    if not gradient_check["passed"]:
        raise RuntimeError("Gradient check failed.")

    # Simplex projection check.
    p_test = project_simplex(np.array([0.8, -0.3, 1.7, 0.1]))
    assert np.all(p_test >= -1e-12)
    assert abs(p_test.sum() - 1.0) < 1e-12

    # Capacity projection check.
    x_test = project_capacity(2.0 * data["max_capacity"], data)
    assert np.all(x_test >= -1e-12)
    assert np.all(x_test <= data["max_capacity"] + 1e-12)
    print("Projection checks: passed")


def main():
    root = Path(__file__).resolve().parent
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    data = generate_data(
        num_facilities=8,
        num_lanes=18,
        num_days=7,
        num_train_scenarios=100,
        num_test_scenarios=500,
        uncertainty_level=UNCERTAINTY_LEVEL,
        seed=SEED,
    )

    run_basic_checks(data)

    # Planning policies.
    nominal_result = solve_nominal(data, CONFIG)
    expected_result = solve_expected(data, CONFIG)

    # Robust optimization: one string chooses the solver used in this call.
    selected_robust_result = run_solver(ALGORITHM, data, CONFIG)
    print(f"Selected robust solver: {ALGORITHM}")

    # Run/reuse the standard methods for the algorithm-comparison figure/table.
    extragradient_result = (
        selected_robust_result
        if ALGORITHM == "extragradient"
        else run_solver("extragradient", data, CONFIG)
    )
    gda_result = (
        selected_robust_result
        if ALGORITHM == "gda"
        else run_solver("gda", data, CONFIG)
    )
    scipy_result = (
        selected_robust_result
        if ALGORITHM == "scipy"
        else run_solver("scipy", data, CONFIG)
    )

    # Feasibility checks for the first-order saddle algorithms.
    for name, result in {"Extragradient": extragradient_result, "GDA": gda_result}.items():
        assert np.all(result["x"] >= -1e-10)
        assert np.all(result["x"] <= data["max_capacity"] + 1e-10)
        assert np.all(result["p"] >= -1e-10)
        assert abs(result["p"].sum() - 1.0) < 1e-10
        print(f"{name} feasibility checks: passed")

    # Use the reliable SciPy robust solution for the business-policy table.
    policy_solutions = {
        "Nominal": nominal_result["x"],
        "Expected Value": expected_result["x"],
        "Robust": extragradient_result["x"],
    }

    policy_metrics = build_policy_metrics(policy_solutions, data)
    policy_metrics.to_csv(root / "results" / "policy_metrics.csv", index=False)

    algorithm_results = {
        "Extragradient": extragradient_result,
        "GDA": gda_result,
        "SciPy reference": scipy_result,
    }
    algorithm_metrics = build_algorithm_metrics(algorithm_results, data)
    algorithm_metrics.to_csv(root / "results" / "algorithm_metrics.csv", index=False)

    plot_policy_cost_comparison(
        policy_metrics, root / "figures" / "policy_cost_comparison.png"
    )
    plot_service_level_comparison(
        policy_metrics, root / "figures" / "service_level_comparison.png"
    )
    plot_algorithm_convergence(
        {"Extragradient": extragradient_result, "GDA": gda_result},
        root / "figures" / "robust_algorithm_convergence.png",
    )
    plot_capacity_comparison(
        policy_solutions, data, root / "figures" / "capacity_comparison.png"
    )

    print("\nHeld-out policy metrics:")
    columns = [
        "policy",
        "reservation_cost",
        "mean_total_cost",
        "p95_total_cost",
        "worst_case_cost",
        "mean_shortage",
        "service_level",
        "total_reserved_capacity",
    ]
    print(policy_metrics[columns].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\nRobust algorithm comparison:")
    print(algorithm_metrics.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

    scipy_obj = float(
        algorithm_metrics.loc[
            algorithm_metrics["algorithm"] == "SciPy reference", "train_robust_objective"
        ].iloc[0]
    )
    eg_obj = float(
        algorithm_metrics.loc[
            algorithm_metrics["algorithm"] == "Extragradient", "train_robust_objective"
        ].iloc[0]
    )
    relative_gap = (eg_obj - scipy_obj) / max(1.0, abs(scipy_obj))
    print(f"\nExtragradient vs SciPy robust-objective relative gap: {relative_gap:.3%}")

    if RUN_STRESS_TEST:
        print("\nRunning lightweight uncertainty stress test...")
        run_stress_test(root)

    print("\nDone. Results are in results/ and figures/.")


if __name__ == "__main__":
    main()
