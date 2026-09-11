"""Uncertainty stress test with Extragradient as the primary robust solver.

The business comparison uses the Extragradient robust plan.

SciPy's epigraph solver is run independently at each uncertainty level only as
an optimization reference, so we can verify that Extragradient is solving the
same robust problem accurately.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Make `python experiments/stress_test.py` work from the repository root.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data import generate_data
from src.evaluation import build_policy_metrics
from src.plots import plot_uncertainty_stress_test
from src.problem import all_scenario_costs, natural_residual
from src.solvers import (
    solve_expected,
    solve_extragradient,
    solve_nominal,
    solve_scipy_robust,
)


def robust_training_objective(x, data):
    """Return max_s C_s(x) over the training scenarios."""
    costs = all_scenario_costs(
        x,
        data["train_demands"],
        data,
    )
    return float(np.max(costs))


def run_stress_test(output_root="."):
    output_root = Path(output_root)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    uncertainty_levels = [0.05, 0.10, 0.20, 0.30, 0.40]

    # ------------------------------------------------------------
    # We create TWO result tables.
    # 1. policy_rows:
    #       business performance of
    #       Nominal / Expected Value / Robust-EG
    # 2. validation_rows:
    #       optimization comparison
    #       Extragradient versus SciPy reference
    # ------------------------------------------------------------
    policy_rows = []
    validation_rows = []
    # ------------------------------------------------------------
    # Numerical settings.
    # The stress test uses fewer scenarios than main.py so that repeating the optimization across several uncertainty levels remains computationally light.
    # Extragradient is the PRIMARY robust solver.
    # SciPy is only an independent reference solver.
    # ------------------------------------------------------------

    config = {
        # Extragradient
        "step_size": 0.010, "max_iterations": 8000, "tolerance": 1e-5, "record_every": 20,
        # General output
        "verbose": False,
        # SciPy nominal / expected
        "scipy_max_iterations": 800,
        # SciPy robust reference
        "scipy_robust_max_iterations": 600, "scipy_robust_ftol": 1e-8,
    }

    # ============================================================
    # LOOP OVER UNCERTAINTY LEVELS
    # ============================================================
    for level in uncertainty_levels:
        # --------------------------------------------------------
        # Generate a new problem instance at this uncertainty level.
        # We use the SAME random seed for every level.
        # Therefore the underlying standardized random shocks are
        # comparable across uncertainty levels; only their magnitude changes.
        # --------------------------------------------------------
        data = generate_data(
            uncertainty_level=level,
            num_train_scenarios=60,
            num_test_scenarios=250,
            seed=2026,
        )
        # ========================================================
        # 1. SOLVE THE THREE PLANNING POLICIES
        # ========================================================
        nominal_result = solve_nominal(data, config)
        expected_result = solve_expected(data, config)

        # --------------------------------------------------------
        # PRIMARY ROBUST SOLUTION
        # This is now the robust solution that will be used in:
        #   - held-out business evaluation
        #   - uncertainty stress-test figure
        #   - uncertainty stress-test CSV
        # --------------------------------------------------------
        eg_result = solve_extragradient(data, config)

        # --------------------------------------------------------
        # REFERENCE ROBUST SOLUTION
        # SciPy solves the epigraph formulation independently so that
        # we can check whether Extragradient reached essentially the
        # same robust optimum.
        # --------------------------------------------------------
        scipy_result = solve_scipy_robust(data, config)
        
        # ========================================================
        # 2. HELD-OUT BUSINESS EVALUATION
        # ========================================================

        policy_solutions = {
            "Nominal": nominal_result["x"], 
            "Expected Value": expected_result["x"], 
            "Robust (Extragradient)": eg_result["x"],
        }

        # IMPORTANT:
        # build_policy_metrics evaluates every capacity plan on
        # data["test_demands"], not training scenarios.
        # Therefore this remains a genuine out-of-sample comparison.

        metrics = build_policy_metrics(policy_solutions, data)

        # Add the uncertainty level so results from different
        # experiments can be concatenated into one table.

        metrics.insert(0, "uncertainty_level", level)
        policy_rows.append(metrics)

        # ========================================================
        # 3. VALIDATE EXTRAGRADIENT AGAINST SCIPY
        # ========================================================

        # Training robust objective of the EG solution
        #     max_s C_s(x_EG)

        eg_objective = robust_training_objective(eg_result["x"], data)

        # Training robust objective of the independent SciPy solution:
        #     max_s C_s(x_SciPy)

        scipy_objective = robust_training_objective(scipy_result["x"], data)

        # Signed relative objective gap:
        #     phi(x_EG) - phi(x_SciPy)
        #     -------------------------
        #            |phi(x_SciPy)|
        # The max(1, ...) avoids division by a tiny number in a different problem scaling.
        
        signed_relative_gap = ((eg_objective - scipy_objective)/max(1.0, abs(scipy_objective)))

        # For solution-quality reporting we mainly care about magnitude rather than sign.

        absolute_relative_gap = abs(signed_relative_gap)

        # Recompute the natural residual at the FINAL EG iterate.
        # This is slightly cleaner than using only the last value
        # stored in residual_history.

        eg_residual = natural_residual(eg_result["x"], eg_result["p"], data)

        validation_rows.append(
            {
                "uncertainty_level": level,
                "eg_train_robust_objective": eg_objective, 
                "scipy_train_robust_objective": scipy_objective, 
                "eg_minus_scipy_relative_gap": signed_relative_gap, 
                "absolute_relative_gap": absolute_relative_gap, 
                "eg_natural_residual": eg_residual, 
                "eg_iterations": eg_result["iterations"], 
                "eg_runtime_seconds": eg_result["runtime"], 
                "eg_status": eg_result["status"], 
                "scipy_iterations": scipy_result["iterations"], 
                "scipy_runtime_seconds": scipy_result["runtime"], 
                "scipy_status": scipy_result["status"],
            }
        )


        # --------------------------------------------------------
        # Progress message.
        # --------------------------------------------------------

        print(
            f"Finished uncertainty {level:.2f} | "
            f"EG obj {eg_objective:.4f} | "
            f"SciPy obj {scipy_objective:.4f} | "
            f"abs rel gap {absolute_relative_gap:.3e} | "
            f"EG residual {eg_residual:.3e}"
        )

    # ============================================================
    # 4. COMBINE RESULT TABLES
    # ============================================================

    stress_df = pd.concat(policy_rows, ignore_index=True)
    validation_df = pd.DataFrame(validation_rows)

    # ============================================================
    # 5. SAVE CSV FILES
    # ============================================================
    # Business / policy stress-test results.

    stress_df.to_csv(output_root / "results" / "uncertainty_stress_test.csv", index=False)

    # Separate algorithm-validation table.

    validation_df.to_csv(output_root / "results" / "stress_test_solver_validation.csv", index=False)

    # ============================================================
    # 6. PLOT
    # ============================================================

    # IMPORTANT:
    # Because stress_df contains
    #     Robust (Extragradient)
    # this figure now plots the EG robust solution.
    # SciPy does not appear as a business policy in this plot.

    plot_uncertainty_stress_test(stress_df, output_root / "figures" / "uncertainty_stress_test.png")

    # Keep the same return type as the old implementation:
    # one DataFrame containing the business stress-test results.
    return stress_df


if __name__ == "__main__":
    stress_df = run_stress_test(".")

    # The validation table was saved by run_stress_test().
    validation_df = pd.read_csv(ROOT / "results" / "stress_test_solver_validation.csv")
    print("\nStress-test business summary:")
    print(
        stress_df[
            [
                "uncertainty_level",
                "policy",
                "worst_case_cost",
                "service_level",
                "total_reserved_capacity",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:,.4f}",
        )
    )

    print("\nExtragradient vs SciPy validation:")
    print(
        validation_df[
            [
                "uncertainty_level",
                "eg_train_robust_objective",
                "scipy_train_robust_objective",
                "absolute_relative_gap",
                "eg_natural_residual",
                "eg_iterations",
                "eg_status",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:,.6g}",
        )
    )