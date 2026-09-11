"""Optimization algorithms.

A solver is deliberately just a function with the API
    result = solver(data, config)
so research algorithms can be added without classes or infrastructure.
"""

import time
import numpy as np
from scipy.optimize import minimize

from src.problem import (
    all_scenario_costs,
    all_scenario_gradients,
    natural_residual,
    project_capacity,
    project_simplex,
    saddle_operator,
    scenario_cost,
    scenario_gradient,
)


def _base_result(x, p, iterations, runtime, status, objective_history, residual_history, iteration_history):
    return {
        "x": np.asarray(x),
        "p": np.asarray(p),
        "iterations": int(iterations),
        "runtime": float(runtime),
        "status": status,
        "objective_history": list(objective_history),
        "residual_history": list(residual_history),
        "iteration_history": list(iteration_history),
    }


def solve_nominal(data, config):
    """Minimize cost at the base/mean forecast demand."""
    start = time.perf_counter()
    x0 = project_capacity(data["nominal_demand"], data)
    bounds = list(zip(np.zeros_like(x0), data["max_capacity"]))
    history = []

    objective = lambda x: scenario_cost(x, data["nominal_demand"], data)
    gradient = lambda x: scenario_gradient(x, data["nominal_demand"], data)

    def callback(xk):
        history.append(objective(xk))

    opt = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": config.get("scipy_max_iterations", 1000), "ftol": 1e-12},
    )

    runtime = time.perf_counter() - start
    p = np.full(data["num_train_scenarios"], 1.0 / data["num_train_scenarios"])
    if not history:
        history = [objective(opt.x)]

    return _base_result(
        opt.x, p, opt.nit, runtime, "converged" if opt.success else opt.message,
        history, [], list(range(1, len(history) + 1))
    )


def solve_expected(data, config):
    """Minimize average cost over training scenarios."""
    start = time.perf_counter()
    demands = data["train_demands"]
    x0 = project_capacity(np.mean(demands, axis=0), data)
    bounds = list(zip(np.zeros_like(x0), data["max_capacity"]))
    history = []

    def objective(x):
        return float(np.mean(all_scenario_costs(x, demands, data)))

    def gradient(x):
        return np.mean(all_scenario_gradients(x, demands, data), axis=0)

    def callback(xk):
        history.append(objective(xk))

    opt = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": config.get("scipy_max_iterations", 1000), "ftol": 1e-12},
    )

    runtime = time.perf_counter() - start
    p = np.full(data["num_train_scenarios"], 1.0 / data["num_train_scenarios"])
    if not history:
        history = [objective(opt.x)]

    return _base_result(
        opt.x, p, opt.nit, runtime, "converged" if opt.success else opt.message,
        history, [], list(range(1, len(history) + 1))
    )


def solve_extragradient(data, config):
    """Korpelevich projected extragradient for the robust saddle problem."""
    start = time.perf_counter()

    step_size = config.get("step_size", 0.01)
    max_iterations = config.get("max_iterations", 8000)
    tolerance = config.get("tolerance", 1e-5)
    record_every = config.get("record_every", 20)
    verbose = config.get("verbose", False)

    num_scenarios = data["num_train_scenarios"]
    x = project_capacity(np.mean(data["train_demands"], axis=0), data)
    p = np.full(num_scenarios, 1.0 / num_scenarios)

    objective_history = []
    residual_history = []
    iteration_history = []
    status = "max_iterations"

    for k in range(max_iterations):
        # Predictor: z_half = P(z_k - eta F(z_k)).
        grad_x, operator_p = saddle_operator(x, p, data)
        x_half = project_capacity(x - step_size * grad_x, data)
        p_half = project_simplex(p - step_size * operator_p)

        # Corrector: z_{k+1} = P(z_k - eta F(z_half)).
        grad_x_half, operator_p_half = saddle_operator(x_half, p_half, data)
        x_new = project_capacity(x - step_size * grad_x_half, data)
        p_new = project_simplex(p - step_size * operator_p_half)

        x, p = x_new, p_new

        should_record = (k == 0) or ((k + 1) % record_every == 0) or (k + 1 == max_iterations)
        if should_record:
            robust_objective = float(np.max(all_scenario_costs(x, data["train_demands"], data)))
            residual = natural_residual(x, p, data)

            objective_history.append(robust_objective)
            residual_history.append(residual)
            iteration_history.append(k + 1)

            if verbose:
                print(
                    f"EG iter {k+1:5d} | robust obj {robust_objective:10.4f} "
                    f"| residual {residual:9.3e}"
                )

            if residual < tolerance:
                status = "converged"
                break

    iterations = k + 1
    runtime = time.perf_counter() - start
    return _base_result(
        x, p, iterations, runtime, status,
        objective_history, residual_history, iteration_history
    )


def solve_gda(data, config):
    """Simple simultaneous projected gradient descent-ascent baseline."""
    start = time.perf_counter()

    step_size = config.get("gda_step_size", config.get("step_size", 0.01))
    max_iterations = config.get("max_iterations", 8000)
    tolerance = config.get("tolerance", 1e-5)
    record_every = config.get("record_every", 20)
    verbose = config.get("verbose", False)

    num_scenarios = data["num_train_scenarios"]
    x = project_capacity(np.mean(data["train_demands"], axis=0), data)
    p = np.full(num_scenarios, 1.0 / num_scenarios)

    objective_history = []
    residual_history = []
    iteration_history = []
    status = "max_iterations"

    for k in range(max_iterations):
        grad_x, operator_p = saddle_operator(x, p, data)

        # Since operator_p = -C(x), this is p + eta * C(x): gradient ascent in p.
        x = project_capacity(x - step_size * grad_x, data)
        p = project_simplex(p - step_size * operator_p)

        should_record = (k == 0) or ((k + 1) % record_every == 0) or (k + 1 == max_iterations)
        if should_record:
            robust_objective = float(np.max(all_scenario_costs(x, data["train_demands"], data)))
            residual = natural_residual(x, p, data)

            objective_history.append(robust_objective)
            residual_history.append(residual)
            iteration_history.append(k + 1)

            if verbose:
                print(
                    f"GDA iter {k+1:5d} | robust obj {robust_objective:10.4f} "
                    f"| residual {residual:9.3e}"
                )

            if residual < tolerance:
                status = "converged"
                break

    iterations = k + 1
    runtime = time.perf_counter() - start
    return _base_result(
        x, p, iterations, runtime, status,
        objective_history, residual_history, iteration_history
    )


def solve_scipy_robust(data, config):
    """Reference robust solver via the epigraph formulation.

    minimize t
    subject to C_s(x) <= t for all training scenarios s,
               0 <= x <= max_capacity.
    """
    start = time.perf_counter()
    demands = data["train_demands"]
    num_x = data["num_variables"]

    x0 = project_capacity(np.mean(demands, axis=0), data)
    t0 = float(np.max(all_scenario_costs(x0, demands, data))) + 1.0
    y0 = np.concatenate([x0, [t0]])

    objective_history = []

    def objective(y):
        return float(y[-1])

    def objective_jac(y):
        grad = np.zeros_like(y)
        grad[-1] = 1.0
        return grad

    def constraints(y):
        x = y[:num_x]
        t = y[-1]
        return t - all_scenario_costs(x, demands, data)

    def constraints_jac(y):
        x = y[:num_x]
        grads = all_scenario_gradients(x, demands, data)
        # Each row is derivative of t - C_s(x).
        return np.column_stack([-grads, np.ones(demands.shape[0])])

    def callback(yk):
        objective_history.append(float(np.max(all_scenario_costs(yk[:num_x], demands, data))))

    bounds = [(0.0, float(ub)) for ub in data["max_capacity"]] + [(0.0, None)]

    opt = minimize(
        objective,
        y0,
        jac=objective_jac,
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints, "jac": constraints_jac},
        method="SLSQP",
        callback=callback,
        options={
            "maxiter": config.get("scipy_robust_max_iterations", 1000),
            "ftol": config.get("scipy_robust_ftol", 1e-9),
            "disp": bool(config.get("verbose", False)),
        },
    )

    x = project_capacity(opt.x[:num_x], data)
    costs = all_scenario_costs(x, demands, data)
    worst_index = int(np.argmax(costs))
    p = np.zeros(data["num_train_scenarios"])
    p[worst_index] = 1.0  # Compatibility only; SLSQP does not return saddle weights here.

    runtime = time.perf_counter() - start
    if not objective_history:
        objective_history = [float(np.max(costs))]

    return _base_result(
        x,
        p,
        opt.nit,
        runtime,
        "converged" if opt.success else str(opt.message),
        objective_history,
        [],
        list(range(1, len(objective_history) + 1)),
    )


SOLVERS = {
    "extragradient": solve_extragradient,
    "gda": solve_gda,
    "scipy": solve_scipy_robust,
}


def run_solver(name, data, config):
    """Run a robust solver by changing only one string."""
    if name not in SOLVERS:
        available = ", ".join(sorted(SOLVERS))
        raise ValueError(f"Unknown solver '{name}'. Available: {available}")
    return SOLVERS[name](data, config)
