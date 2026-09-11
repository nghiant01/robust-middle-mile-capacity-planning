"""Mathematical model and saddle-point primitives."""

import numpy as np


def scenario_cost(x, demand, data):
    """Cost for one demand scenario."""
    x = np.asarray(x)
    demand = np.asarray(demand)

    shortage = np.maximum(demand - x, 0.0)
    idle = np.maximum(x - demand, 0.0)

    reservation = np.dot(data["reservation_cost"], x)
    shortage_cost = data["shortage_penalty"] * np.dot(shortage, shortage)
    idle_cost = data["idle_penalty"] * np.dot(idle, idle)
    return float(reservation + shortage_cost + idle_cost)


def scenario_gradient(x, demand, data):
    """Analytic gradient of scenario_cost with respect to x."""
    x = np.asarray(x)
    demand = np.asarray(demand)

    shortage = np.maximum(demand - x, 0.0)
    idle = np.maximum(x - demand, 0.0)

    gradient = data["reservation_cost"].copy()
    gradient -= 2.0 * data["shortage_penalty"] * shortage
    gradient += 2.0 * data["idle_penalty"] * idle
    return gradient


def all_scenario_costs(x, demands, data):
    """Vector C(x) = [C_1(x), ..., C_S(x)]."""
    demands = np.asarray(demands)
    x = np.asarray(x)

    shortage = np.maximum(demands - x[None, :], 0.0)
    idle = np.maximum(x[None, :] - demands, 0.0)

    reservation = np.dot(data["reservation_cost"], x)
    return (
        reservation
        + data["shortage_penalty"] * np.sum(shortage**2, axis=1)
        + data["idle_penalty"] * np.sum(idle**2, axis=1)
    )


def all_scenario_gradients(x, demands, data):
    """Matrix whose s-th row is grad C_s(x)."""
    demands = np.asarray(demands)
    x = np.asarray(x)

    shortage = np.maximum(demands - x[None, :], 0.0)
    idle = np.maximum(x[None, :] - demands, 0.0)

    return (
        data["reservation_cost"][None, :]
        - 2.0 * data["shortage_penalty"] * shortage
        + 2.0 * data["idle_penalty"] * idle
    )


def project_capacity(x, data):
    """Euclidean projection onto X = {0 <= x <= max_capacity}."""
    return np.clip(np.asarray(x), 0.0, data["max_capacity"])


def project_simplex(p):
    """Euclidean projection onto the probability simplex.

    Implements the standard sorting/thresholding formula for
        Delta = {p >= 0, sum(p) = 1}.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be a one-dimensional vector.")

    sorted_p = np.sort(p)[::-1]
    cumulative = np.cumsum(sorted_p)
    indices = np.arange(1, len(p) + 1)
    active = sorted_p - (cumulative - 1.0) / indices > 0.0

    rho = np.nonzero(active)[0][-1]
    threshold = (cumulative[rho] - 1.0) / (rho + 1.0)
    projected = np.maximum(p - threshold, 0.0)

    # Remove tiny floating-point drift in the sum.
    projected /= projected.sum()
    return projected


def saddle_operator(x, p, data):
    """Return F(x,p) for the robust saddle problem.

    Robust optimization can be written as
        min_x max_{p in Delta} sum_s p_s C_s(x).

    The smooth saddle operator is
        F(x,p) = [sum_s p_s grad C_s(x); -C(x)].

    Together with normal cones N_X(x) and N_Delta(p), the optimality system is
        0 in F(x,p) + N_X(x) x N_Delta(p),
    which is a constrained monotone-inclusion / saddle-point problem.
    """
    demands = data["train_demands"]
    costs = all_scenario_costs(x, demands, data)
    gradients = all_scenario_gradients(x, demands, data)

    grad_x = p @ gradients
    grad_p_for_operator = -costs
    return grad_x, grad_p_for_operator


def natural_residual(x, p, data):
    """Natural residual ||z - P(z - F(z))||.

    The projection P acts separately on the capacity box X and simplex Delta.
    The residual is zero exactly at a fixed point of the projected operator,
    i.e. at a solution of the constrained saddle-point optimality system.
    """
    grad_x, operator_p = saddle_operator(x, p, data)
    projected_x = project_capacity(x - grad_x, data)
    projected_p = project_simplex(p - operator_p)

    dx = x - projected_x
    dp = p - projected_p
    return float(np.sqrt(np.dot(dx, dx) + np.dot(dp, dp)))


def finite_difference_gradient_check(data, epsilon=1e-6, seed=123):
    """Compare the analytic scenario gradient with central finite differences."""
    rng = np.random.default_rng(seed)
    demand = data["train_demands"][0]

    # Stay away from box boundaries and most demand kink locations.
    x = 0.55 * demand + 0.15 * data["max_capacity"]
    x += rng.normal(0.0, 0.01, size=x.size)
    x = project_capacity(x, data)

    analytic = scenario_gradient(x, demand, data)
    finite_diff = np.zeros_like(x)

    for j in range(x.size):
        direction = np.zeros_like(x)
        direction[j] = epsilon
        finite_diff[j] = (
            scenario_cost(x + direction, demand, data)
            - scenario_cost(x - direction, demand, data)
        ) / (2.0 * epsilon)

    max_abs_error = float(np.max(np.abs(analytic - finite_diff)))
    relative_error = float(
        np.linalg.norm(analytic - finite_diff) / max(1.0, np.linalg.norm(analytic))
    )

    return {
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
        "passed": bool(max_abs_error < 1e-5 and relative_error < 1e-6),
    }
