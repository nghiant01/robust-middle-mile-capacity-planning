"""Synthetic data generation for robust middle-mile capacity planning."""

import numpy as np
import pandas as pd


def _make_lane_table(num_facilities, num_lanes, rng):
    """Create a small directed transportation network with unique lanes."""
    facilities = [f"F{i+1}" for i in range(num_facilities)]
    possible_lanes = [(o, d) for o in facilities for d in facilities if o != d]

    if num_lanes > len(possible_lanes):
        raise ValueError("num_lanes is too large for the number of facilities.")

    selected = rng.choice(len(possible_lanes), size=num_lanes, replace=False)

    # Capacity units are intentionally abstract: think reserved driver/load capacity.
    base_demand = rng.uniform(4.0, 10.0, size=num_lanes)
    reservation_cost = rng.uniform(0.08, 0.18, size=num_lanes)
    maximum_capacity = 1.9 * base_demand

    rows = []
    for lane_id, idx in enumerate(selected):
        origin, destination = possible_lanes[idx]
        rows.append(
            {
                "lane_id": f"L{lane_id+1:02d}",
                "origin": origin,
                "destination": destination,
                "base_demand": base_demand[lane_id],
                "reservation_cost": reservation_cost[lane_id],
                "maximum_capacity": maximum_capacity[lane_id],
            }
        )

    return pd.DataFrame(rows)


def _generate_demands(base_demand_matrix, num_scenarios, uncertainty_level, rng):
    """Generate correlated demand scenarios.

    A day-level shock is shared by all lanes on the same day, while an
    independent lane shock adds local variation. Their standard deviations are
    calibrated so the combined uncertainty is approximately uncertainty_level.
    """
    num_days, num_lanes = base_demand_matrix.shape

    common_shock = rng.normal(
        loc=0.0,
        scale=0.60 * uncertainty_level,
        size=(num_scenarios, num_days, 1),
    )
    lane_noise = rng.normal(
        loc=0.0,
        scale=0.80 * uncertainty_level,
        size=(num_scenarios, num_days, num_lanes),
    )

    demand_multiplier = 1.0 + common_shock + lane_noise
    demand_multiplier = np.clip(demand_multiplier, 0.20, None)

    return base_demand_matrix[None, :, :] * demand_multiplier


def flatten_day_lane(array):
    """Flatten [..., day, lane] arrays to [..., day*lane]."""
    return np.asarray(array).reshape(*np.asarray(array).shape[:-2], -1)


def unflatten_day_lane(vector, num_days, num_lanes):
    """Reshape a flat lane-day vector back to [day, lane]."""
    return np.asarray(vector).reshape(num_days, num_lanes)


def generate_data(
    num_facilities=8,
    num_lanes=18,
    num_days=7,
    num_train_scenarios=100,
    num_test_scenarios=500,
    uncertainty_level=0.20,
    shortage_penalty=1.0,
    idle_penalty=0.05,
    seed=42,
):
    """Generate a complete synthetic problem and return one simple dictionary."""
    rng = np.random.default_rng(seed)
    lanes = _make_lane_table(num_facilities, num_lanes, rng)

    # A visible weekly pattern introduces modest day-to-day heterogeneity.
    day_factors = np.array([0.92, 1.00, 1.05, 1.10, 1.15, 0.88, 0.82])
    if num_days != 7:
        # For non-default sizes, use a smooth deterministic pattern.
        grid = np.linspace(0.0, 2.0 * np.pi, num_days, endpoint=False)
        day_factors = 1.0 + 0.12 * np.sin(grid - 0.5)

    lane_base = lanes["base_demand"].to_numpy()
    base_demand_matrix = day_factors[:, None] * lane_base[None, :]

    train_3d = _generate_demands(
        base_demand_matrix, num_train_scenarios, uncertainty_level, rng
    )
    test_3d = _generate_demands(
        base_demand_matrix, num_test_scenarios, uncertainty_level, rng
    )

    reservation_cost = np.tile(lanes["reservation_cost"].to_numpy(), num_days)
    max_capacity = np.tile(lanes["maximum_capacity"].to_numpy(), num_days)
    nominal_demand = base_demand_matrix.reshape(-1)

    data = {
        "lanes": lanes,
        "num_facilities": num_facilities,
        "num_lanes": num_lanes,
        "num_days": num_days,
        "num_variables": num_days * num_lanes,
        "num_train_scenarios": num_train_scenarios,
        "num_test_scenarios": num_test_scenarios,
        "uncertainty_level": uncertainty_level,
        "shortage_penalty": shortage_penalty,
        "idle_penalty": idle_penalty,
        "reservation_cost": reservation_cost,
        "max_capacity": max_capacity,
        "nominal_demand": nominal_demand,
        "train_demands": flatten_day_lane(train_3d),
        "test_demands": flatten_day_lane(test_3d),
        "seed": seed,
    }
    return data
