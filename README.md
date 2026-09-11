# Robust Middle-Mile Capacity Planning under Demand Uncertainty

A synthetic optimization project for transportation capacity planning under uncertain freight demand. The project compares nominal, expected-value, and robust planning policies, and connects the robust model to convex-concave saddle-point optimization and monotone inclusions.

> **Scope.** This is a synthetic portfolio project. It does **not** use Amazon data and does **not** attempt to reproduce Amazon's production transportation systems.

## Business motivation

A middle-mile transportation planner must reserve lane-level carrying capacity before future freight demand is fully known. Reserving too little capacity can create shortages and service failures, while reserving too much creates unnecessary reservation and idle-capacity costs.

The goal is to study how different planning policies trade average efficiency against protection from demand spikes. The default synthetic network contains 8 facilities, 18 directed lanes, 7 planning days, 100 training demand scenarios, and 500 independent test scenarios. Demand uncertainty combines a common daily shock, which creates correlation across lanes, with lane-specific random noise.

## Mathematical formulation

For demand scenario $s$, let $x$ denote the vector of reserved lane-day capacities. The scenario cost is

$$ C_s(x) = c^\top x + \lambda_{\mathrm{short}}\sum_j [d_{s,j}-x_j]_+^2 + \lambda_{\mathrm{idle}}\sum_j [x_j-d_{s,j}]_+^2,
$$
subject to
$$ 0 \le x \le \bar{x}, $$

where $c$ is the reservation-cost vector, $d_s$ is scenario demand, and $\bar{x}$ is the maximum available capacity.

Three planning policies are compared.

**Nominal planning**

$$ \min_x C_{\mathrm{forecast}}(x). $$

**Expected-value planning**

$$ \min_x \frac{1}{S}\sum_{s=1}^{S} C_s(x). $$

**Robust planning**

$$ \min_x \max_{s=1,\ldots,S} C_s(x). $$

For a finite set of training scenarios,

$$ \max_s C_s(x) = \max_{p\in\Delta_S}\sum_{s=1}^{S} p_s C_s(x),$$

where $\Delta_S = \left\{ p\in\mathbb{R}^S: p\ge 0, \mathbf{1}^\top p=1 \right\}$ is the probability simplex in $\mathbb{R}^S$.

Therefore the robust problem has the equivalent saddle-point form

$$ \min_{x\in X}\max_{p\in\Delta_S} L(x,p), \qquad L(x,p)=\sum_{s=1}^{S}p_sC_s(x). $$

The probability vector $p$ can be interpreted as an **adversarial weighting** of expensive training scenarios.

## Saddle operator and monotone inclusion

The saddle operator is

$$ F(x,p) =
\begin{bmatrix}
\sum_{s=1}^{S} p_s\nabla C_s(x)\\
-C(x)
\end{bmatrix},
$$

where

$$
C(x)
=
\begin{bmatrix}
C_1(x)\\
\vdots\\
C_S(x)
\end{bmatrix}.
$$

Let $X$ denote the capacity box and $\Delta_S$ the probability simplex. Saddle-point optimality can be written as the monotone inclusion

$$ 0 \in F(x,p) + N_X(x)\times N_{\Delta_S}(p),$$

where the normal cones enforce the capacity and simplex constraints.

For convergence diagnostics, the code uses the natural residual

$$ \left\| z-P\bigl(z-F(z)\bigr) \right\|, \qquad z=(x,p), $$

which vanishes at a projected fixed point of the optimality system.

## Algorithms

The robust model is solved with three methods:

- **Projected Extragradient:** the primary first-order saddle-point method. Predictor and corrector projections are implemented explicitly.
- **Projected gradient descent-ascent (GDA):** a deliberately simple simultaneous baseline.
- **SciPy SLSQP reference:** independently solves the equivalent epigraph formulation
  $$ \min_{x,t} t $$
  subject to
  $$  C_s(x)\le t,\qquad s=1,\ldots,S. $$

Nominal and expected-value policies are solved with SciPy L-BFGS-B.

All robust algorithms share the same simple API:

```python
result = solver(data, config)
```

A solver is selected with one string:

```python
algorithm = "extragradient"
result = run_solver(algorithm, data, config)
```

This makes it easy to replace Extragradient with another research algorithm without changing the transportation model or evaluation pipeline.

## Training and out-of-sample evaluation

Optimization uses **training scenarios only**. After a capacity plan is computed, its business performance is evaluated on a separate set of 500 independent test scenarios generated from the same synthetic demand process.

This separation avoids evaluating a policy only on the scenarios used to optimize it.

The reported metrics include reservation cost, mean/median/95th-percentile/worst test cost, shortage statistics, unused capacity, and a demand-weighted service level:

$$\text{service level} = 1-\frac{\text{total shortage}}{\text{total demand}}.$$

## Results

All numerical results below come from actual executions of this repository.

### Default experiment

The default experiment uses 20% demand uncertainty, 100 training scenarios, and 500 independent test scenarios.

| Policy | Reservation cost | Mean test cost | P95 test cost | Worst test cost | Mean shortage | Service level | Reserved capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 113.8 | 271.8 | 398.6 | 622.8 | 77.4 | 91.48% | 898.1 |
| Expected Value | 135.2 | 187.5 | 221.9 | 276.6 | 19.4 | 97.86% | 1069.0 |
| Robust (Extragradient) | 140.0 | 195.7 | 225.0 | 285.6 | 15.6 | 98.28% | 1108.7 |

Compared with nominal planning, both uncertainty-aware policies reserve more capacity but substantially reduce shortages and test-scenario costs.

Relative to expected-value planning, the robust Extragradient plan reserves approximately **3.7% more capacity**, reduces mean shortage by approximately **19.9%**, and increases service level from **97.86% to 98.28%**. In this particular independent test sample, however, the robust policy has a **3.3% higher worst test-scenario cost** than the expected-value policy. This illustrates that robustness with respect to a finite training-scenario set does not guarantee dominance on every independent finite test sample.

### Robust algorithm comparison

| Algorithm | Training robust objective | Natural residual | Iterations | Status |
|---|---:|---:|---:|---|
| Extragradient | 205.961 | \(9.91\times10^{-6}\) | 3600 | Converged |
| GDA | 206.279 | 1.17 | 8000 | Maximum iterations |
| SciPy reference | 205.961 | — | 22 | Converged |

Projected Extragradient reaches the \(10^{-5}\) natural-residual tolerance and matches the independent SciPy epigraph reference objective to the reported precision.

GDA obtains a relatively close robust objective but stalls at a natural residual of approximately \(1.17\). This illustrates why objective value alone is not sufficient for assessing convergence of a saddle-point method: the full pair \((x,p)\) must also satisfy the saddle-point optimality conditions.

SciPy is substantially faster on this small synthetic problem and is used as an independent reference rather than as an algorithmic baseline that Extragradient is expected to outperform in wall-clock time.

### Uncertainty stress test

The stress test repeats the experiment for uncertainty levels

$$ 0.05, 0.10, 0.20, 0.30, 0.40.$$

For each uncertainty level:

1. nominal, expected-value, and robust policies are recomputed;
2. the robust policy is solved using projected Extragradient;
3. SciPy independently solves the same robust problem as a numerical reference;
4. all business metrics are evaluated on independent test scenarios.

Extragradient matches the SciPy robust objective extremely closely across all five stress-test instances:

| Uncertainty | EG robust objective | SciPy robust objective | Relative objective gap | EG natural residual |
|---:|---:|---:|---:|---:|
| 0.05 | 118.7109 | 118.7109 | $1.93\times10^{-8}$ | $9.95\times10^{-6}$ |
| 0.10 | 135.1060 | 135.1060 | $1.10\times10^{-8}$ | $9.43\times10^{-6}$ |
| 0.20 | 186.1932 | 186.1932 | $6.95\times10^{-9}$ | $9.47\times10^{-6}$ |
| 0.30 | 261.9092 | 261.9092 | $1.89\times10^{-10}$ | $9.62\times10^{-6}$ |
| 0.40 | 359.8082 | 359.8082 | $3.69\times10^{-11}$ | $9.81\times10^{-6}$ |

The stress test also shows an increasingly visible tail-risk benefit from robust planning as uncertainty grows. At uncertainty level $0.40$, the robust Extragradient policy reserves approximately **4.1% more capacity** than the expected-value policy while reducing the worst observed test-scenario cost from approximately **780.3 to 687.9**, an **11.8% reduction**.

## Business interpretation

The experiments show a clear tradeoff between capacity cost and protection from uncertain demand.

Nominal planning reserves the least capacity, but it is highly exposed to demand shocks. In the default experiment, its worst test-scenario cost is 622.8, compared with 276.6 for expected-value planning and 285.6 for robust planning.

Expected-value planning performs best on average in the default test sample. Robust planning deliberately reserves additional capacity and accepts somewhat higher average cost in exchange for fewer shortages and stronger service reliability. Relative to expected-value planning, the default robust plan reserves 3.7% more capacity, reduces mean shortage by 19.9%, and raises service level from 97.86% to 98.28%.

The uncertainty stress test makes the tail-risk tradeoff more visible. As uncertainty increases, the robust policy becomes increasingly valuable for protection against extreme test-scenario costs. At the highest tested uncertainty level, robust planning reduces the worst observed test cost by 11.8% relative to expected-value planning while requiring approximately 4.1% additional reserved capacity.

The robust policy does **not** dominate expected-value planning on every metric or every finite test sample. Its value is instead the deliberate exchange of additional capacity and somewhat higher average cost for stronger shortage and tail-risk protection.

This is a small synthetic experiment. The results demonstrate the optimization tradeoff and algorithmic framework; they should not be interpreted as a production recommendation.

## Figures

Running the project generates:

- `figures/policy_cost_comparison.png` — mean, 95th-percentile, and worst test costs across policies
- `figures/service_level_comparison.png` — demand-weighted service levels
- `figures/robust_algorithm_convergence.png` — natural-residual convergence of Extragradient and GDA
- `figures/capacity_comparison.png` — weekly reserved capacity by transportation lane
- `figures/uncertainty_stress_test.png` — worst test-scenario cost as demand uncertainty increases

## Result files

The main numerical outputs are saved to:

- `results/policy_metrics.csv`
- `results/algorithm_metrics.csv`
- `results/uncertainty_stress_test.csv`
- `results/stress_test_solver_validation.csv`

## Repository structure

```text
robust-middle-mile/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── problem.py
│   ├── solvers.py
│   ├── evaluation.py
│   └── plots.py
├── experiments/
│   ├── __init__.py
│   └── stress_test.py
├── figures/
└── results/
```

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the complete experiment:

```bash
python main.py
```

`main.py` generates the synthetic network and demand scenarios, checks analytic gradients and projections, solves the three planning policies, compares robust optimization algorithms, evaluates the resulting plans on independent test scenarios, saves result tables, generates figures, and runs the uncertainty stress test.

To run only the stress test:

```bash
python experiments/stress_test.py
```

## How to add a new research algorithm

Open `src/solvers.py` and add one plain function with the same interface:

```python
def solve_my_algorithm(data, config):
    x = ...
    p = ...

    for k in range(...):
        # Your mathematically transparent iteration.
        ...

    return {
        "x": x,
        "p": p,
        "iterations": ...,
        "runtime": ...,
        "status": ...,
        "objective_history": ...,
        "residual_history": ...,
        "iteration_history": ...,
    }
```

Then register it:

```python
SOLVERS["my_algorithm"] = solve_my_algorithm
```

and switch algorithms with one string:

```python
algorithm = "my_algorithm"
result = run_solver(algorithm, data, config)
```

No base class, inheritance, factory, or optimization framework is required.

## Author

**Nghia Nguyen-Trung**  
Ph.D. Candidate in Statistics and Operations Research  
University of North Carolina at Chapel Hill