# NourishFlow

**A Multi-Agent Food Rescue and Allocation Optimization System**

NourishFlow is a student proof of concept for coordinating surplus food donations with community recipients. It allocates eligible food under supply, demand, storage, food-type compatibility, urgency, expiry-window and recipient-deadline constraints. The Streamlit dashboard is designed to be explainable and testable, with synthetic data, a visual allocation-flow chart, execution traces and downloadable outputs.

> **Safety scope:** The sample organizations, quantities and coordinates are synthetic. This is not an operational food-safety, dispatch, or donor/recipient verification system. Qualified people must verify food safety, handling conditions, eligibility and real delivery decisions.

## Current milestone

Implemented:
- Refreshed Streamlit UI with scenario selection, upload controls, input summaries, results, agent trace, field dictionary and workflow guide.
- Deterministic Data Intake, Planner, Allocation Optimizer, Critic, Correction and Report stages.
- Mixed-integer linear optimization using `scipy.optimize.milp`; allocations are whole kilograms.
- Constraint checks for supply, recipient demand, storage capacity, accepted food types and estimated delivery windows.
- Eligibility filtering that records why a donor-recipient link is excluded.
- Reproducible scenarios: baseline, time-critical food, donation disruption, supply shortage and invalid input.
- Critic fault injection that deliberately corrupts a plan, demonstrates rejection, and triggers one correction/retry.
- Interactive Plotly Sankey chart (food flow from donors to recipients), demand vs. allocation chart, donation-category chart and synthetic-coordinate view.
- CSV downloads for the allocation plan, execution log and starter datasets.
- Automated tests for baseline output, capacity constraints, invalid input, correction/retry and disruption behavior.

Next milestone:
- Add OR-Tools route optimization with vehicle capacities and time windows.
- Replan when vehicles or donations become unavailable.
- Compare optimized plans against a simple baseline and report quantitative performance.

## Project layout

```text
RescueRoute/
├── app.py
├── requirements.txt
├── README.md
├── rescue_route/
│   ├── __init__.py
│   ├── agents.py
│   ├── data.py
│   └── optimizer.py
├── sample_data/
│   ├── donations_sample.csv
│   └── recipients_sample.csv
└── tests/
    └── test_workflow.py
```

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

If you already created the virtual environment for the previous version, keep it and install the newly added dependency:

```powershell
python -m pip install -r requirements.txt
```

## Run automated tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Scenarios to demonstrate

1. **Baseline:** the optimizer allocates available food to compatible recipients while observing supply, storage and time constraints. The sample dataset contains 495 kg of listed available supply against 485 kg of effective recipient demand/capacity; in the baseline solution, 485 kg is allocated.
2. **Time-critical food:** a short food window and deadline alter the set of eligible allocation links.
3. **Donation disruption:** donation D03 is removed and the optimizer rebuilds allocations using remaining supply.
4. **Supply shortage:** donation quantities are reduced to 40%, making unmet demand visible.
5. **Invalid input:** the required `quantity_kg` column is removed, so Intake rejects the data.
6. **Critic + retry:** enable fault injection; the demo deliberately sets one candidate quantity to -1 kg. The Critic rejects it, the Correction Agent reruns optimization from the original input, and the second plan is validated.

## Data fields

### Donations

| Field | Meaning |
|---|---|
| `donation_id` | Unique ID for a donation record, e.g. D01. |
| `donor_name` | Fictional market, canteen, farm or warehouse making the offer. |
| `food_type` | Food category used for compatibility checks. |
| `quantity_kg` | Quantity offered, in kilograms. |
| `expiry_hours` | Simplified remaining planning window. It is not a food-safety determination. |
| `x_km`, `y_km` | Synthetic Cartesian coordinates used to estimate straight-line distance. |
| `available` | Whether the optimizer can use this donation. |

### Recipients

| Field | Meaning |
|---|---|
| `recipient_id` | Unique ID for a community recipient, e.g. R01. |
| `recipient_name` | Fictional shelter, kitchen or support organization. |
| `demand_kg` | Quantity requested. |
| `storage_capacity_kg` | Maximum quantity the recipient can store. |
| `urgency` | Integer from 1 (lower) to 5 (higher) used as a priority weight in the simulation. |
| `delivery_deadline_hours` | Latest modeled arrival time for that recipient. |
| `accepts_food_types` | Pipe-separated categories such as `Produce|Grains|Prepared meals`. |
| `x_km`, `y_km` | Synthetic coordinates used for distance estimates. |

### Calculated fields

- `allocated_kg`: whole-kilogram quantity assigned to one donation-recipient pair.
- `distance_km`: Euclidean distance between synthetic coordinates (not road distance).
- `estimated_delivery_hours`: a 0.25-hour handling buffer plus straight-line distance divided by an assumed average speed of 25 km/h.
- `objective_weight`: demonstration score per kilogram used by the optimizer to balance urgency, distance and a shorter remaining expiry window.

## Workflow

1. **Data Intake Agent:** validates schema, IDs, numeric values and value ranges. Invalid data is rejected before optimization.
2. **Planner Agent:** profiles available supply, recipients, urgency and effective recipient capacity (`min(demand_kg, storage_capacity_kg)`).
3. **Eligibility builder:** checks food compatibility and estimates distance/time. It excludes links that cannot meet expiry windows or recipient deadlines and records why.
4. **Allocation Optimizer:** solves a mixed-integer linear program. Each feasible donor-recipient link is an integer number of kilograms; supply and recipient capacity constraints are enforced.
5. **Critic Agent:** checks the candidate plan against source data and the implemented constraints.
6. **Correction Agent:** if validation fails, reruns optimization from the original data once; a second failure is escalated for manual review.
7. **Report Agent:** calculates allocated quantity, effective demand coverage and number of recipients served, then exposes the result and logs in the UI.

The agents are deterministic Python components in this milestone, not LLM-driven autonomous agents. Their agentic behavior is demonstrated through conditional workflow execution, tool use, state passing, validation, rejection, correction and retry. The numerical allocation is computed by the optimizer rather than an LLM.

## Important assumptions and current limits

The optimizer's demonstration score is:

`100 + 25 * urgency - 2 * distance_km + 8 / max(expiry_hours, 0.5)`

This is a configurable educational heuristic, not learned from historical outcomes or intended as official food-distribution policy. Since each feasible link gets a positive score, eligible demand is generally filled when supply permits.

The current milestone estimates straight-line donor-to-recipient distance and a simple delivery time. It **does not yet build multi-stop vehicle routes**, use live maps/traffic, or replan around a vehicle breakdown. Routing optimization is planned for the next milestone.
