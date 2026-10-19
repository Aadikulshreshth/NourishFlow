# NourishFlow

**A Multi-Agent Food Redistribution and Logistics Optimization System**

NourishFlow is a student proof of concept for coordinating surplus food donations with community recipients. It combines a mixed-integer allocation model with pickup-and-delivery vehicle routing, explicit validation, and simulated disruption recovery.

> **Safety scope:** The sample organizations, quantities, coordinates, fleet and travel times are synthetic. This is not an operational food-safety, dispatch, donor/recipient verification, or road-navigation system. Qualified people must verify food quality, handling conditions, eligibility and real delivery decisions.

## Current milestone

Implemented:
- Streamlit dashboard with scenario selection, CSV uploads, allocation flow visualizations, route sequence view, agent execution trace, data dictionary and workflow guide.
- Deterministic Intake, Planner, Allocation Optimizer, Allocation Critic, Correction, Vehicle Routing, Route Critic, Disruption/Replanning, Baseline Comparator and Report components.
- Mixed-integer linear allocation using `scipy.optimize.milp`; allocation quantities are whole kilograms.
- Pickup-and-delivery vehicle routing using Google OR-Tools `RoutingModel` when it is available.
- Explicitly labelled deadline-first greedy fallback, so the route demo can run in restricted environments where OR-Tools cannot be loaded.
- Vehicle capacity constraints, pickup-before-delivery precedence, same-vehicle shipment pairing, donation expiry windows and recipient delivery deadlines.
- Simulated vehicle breakdown: the busiest initially planned vehicle is marked unavailable, and all shipment routes are recalculated with the remaining fleet.
- Route Critic checks shipment completeness, pickup/delivery pairing, delivery order, time windows, vehicle load, total delivered quantity, and return-to-hub load.
- Greedy route baseline for distance comparisons.
- Reproducible scenarios: Baseline, Time-critical food, Donation disruption, Supply shortage, Vehicle breakdown / replan, and Invalid input.
- Critic fault injection for allocation validation and one correction retry.
- CSV downloads for allocation and route results, plus sample-data downloads.
- Automated tests for allocation validity, routing, vehicle capacity, invalid input, correction/retry, donation disruption, shortage and vehicle-breakdown replanning.

## Project layout

```text
NourishFlow/
├── app.py
├── requirements.txt
├── README.md
├── rescue_route/
│   ├── agents.py
│   ├── data.py
│   ├── optimizer.py
│   └── routing.py
└── tests/
    └── test_workflow.py
```

## Run on Windows PowerShell

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If the virtual environment already exists, do not recreate it. Activate it and install updated dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Run automated tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Scenarios

1. **Baseline:** allocate eligible donations, build routes and validate both plans.
2. **Time-critical food:** tighten D04's remaining window and R04's deadline; some allocation links become infeasible.
3. **Donation disruption:** remove donation D03 and allocate with the remaining supply.
4. **Supply shortage:** reduce donation quantities to 40% to demonstrate unmet demand.
5. **Vehicle breakdown / replan:** compute the initial routes, mark the vehicle carrying the largest amount unavailable, and re-optimize the route plan with the remaining fleet. If it cannot find a feasible route plan, the system escalates instead of claiming success.
6. **Invalid input:** remove required `quantity_kg` to show that intake validation rejects input before optimization.
7. **Critic + retry:** enable the sidebar fault-injection checkbox. It changes a candidate allocation quantity to -1 kg, the Allocation Critic rejects it, and the Correction Agent recomputes from trusted inputs.

## Routing model and assumptions

- Default fleet: three synthetic vehicles, each with a 250 kg capacity. Both fleet size and capacity can be changed in the sidebar.
- Each donor-recipient allocation row is represented as a distinct shipment with a pickup node and a delivery node. The pair must use the same vehicle, and the pickup must occur first.
- A shipment's vehicle load increases at pickup and decreases at delivery; simultaneous load may not exceed vehicle capacity.
- A recipient may receive shipments from multiple donors and vehicles.
- The synthetic hub is located at the centroid of the sites involved in the plan.
- Coordinates are two-dimensional grid values in kilometres. Travel distance is Euclidean, not road-network distance.
- Assumed speed is 25 km/h. Pickup service is 4 minutes per shipment; delivery service is 3 minutes per shipment.
- Delivery arrival must not exceed the smaller of the donation's remaining planning window and the recipient deadline.
- The routing baseline is a simple deadline-first greedy pickup/delivery schedule using the same allocation and fleet. It is a simulation benchmark, not a claim about real delivery operations.
- The breakdown scenario removes the busiest planned vehicle and recomputes every shipment route using the remaining fleet. The food allocation is retained; if routing becomes infeasible, the app records the failure for manual review.
- If OR-Tools is not available or it does not return a feasible result, the app may use a clearly labelled greedy fallback. The UI and metadata disclose which solver was used.

## Allocation model

Each feasible donor-recipient link has an integer decision variable in kilograms. The objective score is:

\[
\text{score}_{ij}=100+25u_j-2d_{ij}+\frac{8}{\max(e_i,0.5)}
\]

where `u_j` is recipient urgency (1–5), `d_ij` is straight-line donor-recipient distance, and `e_i` is the remaining donor planning window. The score weights are illustrative choices for this educational prototype, not an official food distribution policy.

## CSV input fields

**Donations:** `donation_id`, `donor_name`, `food_type`, `quantity_kg`, `expiry_hours`, `x_km`, `y_km`, `available`.

**Recipients:** `recipient_id`, `recipient_name`, `demand_kg`, `storage_capacity_kg`, `urgency`, `delivery_deadline_hours`, `accepts_food_types`, `x_km`, `y_km`.

`accepts_food_types` is a pipe-separated list such as `Produce|Grains|Prepared meals`. Coordinates are synthetic local-grid kilometres.

## Limitations

The workflow uses deterministic Python components; the agents are not independent LLMs. SciPy solves food allocation, and OR-Tools is the preferred vehicle routing solver. Neither solver uses a real road network, traffic, live vehicle telemetry, verified organization addresses or actual food-safety records. A greedy fallback is included for environments where OR-Tools is unavailable. Real-world operation would require reliable data integration, route recalculation against a road network, food-safety procedures, privacy controls, monitoring and human approval.
