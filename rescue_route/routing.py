"""Pickup-and-delivery routing for NourishFlow.

The preferred solver is Google OR-Tools RoutingModel. A clearly labelled,
deterministic deadline-first fallback is provided so the demo remains usable
in restricted environments where OR-Tools could not be installed.

Coordinates and travel times are synthetic assumptions, not road-network ETAs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import ceil, hypot
from typing import Any

import pandas as pd

try:  # Optional at import time; requirements.txt installs OR-Tools for normal use.
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the runtime environment
    pywrapcp = None  # type: ignore[assignment]
    routing_enums_pb2 = None  # type: ignore[assignment]
    ORTOOLS_AVAILABLE = False

ROUTE_COLUMNS = [
    "vehicle_id", "stop_order", "stop_type", "location_id", "location_name",
    "donation_id", "recipient_id", "food_type", "quantity_kg", "arrival_minutes",
    "arrival_hours", "load_after_kg", "leg_distance_km", "x_km", "y_km",
    "deadline_hours", "expiry_hours",
]

DEFAULT_SPEED_KMH = 25.0
PICKUP_SERVICE_MINUTES = 4
DELIVERY_SERVICE_MINUTES = 3
DEFAULT_VEHICLE_COUNT = 3
DEFAULT_VEHICLE_CAPACITY_KG = 250


@dataclass
class RouteValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def default_fleet(vehicle_count: int = DEFAULT_VEHICLE_COUNT,
                  capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG) -> list[dict[str, Any]]:
    """Return a simple synthetic fleet with a shared carrying capacity."""
    if vehicle_count < 1:
        raise ValueError("vehicle_count must be at least 1")
    if capacity_kg < 1:
        raise ValueError("capacity_kg must be positive")
    return [
        {"vehicle_id": f"V{i + 1:02d}", "capacity_kg": int(capacity_kg)}
        for i in range(vehicle_count)
    ]


def _available_vehicle_specs(vehicle_specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    specs = deepcopy(vehicle_specs if vehicle_specs is not None else default_fleet())
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, spec in enumerate(specs):
        vehicle_id = str(spec.get("vehicle_id", f"V{index + 1:02d}"))
        capacity = int(spec.get("capacity_kg", DEFAULT_VEHICLE_CAPACITY_KG))
        if not vehicle_id.strip() or vehicle_id in seen:
            raise ValueError("Vehicle IDs must be non-empty and unique.")
        if capacity <= 0:
            raise ValueError(f"Capacity for vehicle {vehicle_id} must be positive.")
        normalized.append({"vehicle_id": vehicle_id, "capacity_kg": capacity})
        seen.add(vehicle_id)
    return normalized


def _distance_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    return hypot(float(a["x_km"]) - float(b["x_km"]),
                 float(a["y_km"]) - float(b["y_km"]))


def _travel_minutes(distance_km: float, speed_kmh: float = DEFAULT_SPEED_KMH) -> int:
    if distance_km <= 0:
        return 0
    return max(1, int(ceil((distance_km / speed_kmh) * 60.0)))


def _route_failure(message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "status": "infeasible",
        "stops": pd.DataFrame(columns=ROUTE_COLUMNS),
        "summary": {
            "vehicles_used": 0,
            "vehicles_available": len((metadata or {}).get("vehicle_specs", [])),
            "total_distance_km": 0.0,
            "total_delivered_kg": 0,
            "delivery_events": 0,
            "max_vehicle_load_kg": 0,
            "max_route_duration_hours": 0.0,
        },
        "metadata": {**(metadata or {}), "failure_reason": message},
        "validation": RouteValidationResult(False, [message], []),
    }


def _prepare_routing_problem(
    plan: pd.DataFrame,
    donations: pd.DataFrame,
    recipients: pd.DataFrame,
    vehicle_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    required_plan = {"donation_id", "recipient_id", "allocated_kg"}
    if not required_plan.issubset(plan.columns):
        raise ValueError("Allocation plan must contain donation_id, recipient_id and allocated_kg.")
    if plan.empty:
        raise ValueError("There are no allocation rows to route.")

    donations = donations.copy()
    recipients = recipients.copy()
    donations["donation_id"] = donations["donation_id"].astype(str)
    recipients["recipient_id"] = recipients["recipient_id"].astype(str)
    plan = plan.copy()
    plan["donation_id"] = plan["donation_id"].astype(str)
    plan["recipient_id"] = plan["recipient_id"].astype(str)
    plan["allocated_kg"] = pd.to_numeric(plan["allocated_kg"], errors="coerce")
    if plan["allocated_kg"].isna().any() or (plan["allocated_kg"] <= 0).any():
        raise ValueError("Every routed allocation must have a positive numeric quantity.")
    if (plan["allocated_kg"] % 1 != 0).any():
        raise ValueError("Routing requires whole-kilogram allocation quantities.")

    donation_by_id = donations.set_index("donation_id", drop=False)
    recipient_by_id = recipients.set_index("recipient_id", drop=False)
    unknown_donations = sorted(set(plan["donation_id"]) - set(donation_by_id.index))
    unknown_recipients = sorted(set(plan["recipient_id"]) - set(recipient_by_id.index))
    if unknown_donations or unknown_recipients:
        raise ValueError(f"Unknown IDs in allocation plan. Donations={unknown_donations}; recipients={unknown_recipients}.")

    plan = plan.reset_index(drop=True)
    pickup_nodes_for_rows: dict[int, int] = {}
    delivery_nodes_for_rows: dict[int, int] = {}

    used_donations = donations[donations["donation_id"].isin(set(plan["donation_id"]))].copy()
    used_recipient_ids = set(plan["recipient_id"])
    used_recipients = recipients[recipients["recipient_id"].isin(used_recipient_ids)].copy()
    site_x = list(pd.to_numeric(used_donations["x_km"], errors="coerce")) + list(pd.to_numeric(used_recipients["x_km"], errors="coerce"))
    site_y = list(pd.to_numeric(used_donations["y_km"], errors="coerce")) + list(pd.to_numeric(used_recipients["y_km"], errors="coerce"))
    depot_x = float(sum(site_x) / len(site_x))
    depot_y = float(sum(site_y) / len(site_y))

    # Node 0 is a synthetic central hub. Every donor-recipient allocation line
    # becomes a unique shipment with a unique pickup and delivery node. This is
    # compatible with OR-Tools' one-pickup/one-delivery pairing and permits split
    # shipments to travel on different vehicles when required.
    nodes: list[dict[str, Any]] = [{
        "node_type": "depot", "location_id": "HUB", "location_name": "Synthetic central hub",
        "donation_id": "", "recipient_id": "", "food_type": "", "quantity_kg": 0,
        "load_delta": 0, "x_km": depot_x, "y_km": depot_y,
        "service_minutes": 0, "latest_minutes": 24 * 60, "deadline_hours": None,
        "expiry_hours": None, "allocation_row": None,
    }]

    for row_number, allocation in plan.iterrows():
        donation = donation_by_id.loc[str(allocation["donation_id"])]
        recipient = recipient_by_id.loc[str(allocation["recipient_id"])]
        quantity = int(allocation["allocated_kg"])
        expiry = float(donation["expiry_hours"])
        deadline = float(recipient["delivery_deadline_hours"])
        node_id = len(nodes)
        pickup_nodes_for_rows[int(row_number)] = node_id
        nodes.append({
            "node_type": "pickup", "location_id": str(donation["donation_id"]),
            "location_name": str(donation["donor_name"]), "donation_id": str(allocation["donation_id"]),
            "recipient_id": str(allocation["recipient_id"]), "food_type": str(donation["food_type"]),
            "quantity_kg": quantity, "load_delta": quantity,
            "x_km": float(donation["x_km"]), "y_km": float(donation["y_km"]),
            "service_minutes": PICKUP_SERVICE_MINUTES,
            "latest_minutes": max(0, int(min(expiry, deadline) * 60)),
            "deadline_hours": deadline, "expiry_hours": expiry,
            "allocation_row": int(row_number),
        })

    for row_number, allocation in plan.iterrows():
        donation = donation_by_id.loc[str(allocation["donation_id"])]
        recipient = recipient_by_id.loc[str(allocation["recipient_id"])]
        expiry = float(donation["expiry_hours"])
        deadline = float(recipient["delivery_deadline_hours"])
        latest = max(0, int(min(expiry, deadline) * 60))
        node_id = len(nodes)
        delivery_nodes_for_rows[int(row_number)] = node_id
        nodes.append({
            "node_type": "delivery", "location_id": str(recipient["recipient_id"]),
            "location_name": str(recipient["recipient_name"]),
            "donation_id": str(allocation["donation_id"]),
            "recipient_id": str(allocation["recipient_id"]),
            "food_type": str(donation["food_type"]),
            "quantity_kg": int(allocation["allocated_kg"]),
            "load_delta": -int(allocation["allocated_kg"]),
            "x_km": float(recipient["x_km"]), "y_km": float(recipient["y_km"]),
            "service_minutes": DELIVERY_SERVICE_MINUTES,
            "latest_minutes": latest, "deadline_hours": deadline,
            "expiry_hours": expiry, "allocation_row": int(row_number),
        })

    pair_nodes = [
        (pickup_nodes_for_rows[int(row_number)], delivery_nodes_for_rows[int(row_number)])
        for row_number in plan.index
    ]
    for row in plan.itertuples(index=False):
        quantity = int(row.allocated_kg)
        if quantity > max(v["capacity_kg"] for v in vehicle_specs):
            raise ValueError(
                f"Shipment {row.donation_id} → {row.recipient_id} is {quantity} kg, "
                f"which exceeds every available vehicle capacity. Increase capacity or reduce shipment size."
            )

    # Each allocation row is a shipment. A shipment's pickup and delivery must
    # use the same vehicle; separate shipment rows may be split across vehicles.
    horizon_hours = max(
        24.0,
        max(float(recipients.loc[recipients["recipient_id"].astype(str) == rid, "delivery_deadline_hours"].iloc[0]) for rid in used_recipient_ids) + 2.0,
    )
    return {
        "plan": plan,
        "nodes": nodes,
        "pickup_nodes_for_rows": pickup_nodes_for_rows,
        "delivery_nodes_for_rows": delivery_nodes_for_rows,
        "pair_nodes": pair_nodes,
        "vehicle_specs": vehicle_specs,
        "depot_x_km": depot_x,
        "depot_y_km": depot_y,
        "horizon_minutes": int(ceil(horizon_hours * 60)),
        "speed_kmh": DEFAULT_SPEED_KMH,
        "pickup_service_minutes": PICKUP_SERVICE_MINUTES,
        "delivery_service_minutes": DELIVERY_SERVICE_MINUTES,
    }


def _event_record(vehicle_id: str, stop_order: int, stop_type: str, node: dict[str, Any],
                  arrival_minutes: int, load_after_kg: int, leg_distance_km: float) -> dict[str, Any]:
    return {
        "vehicle_id": vehicle_id,
        "stop_order": int(stop_order),
        "stop_type": stop_type,
        "location_id": node.get("location_id", ""),
        "location_name": node.get("location_name", ""),
        "donation_id": node.get("donation_id", ""),
        "recipient_id": node.get("recipient_id", ""),
        "food_type": node.get("food_type", ""),
        "quantity_kg": int(node.get("quantity_kg", 0)),
        "arrival_minutes": int(arrival_minutes),
        "arrival_hours": round(float(arrival_minutes) / 60.0, 3),
        "load_after_kg": int(load_after_kg),
        "leg_distance_km": round(float(leg_distance_km), 3),
        "x_km": float(node.get("x_km", 0.0)),
        "y_km": float(node.get("y_km", 0.0)),
        "deadline_hours": node.get("deadline_hours"),
        "expiry_hours": node.get("expiry_hours"),
    }


def _summarize_routes(stops: pd.DataFrame, vehicle_specs: list[dict[str, Any]]) -> dict[str, Any]:
    if stops.empty:
        return {
            "vehicles_used": 0, "vehicles_available": len(vehicle_specs),
            "total_distance_km": 0.0, "total_delivered_kg": 0,
            "delivery_events": 0, "max_vehicle_load_kg": 0,
            "max_route_duration_hours": 0.0,
        }
    deliveries = stops[stops["stop_type"] == "Delivery"]
    active_vehicle_ids = set(stops.loc[stops["stop_type"].isin(["Pickup", "Delivery"]), "vehicle_id"])
    peak_load = int(stops["load_after_kg"].max()) if not stops.empty else 0
    return {
        "vehicles_used": len(active_vehicle_ids),
        "vehicles_available": len(vehicle_specs),
        "total_distance_km": round(float(stops["leg_distance_km"].sum()), 2),
        "total_delivered_kg": int(deliveries["quantity_kg"].sum()) if not deliveries.empty else 0,
        "delivery_events": int(len(deliveries)),
        "max_vehicle_load_kg": peak_load,
        "max_route_duration_hours": round(float(stops[stops["stop_type"] == "Depot return"]["arrival_hours"].max()), 2)
        if (stops["stop_type"] == "Depot return").any() else 0.0,
    }


def _base_metadata(problem: dict[str, Any], solver_name: str, fallback_used: bool) -> dict[str, Any]:
    return {
        "solver": solver_name,
        "fallback_used": fallback_used,
        "vehicle_specs": deepcopy(problem["vehicle_specs"]),
        "depot_x_km": round(problem["depot_x_km"], 3),
        "depot_y_km": round(problem["depot_y_km"], 3),
        "average_speed_kmh": problem["speed_kmh"],
        "pickup_service_minutes_per_shipment": problem["pickup_service_minutes"],
        "delivery_service_minutes_per_shipment": problem["delivery_service_minutes"],
        "time_windows": "Delivery arrival must be no later than min(donation expiry window, recipient deadline).",
        "capacity_model": "Each allocation line is a pickup-and-delivery shipment; each pair stays on one vehicle, and load falls after its delivery.",
    }


def _optimize_with_ortools(
    problem: dict[str, Any],
    time_limit_seconds: int = 4,
    require_each_vehicle_used: bool = False,
) -> dict[str, Any]:
    nodes = problem["nodes"]
    vehicle_specs = problem["vehicle_specs"]
    manager = pywrapcp.RoutingIndexManager(len(nodes), len(vehicle_specs), 0)
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix_m: list[list[int]] = []
    time_matrix_min: list[list[int]] = []
    for a in nodes:
        distance_row: list[int] = []
        time_row: list[int] = []
        for b in nodes:
            distance = _distance_km(a, b)
            distance_row.append(int(round(distance * 1000.0)))
            time_row.append(_travel_minutes(distance, problem["speed_kmh"]))
        distance_matrix_m.append(distance_row)
        time_matrix_min.append(time_row)

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix_m[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return time_matrix_min[from_node][manager.IndexToNode(to_index)] + int(nodes[from_node]["service_minutes"])

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        problem["horizon_minutes"],
        problem["horizon_minutes"],
        True,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    def demand_callback(from_index: int) -> int:
        node_id = manager.IndexToNode(from_index)
        return int(nodes[node_id]["load_delta"])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    capacities = [int(v["capacity_kg"]) for v in vehicle_specs]
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, capacities, True, "Load")
    load_dimension = routing.GetDimensionOrDie("Load")

    for node_id in range(1, len(nodes)):
        index = manager.NodeToIndex(node_id)
        latest = min(int(nodes[node_id]["latest_minutes"]), problem["horizon_minutes"])
        time_dimension.CumulVar(index).SetRange(0, latest)
        load_dimension.CumulVar(index).SetRange(0, max(capacities))

    solver = routing.solver()
    if require_each_vehicle_used:
        # The breakdown demo deliberately distributes the initial workload across
        # the fleet so the effect of removing a vehicle is visible and testable.
        # Outside that scenario, vehicle use remains fully optimized as usual.
        vehicles_to_activate = min(len(problem["plan"]), len(vehicle_specs))
        for vehicle_index in range(vehicles_to_activate):
            solver.Add(routing.ActiveVehicleVar(vehicle_index) == 1)
    for pickup_node, delivery_node in problem["pair_nodes"]:
        pickup_index = manager.NodeToIndex(pickup_node)
        delivery_index = manager.NodeToIndex(delivery_node)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        solver.Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = max(1, int(time_limit_seconds))
    search_parameters.log_search = False
    solution = routing.SolveWithParameters(search_parameters)
    metadata = _base_metadata(problem, "Google OR-Tools RoutingModel", False)
    metadata["search_method"] = "Pickup-and-delivery VRP; distance objective; load and time dimensions"
    metadata["fleet_distribution_for_breakdown_demo"] = bool(require_each_vehicle_used)
    if solution is None:
        return _route_failure("OR-Tools did not find a feasible route plan under the configured capacities and time windows.", metadata)

    records: list[dict[str, Any]] = []
    for vehicle_index, vehicle in enumerate(vehicle_specs):
        index = routing.Start(vehicle_index)
        vehicle_id = str(vehicle["vehicle_id"])
        stop_order = 0
        current_load = 0
        previous_node_id = 0
        start_node = nodes[0]
        records.append(_event_record(vehicle_id, stop_order, "Depot start", start_node, 0, 0, 0.0))
        used = False
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            next_node_id = manager.IndexToNode(next_index)
            if routing.IsEnd(next_index):
                end_time = int(solution.Value(time_dimension.CumulVar(next_index)))
                leg_distance = _distance_km(nodes[previous_node_id], nodes[0])
                total_load = current_load
                stop_order += 1
                records.append(_event_record(vehicle_id, stop_order, "Depot return", nodes[0], end_time, total_load, leg_distance))
                break

            node = nodes[next_node_id]
            arrival = int(solution.Value(time_dimension.CumulVar(next_index)))
            leg_distance = _distance_km(nodes[previous_node_id], node)
            current_load += int(node["load_delta"])
            used = True
            stop_order += 1
            stop_type = "Pickup" if node["node_type"] == "pickup" else "Delivery"
            records.append(_event_record(vehicle_id, stop_order, stop_type, node, arrival, current_load, leg_distance))
            previous_node_id = next_node_id
            index = next_index

        # Only retain used vehicle routes. Unused vehicle specs remain in metadata.
        if not used:
            records = [row for row in records if row["vehicle_id"] != vehicle_id]

    stops = pd.DataFrame(records, columns=ROUTE_COLUMNS)
    summary = _summarize_routes(stops, vehicle_specs)
    return {
        "success": True,
        "status": "feasible",
        "stops": stops,
        "summary": summary,
        "metadata": {**metadata, "objective_distance_m": int(solution.ObjectiveValue())},
        "validation": None,
    }


def _simulate_group(state: dict[str, Any], pickup_node: dict[str, Any], delivery_nodes: list[dict[str, Any]],
                    vehicle: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any] | None:
    """Try appending one donor's pickup and all its delivery shipments to a route."""
    candidate = deepcopy(state)
    cap = int(vehicle["capacity_kg"])
    current_location = {"x_km": candidate["x_km"], "y_km": candidate["y_km"]}
    pickup_distance = _distance_km(current_location, pickup_node)
    pickup_arrival = int(candidate["time_min"]) + _travel_minutes(pickup_distance, problem["speed_kmh"])
    if pickup_arrival > int(pickup_node["latest_minutes"]):
        return None
    pickup_load = int(candidate["load_kg"]) + int(pickup_node["load_delta"])
    if pickup_load > cap:
        return None
    candidate["distance_km"] += pickup_distance
    candidate["time_min"] = pickup_arrival
    candidate["load_kg"] = pickup_load
    candidate["max_load_kg"] = max(int(candidate["max_load_kg"]), pickup_load)
    candidate["events"].append(("Pickup", pickup_node, pickup_arrival, pickup_load, pickup_distance))
    candidate["time_min"] += int(pickup_node["service_minutes"])
    candidate["x_km"] = float(pickup_node["x_km"])
    candidate["y_km"] = float(pickup_node["y_km"])

    ordered_deliveries = sorted(
        delivery_nodes,
        key=lambda node: (
            int(node["latest_minutes"]),
            _distance_km(pickup_node, node),
            str(node["recipient_id"]),
        ),
    )
    for delivery_node in ordered_deliveries:
        current_location = {"x_km": candidate["x_km"], "y_km": candidate["y_km"]}
        leg = _distance_km(current_location, delivery_node)
        arrival = int(candidate["time_min"]) + _travel_minutes(leg, problem["speed_kmh"])
        if arrival > int(delivery_node["latest_minutes"]):
            return None
        new_load = int(candidate["load_kg"]) + int(delivery_node["load_delta"])
        if new_load < 0 or new_load > cap:
            return None
        candidate["distance_km"] += leg
        candidate["time_min"] = arrival
        candidate["load_kg"] = new_load
        candidate["events"].append(("Delivery", delivery_node, arrival, new_load, leg))
        candidate["time_min"] += int(delivery_node["service_minutes"])
        candidate["x_km"] = float(delivery_node["x_km"])
        candidate["y_km"] = float(delivery_node["y_km"])

    if candidate["load_kg"] != 0:
        return None
    candidate["groups_assigned"] += 1
    return candidate


def _optimize_with_greedy_fallback(
    problem: dict[str, Any],
    reason: str | None = None,
    require_each_vehicle_used: bool = False,
) -> dict[str, Any]:
    """Deadline-first heuristic fallback; not represented as an OR-Tools result."""
    nodes = problem["nodes"]
    plan = problem["plan"]
    fleet = problem["vehicle_specs"]
    metadata = _base_metadata(problem, "Greedy fallback (OR-Tools unavailable or did not return a solution)", True)
    metadata["fleet_distribution_for_breakdown_demo"] = bool(require_each_vehicle_used)
    if reason:
        metadata["fallback_reason"] = reason

    node_by_id = {i: node for i, node in enumerate(nodes)}
    delivery_ids_by_row = problem["delivery_nodes_for_rows"]
    groups: list[dict[str, Any]] = []
    pickup_ids_by_row = problem["pickup_nodes_for_rows"]
    delivery_ids_by_row = problem["delivery_nodes_for_rows"]
    for row_index, allocation in plan.iterrows():
        pickup_node = node_by_id[pickup_ids_by_row[int(row_index)]]
        delivery_node = node_by_id[delivery_ids_by_row[int(row_index)]]
        groups.append({
            "shipment_id": f"{allocation['donation_id']}->{allocation['recipient_id']}",
            "donation_id": str(allocation["donation_id"]),
            "pickup_node": pickup_node,
            "delivery_nodes": [delivery_node],
            "tightest_window": min(int(pickup_node["latest_minutes"]), int(delivery_node["latest_minutes"])),
            "total_kg": int(allocation["allocated_kg"]),
        })
    groups.sort(key=lambda g: (g["tightest_window"], -g["total_kg"], g["donation_id"]))

    hub = nodes[0]
    states: dict[str, dict[str, Any]] = {}
    for vehicle in fleet:
        vehicle_id = str(vehicle["vehicle_id"])
        states[vehicle_id] = {
            "vehicle_id": vehicle_id, "time_min": 0, "x_km": float(hub["x_km"]),
            "y_km": float(hub["y_km"]), "load_kg": 0, "max_load_kg": 0,
            "distance_km": 0.0, "events": [], "groups_assigned": 0,
        }

    for group_index, group in enumerate(groups):
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        empty_vehicle_ids = {
            vehicle_id for vehicle_id, state in states.items()
            if int(state["groups_assigned"]) == 0
        }
        for vehicle in fleet:
            vehicle_id = str(vehicle["vehicle_id"])
            state = states[vehicle_id]
            candidate = _simulate_group(state, group["pickup_node"], group["delivery_nodes"], vehicle, problem)
            if candidate is None:
                continue
            added_distance = candidate["distance_km"] - state["distance_km"]
            # In the breakdown demo, seed one shipment on each vehicle first so the
            # simulated failure has real work to redistribute. Normal runs keep the
            # standard deadline-first heuristic and its modest balancing term.
            if require_each_vehicle_used and empty_vehicle_ids and vehicle_id not in empty_vehicle_ids:
                continue
            score = added_distance + 0.15 * candidate["time_min"] / 60.0 + 1.5 * state["groups_assigned"]
            candidates.append((score, vehicle_id, candidate))
        if not candidates and require_each_vehicle_used and empty_vehicle_ids:
            # Different capacity profiles can make an empty vehicle unsuitable for
            # this shipment; fall back to all feasible candidates instead of failing
            # solely because demo balancing could not be achieved.
            for vehicle in fleet:
                vehicle_id = str(vehicle["vehicle_id"])
                state = states[vehicle_id]
                candidate = _simulate_group(state, group["pickup_node"], group["delivery_nodes"], vehicle, problem)
                if candidate is None:
                    continue
                added_distance = candidate["distance_km"] - state["distance_km"]
                score = added_distance + 0.15 * candidate["time_min"] / 60.0 + 1.5 * state["groups_assigned"]
                candidates.append((score, vehicle_id, candidate))
        if not candidates:
            return _route_failure(
                f"Greedy fallback could not place donation {group['donation_id']} within vehicle capacity and time windows.",
                metadata,
            )
        _, vehicle_id, best_state = min(candidates, key=lambda item: (item[0], item[1]))
        states[vehicle_id] = best_state

    records: list[dict[str, Any]] = []
    for vehicle in fleet:
        vehicle_id = str(vehicle["vehicle_id"])
        state = states[vehicle_id]
        if not state["events"]:
            continue
        stop_order = 0
        records.append(_event_record(vehicle_id, stop_order, "Depot start", hub, 0, 0, 0.0))
        for stop_type, node, arrival, load_after, leg_distance in state["events"]:
            stop_order += 1
            records.append(_event_record(vehicle_id, stop_order, stop_type, node, arrival, load_after, leg_distance))
        current_location = {"x_km": state["x_km"], "y_km": state["y_km"]}
        return_distance = _distance_km(current_location, hub)
        return_arrival = int(state["time_min"]) + _travel_minutes(return_distance, problem["speed_kmh"])
        stop_order += 1
        records.append(_event_record(vehicle_id, stop_order, "Depot return", hub, return_arrival, int(state["load_kg"]), return_distance))

    stops = pd.DataFrame(records, columns=ROUTE_COLUMNS)
    return {
        "success": True,
        "status": "feasible_fallback",
        "stops": stops,
        "summary": _summarize_routes(stops, fleet),
        "metadata": {
            **metadata,
            "search_method": (
                "Deadline-first heuristic with fleet-distribution seeding for breakdown demo"
                if require_each_vehicle_used else "Deadline-first pickup-and-delivery heuristic"
            ),
        },
        "validation": None,
    }


def optimize_vehicle_routes(
    plan: pd.DataFrame,
    donations: pd.DataFrame,
    recipients: pd.DataFrame,
    vehicle_specs: list[dict[str, Any]] | None = None,
    time_limit_seconds: int = 4,
    force_greedy: bool = False,
    require_each_vehicle_used: bool = False,
) -> dict[str, Any]:
    """Optimize pickup-and-delivery routes subject to load and arrival-time limits.

    Each allocation row is modelled as a unique pickup-and-delivery shipment. A
    recipient may receive shipments from different donors/vehicles. OR-Tools is
    used when installed; otherwise a labelled greedy fallback keeps the demo usable.
    """
    try:
        fleet = _available_vehicle_specs(vehicle_specs)
    except ValueError as exc:
        return _route_failure(str(exc), {"vehicle_specs": vehicle_specs or []})
    if not fleet:
        return _route_failure("No available vehicles remain for routing.", {"vehicle_specs": []})
    if plan.empty:
        return _route_failure("No allocation rows are available for routing.", {"vehicle_specs": fleet})
    try:
        problem = _prepare_routing_problem(plan, donations, recipients, fleet)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return _route_failure(str(exc), {"vehicle_specs": fleet})

    if require_each_vehicle_used:
        # One shipment per active vehicle is required only for the scripted
        # breakdown demonstration, not for ordinary routing runs.
        problem["require_each_vehicle_used"] = True
    if force_greedy:
        return _optimize_with_greedy_fallback(
            problem,
            reason="Greedy benchmark requested for comparison.",
            require_each_vehicle_used=require_each_vehicle_used,
        )
    if ORTOOLS_AVAILABLE:
        try:
            result = _optimize_with_ortools(
                problem,
                time_limit_seconds=time_limit_seconds,
                require_each_vehicle_used=require_each_vehicle_used,
            )
            if result["success"]:
                return result
            # Fallback is explicit in metadata; it only takes over if OR-Tools did
            # not produce a feasible plan for this small demonstration instance.
            fallback = _optimize_with_greedy_fallback(
                problem,
                reason=result["metadata"].get("failure_reason"),
                require_each_vehicle_used=require_each_vehicle_used,
            )
            return fallback if fallback["success"] else result
        except Exception as exc:  # Keep UI usable and expose the solver issue in metadata.
            fallback = _optimize_with_greedy_fallback(
                problem,
                reason=f"OR-Tools execution error: {type(exc).__name__}: {exc}",
                require_each_vehicle_used=require_each_vehicle_used,
            )
            return fallback
    return _optimize_with_greedy_fallback(
        problem,
        reason="OR-Tools is not installed in this environment.",
        require_each_vehicle_used=require_each_vehicle_used,
    )


def validate_route_plan(route_result: dict[str, Any], plan: pd.DataFrame) -> RouteValidationResult:
    """Critic checks route completeness, precedence, deadlines and vehicle load."""
    errors: list[str] = []
    warnings: list[str] = []
    if not route_result.get("success", False):
        errors.append(str(route_result.get("metadata", {}).get("failure_reason", "Routing did not produce a feasible plan.")))
        return RouteValidationResult(False, errors, warnings)
    stops = route_result.get("stops", pd.DataFrame(columns=ROUTE_COLUMNS))
    if stops.empty:
        errors.append("Routing marked the plan feasible but returned no stops.")
        return RouteValidationResult(False, errors, warnings)

    vehicle_specs = route_result.get("metadata", {}).get("vehicle_specs", [])
    capacity_by_vehicle = {str(v["vehicle_id"]): int(v["capacity_kg"]) for v in vehicle_specs}
    pickup_stops = stops[stops["stop_type"] == "Pickup"]
    delivery_stops = stops[stops["stop_type"] == "Delivery"]
    plan = plan.copy()
    plan["donation_id"] = plan["donation_id"].astype(str)
    plan["recipient_id"] = plan["recipient_id"].astype(str)

    # Every donor-recipient allocation becomes one pickup-and-delivery shipment.
    expected_pickups = plan.groupby("donation_id", as_index=False)["allocated_kg"].sum()
    for row in expected_pickups.itertuples(index=False):
        actual = pickup_stops[pickup_stops["donation_id"].astype(str) == str(row.donation_id)]
        if int(pd.to_numeric(actual["quantity_kg"], errors="coerce").sum()) != int(row.allocated_kg):
            errors.append(f"Pickup quantities for {row.donation_id} do not match the allocation total.")

    for row in plan.itertuples(index=False):
        matching_pickup = pickup_stops[
            (pickup_stops["donation_id"].astype(str) == str(row.donation_id))
            & (pickup_stops["recipient_id"].astype(str) == str(row.recipient_id))
            & (pd.to_numeric(pickup_stops["quantity_kg"], errors="coerce") == int(row.allocated_kg))
        ]
        matching_delivery = delivery_stops[
            (delivery_stops["donation_id"].astype(str) == str(row.donation_id))
            & (delivery_stops["recipient_id"].astype(str) == str(row.recipient_id))
            & (pd.to_numeric(delivery_stops["quantity_kg"], errors="coerce") == int(row.allocated_kg))
        ]
        if len(matching_pickup) != 1:
            errors.append(f"Shipment {row.donation_id} → {row.recipient_id} must have one matching pickup; found {len(matching_pickup)}.")
        if len(matching_delivery) != 1:
            errors.append(f"Shipment {row.donation_id} → {row.recipient_id} must have one matching delivery event; found {len(matching_delivery)}.")
            continue
        delivery = matching_delivery.iloc[0]
        if len(matching_pickup) == 1:
            pickup = matching_pickup.iloc[0]
            if str(pickup["vehicle_id"]) != str(delivery["vehicle_id"]):
                errors.append(f"Shipment {row.donation_id} → {row.recipient_id} uses different vehicles for pickup and delivery.")
            if int(pickup["stop_order"]) >= int(delivery["stop_order"]):
                errors.append(f"Shipment {row.donation_id} → {row.recipient_id} is delivered before pickup.")
        latest = delivery["deadline_hours"]
        expiry = delivery["expiry_hours"]
        max_allowed = min(float(latest), float(expiry))
        if float(delivery["arrival_hours"]) > max_allowed + 1e-6:
            errors.append(f"Shipment {row.donation_id} → {row.recipient_id} arrives after a configured time window.")

    for vehicle_id, group in stops.groupby("vehicle_id"):
        ordered = group.sort_values("stop_order")
        if not (ordered["stop_type"] == "Depot start").any() or not (ordered["stop_type"] == "Depot return").any():
            errors.append(f"Vehicle {vehicle_id} route must start and finish at the synthetic hub.")
        cap = capacity_by_vehicle.get(str(vehicle_id))
        for stop in ordered.itertuples(index=False):
            if stop.load_after_kg < -1e-6:
                errors.append(f"Vehicle {vehicle_id} has negative load after stop {stop.stop_order}.")
            if cap is not None and stop.load_after_kg > cap:
                errors.append(f"Vehicle {vehicle_id} exceeds its {cap} kg capacity.")
        final_load = int(ordered.iloc[-1]["load_after_kg"])
        if final_load != 0:
            errors.append(f"Vehicle {vehicle_id} returns to the hub carrying {final_load} kg; expected zero.")

    expected_total = int(plan["allocated_kg"].sum())
    actual_total = int(delivery_stops["quantity_kg"].sum()) if not delivery_stops.empty else 0
    if actual_total != expected_total:
        errors.append(f"Routed deliveries total {actual_total} kg but the allocation plan contains {expected_total} kg.")

    if route_result.get("metadata", {}).get("fallback_used"):
        warnings.append("A deterministic greedy fallback was used because the OR-Tools solver was unavailable or did not return a feasible plan.")
    return RouteValidationResult(not errors, errors, warnings)
