"""Lightweight, deterministic multi-agent workflow for NourishFlow."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .optimizer import (
    ValidationResult,
    corrupt_plan_for_demo,
    optimize_allocations,
    validate_inputs,
    validate_plan,
)
from .routing import (
    RouteValidationResult,
    default_fleet,
    optimize_vehicle_routes,
    validate_route_plan,
)


@dataclass
class AgentLog:
    stage: str
    action: str
    result: str
    status: str = "INFO"
    details: dict[str, Any] = field(default_factory=dict)


def _log(logs: list[AgentLog], stage: str, action: str, result: str,
         status: str = "INFO", **details: Any) -> None:
    logs.append(AgentLog(stage=stage, action=action, result=result, status=status, details=details))


def _empty_route_replan(requested: bool = False) -> dict[str, Any]:
    return {
        "requested": requested,
        "success": None,
        "broken_vehicle_id": None,
        "message": "Routing was not run.",
        "initial_summary": None,
        "final_summary": None,
    }


def _run_routing_stages(
    plan: pd.DataFrame,
    donations: pd.DataFrame,
    recipients: pd.DataFrame,
    logs: list[AgentLog],
    vehicle_count: int,
    vehicle_capacity_kg: int,
    simulate_vehicle_breakdown: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Plan, validate, benchmark and optionally replan after a simulated breakdown."""
    fleet = default_fleet(vehicle_count, vehicle_capacity_kg)
    _log(
        logs,
        "Vehicle Routing Agent",
        "Build pickup-and-delivery tasks and solve a capacity/time-window route model.",
        f"Fleet: {len(fleet)} vehicles; {vehicle_capacity_kg} kg capacity each; assumed speed 25 km/h.",
        "INFO",
        vehicle_specs=fleet,
    )
    initial_routing = optimize_vehicle_routes(plan, donations, recipients, vehicle_specs=fleet)
    initial_validation = validate_route_plan(initial_routing, plan)
    route_baseline = optimize_vehicle_routes(plan, donations, recipients, vehicle_specs=fleet, force_greedy=True)
    route_baseline["validation"] = validate_route_plan(route_baseline, plan)
    if route_baseline.get("success") and route_baseline["validation"].valid:
        _log(
            logs, "Baseline Comparator", "Compare routing output with a deadline-first greedy benchmark.",
            f"Greedy baseline distance: {route_baseline['summary']['total_distance_km']:.2f} km; "
            + (f"selected route distance: {initial_routing['summary']['total_distance_km']:.2f} km." if initial_routing.get("success") else "selected route plan is infeasible."),
            "PASS" if initial_routing.get("success") and initial_validation.valid else "WARN",
            baseline_distance_km=route_baseline["summary"]["total_distance_km"],
            selected_distance_km=initial_routing.get("summary", {}).get("total_distance_km"),
        )
    else:
        _log(logs, "Baseline Comparator", "Attempt greedy route benchmark", "The fallback benchmark could not produce a feasible route for this input.", "WARN")
    initial_routing["validation"] = initial_validation
    if not initial_validation.valid:
        initial_routing["success"] = False
        initial_routing["status"] = "route_validation_failed"
        _log(
            logs,
            "Route Critic Agent",
            "Reject route candidate",
            "; ".join(initial_validation.errors),
            "FAIL",
            errors=initial_validation.errors,
        )
    else:
        _log(
            logs,
            "Route Critic Agent",
            "Approve route candidate",
            f"All routed shipments are covered; {initial_routing['summary']['total_distance_km']:.2f} synthetic-grid km across {initial_routing['summary']['vehicles_used']} vehicle(s).",
            "PASS",
            solver=initial_routing.get("metadata", {}).get("solver"),
            route_distance_km=initial_routing["summary"]["total_distance_km"],
        )
        for warning in initial_validation.warnings:
            _log(logs, "Route Critic Agent", "Record routing caveat", warning, "WARN")
    routing = initial_routing
    replan = _empty_route_replan(simulate_vehicle_breakdown)

    if not simulate_vehicle_breakdown:
        return routing, initial_routing, replan, route_baseline

    if not initial_routing.get("success") or not initial_validation.valid:
        message = "Cannot simulate a breakdown because the initial route plan did not pass validation."
        replan["success"] = False
        replan["message"] = message
        _log(logs, "Disruption Agent", "Request vehicle-breakdown simulation", message, "FAIL")
        return routing, initial_routing, replan, route_baseline

    stops = initial_routing["stops"]
    delivery_stops = stops[stops["stop_type"] == "Delivery"]
    if delivery_stops.empty:
        message = "No delivery stops exist to simulate a vehicle breakdown against."
        replan["success"] = False
        replan["message"] = message
        _log(logs, "Disruption Agent", "Request vehicle-breakdown simulation", message, "WARN")
        return routing, initial_routing, replan, route_baseline

    delivered_by_vehicle = delivery_stops.groupby("vehicle_id")["quantity_kg"].sum().sort_values(ascending=False)
    broken_vehicle_id = str(delivered_by_vehicle.index[0])
    available_fleet = [spec for spec in fleet if str(spec["vehicle_id"]) != broken_vehicle_id]
    replan["broken_vehicle_id"] = broken_vehicle_id
    replan["initial_summary"] = dict(initial_routing["summary"])
    _log(
        logs,
        "Disruption Agent",
        "Mark the busiest planned vehicle unavailable.",
        f"Vehicle {broken_vehicle_id} is simulated as broken down after the initial plan; route planning must proceed without it.",
        "WARN",
        broken_vehicle_id=broken_vehicle_id,
        remaining_vehicle_ids=[v["vehicle_id"] for v in available_fleet],
    )
    _log(
        logs,
        "Replanning Agent",
        "Re-solve all pickup-and-delivery routes with the remaining fleet.",
        f"The allocation plan is retained, but all vehicle routes are recomputed using {len(available_fleet)} available vehicle(s).",
        "INFO",
    )
    replanned = optimize_vehicle_routes(plan, donations, recipients, vehicle_specs=available_fleet)
    replanned_validation = validate_route_plan(replanned, plan)
    replanned["validation"] = replanned_validation
    replanned["metadata"]["replanned_after_breakdown"] = True
    replanned["metadata"]["broken_vehicle_id"] = broken_vehicle_id
    replanned["metadata"]["unavailable_vehicle_ids"] = [broken_vehicle_id]
    replanned["metadata"]["initial_route_summary"] = dict(initial_routing["summary"])

    if replanned.get("success") and replanned_validation.valid:
        routing = replanned
        replan.update({
            "success": True,
            "message": f"Replanning succeeded without vehicle {broken_vehicle_id}.",
            "final_summary": dict(replanned["summary"]),
        })
        _log(
            logs,
            "Route Critic Agent",
            "Approve revised routes",
            f"Replanned successfully with {replanned['summary']['vehicles_used']} vehicle(s); route distance is {replanned['summary']['total_distance_km']:.2f} synthetic-grid km.",
            "PASS",
            broken_vehicle_id=broken_vehicle_id,
            route_distance_km=replanned["summary"]["total_distance_km"],
        )
    else:
        routing = replanned
        reasons = replanned_validation.errors or [str(replanned.get("metadata", {}).get("failure_reason", "No feasible revised route."))]
        message = "Replanning failed: " + "; ".join(reasons)
        replan.update({"success": False, "message": message, "final_summary": dict(replanned["summary"])})
        _log(logs, "Route Critic Agent", "Reject revised routes", message, "FAIL", errors=reasons)
        _log(logs, "Orchestrator", "Escalate route disruption", "No feasible revised vehicle plan was validated; manual coordination is required.", "FAIL")
    return routing, initial_routing, replan, route_baseline


def run_rescue_workflow(
    donations: pd.DataFrame,
    recipients: pd.DataFrame,
    inject_critic_failure: bool = False,
    simulate_vehicle_breakdown: bool = False,
    vehicle_count: int = 3,
    vehicle_capacity_kg: int = 250,
) -> dict[str, Any]:
    """Run intake -> allocation -> Critic -> routing -> optional route replanning.

    The agents are deterministic Python components (not LLMs). SciPy performs the
    allocation MILP; Google OR-Tools performs the routing VRP when installed. A
    labelled greedy routing fallback is used in restricted environments without it.
    """
    logs: list[AgentLog] = []
    started_at = datetime.now().isoformat(timespec="seconds")
    no_replan = _empty_route_replan(simulate_vehicle_breakdown)

    _log(logs, "Data Intake Agent", "Check required fields, value ranges, identifiers and data types.",
         "Input validation started.")
    input_validation = validate_inputs(donations, recipients)
    if not input_validation.valid:
        for error in input_validation.errors:
            _log(logs, "Data Intake Agent", "Reject invalid input", error, "FAIL")
        return {
            "success": False,
            "status": "rejected",
            "input_validation": input_validation,
            "plan": pd.DataFrame(),
            "critic_validation": None,
            "routing": None,
            "initial_routing": None,
            "route_baseline": None,
            "routing_replan": no_replan,
            "logs": logs,
            "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
            "summary": {"total_donated_kg": 0, "total_allocated_kg": 0, "demand_fulfilled_pct": 0.0,
                        "total_route_distance_km_estimate": 0.0},
        }
    for warning in input_validation.warnings:
        _log(logs, "Data Intake Agent", "Record input warning", warning, "WARN")
    _log(logs, "Data Intake Agent", "Accept input", "Required schema and basic data checks passed.", "PASS")

    total_supply = int(pd.to_numeric(donations.loc[
        donations["available"].astype(str).str.lower().str.strip().isin({"true", "1", "yes"}), "quantity_kg"
    ]).sum())
    effective_demand = int(sum(min(float(row.demand_kg), float(row.storage_capacity_kg))
                               for row in recipients.itertuples(index=False)))
    _log(logs, "Planner Agent", "Profile supply, demand, urgency, food compatibility and delivery windows.",
         f"{len(donations)} donation rows; {len(recipients)} recipients; {total_supply} kg available; {effective_demand} kg demand/storage capacity.",
         supply_kg=total_supply, demand_capacity_kg=effective_demand)

    _log(logs, "Allocation Optimizer", "Solve integer allocation model.",
         "Maximize urgency-weighted eligible kilograms with a distance penalty, subject to supply, demand and storage limits.")
    try:
        plan, solver_metadata = optimize_allocations(donations, recipients)
    except Exception as exc:  # A failed solver should be visible, not disguised.
        _log(logs, "Allocation Optimizer", "Stop after solver failure", str(exc), "FAIL")
        return {
            "success": False,
            "status": "solver_failed",
            "input_validation": input_validation,
            "plan": pd.DataFrame(),
            "critic_validation": None,
            "routing": None,
            "initial_routing": None,
            "route_baseline": None,
            "routing_replan": no_replan,
            "logs": logs,
            "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
            "summary": {"total_donated_kg": total_supply, "total_allocated_kg": 0,
                        "demand_capacity_kg": effective_demand, "demand_fulfilled_pct": 0.0,
                        "total_route_distance_km_estimate": 0.0},
        }
    _log(logs, "Allocation Optimizer", "Return candidate plan",
         f"Solver completed: {len(plan)} allocation rows; {int(plan['allocated_kg'].sum()) if not plan.empty else 0} kg allocated.",
         "PASS", solver=solver_metadata["solver"], objective_value=solver_metadata["objective_value"])

    candidate_plan = corrupt_plan_for_demo(plan) if inject_critic_failure else plan.copy()
    if inject_critic_failure:
        _log(logs, "Demo Fault Injector", "Inject an invalid allocation to test validation.",
             "One allocation quantity was deliberately changed to -1 kg; this is a test fault, not optimizer output.", "WARN")

    _log(logs, "Critic Agent", "Verify candidate allocation against supply, demand, accepted food types and time windows.",
         "Run deterministic allocation constraint validation.")
    critic_result = validate_plan(candidate_plan, donations, recipients)
    if critic_result.valid:
        _log(logs, "Critic Agent", "Approve candidate allocation", "All checked allocation constraints passed.", "PASS")
        final_plan = candidate_plan
        final_validation = critic_result
        status = "completed"
    else:
        for error in critic_result.errors:
            _log(logs, "Critic Agent", "Reject candidate allocation", error, "FAIL")
        _log(logs, "Correction Agent", "Recompute the allocation from trusted source data.",
             "One automatic correction attempt will be made.", "WARN")
        final_plan, solver_metadata = optimize_allocations(donations, recipients)
        retry_result = validate_plan(final_plan, donations, recipients)
        if retry_result.valid:
            _log(logs, "Critic Agent", "Approve corrected allocation", "Retry passed all deterministic checks.", "PASS")
            final_validation = retry_result
            status = "completed_after_retry"
        else:
            for error in retry_result.errors:
                _log(logs, "Critic Agent", "Reject corrected allocation", error, "FAIL")
            _log(logs, "Orchestrator", "Escalate after retry limit", "Allocation did not pass validation after one correction attempt.", "FAIL")
            final_validation = retry_result
            status = "manual_review_required"

    allocated = int(final_plan["allocated_kg"].sum()) if not final_plan.empty else 0
    recipient_delivered = final_plan.groupby("recipient_id")["allocated_kg"].sum().to_dict() if not final_plan.empty else {}
    demand_fulfilled = 0.0
    for recipient in recipients.itertuples(index=False):
        cap = min(float(recipient.demand_kg), float(recipient.storage_capacity_kg))
        if cap > 0:
            demand_fulfilled += min(float(recipient_delivered.get(recipient.recipient_id, 0)), cap)
    fulfillment_pct = (100.0 * demand_fulfilled / effective_demand) if effective_demand else 0.0

    routing = None
    initial_routing = None
    routing_replan = no_replan
    route_baseline = None
    if final_validation.valid and status != "manual_review_required":
        routing, initial_routing, routing_replan, route_baseline = _run_routing_stages(
            final_plan, donations, recipients, logs,
            vehicle_count=vehicle_count,
            vehicle_capacity_kg=vehicle_capacity_kg,
            simulate_vehicle_breakdown=simulate_vehicle_breakdown,
        )

    route_distance = 0.0
    if routing and routing.get("success"):
        route_distance = float(routing.get("summary", {}).get("total_distance_km", 0.0))
    _log(
        logs,
        "Report Agent",
        "Summarize allocation, route feasibility and unmet demand.",
        f"Allocation: {allocated} kg of {effective_demand} kg effective demand ({fulfillment_pct:.1f}%); "
        + (f"routing feasible at {route_distance:.2f} synthetic-grid km." if routing and routing.get("success")
           else "routing requires review or did not return a feasible route."),
        "PASS" if final_validation.valid and routing and routing.get("success") else "WARN",
    )

    return {
        "success": bool(final_validation.valid),
        "status": status,
        "input_validation": input_validation,
        "plan": final_plan,
        "critic_validation": final_validation,
        "routing": routing,
        "initial_routing": initial_routing,
        "route_baseline": route_baseline,
        "routing_replan": routing_replan,
        "logs": logs,
        "solver_metadata": solver_metadata,
        "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
        "summary": {
            "total_donated_kg": max(total_supply, 0),
            "total_allocated_kg": allocated,
            "demand_capacity_kg": max(effective_demand, 0),
            "demand_fulfilled_pct": round(fulfillment_pct, 1),
            "allocation_rows": len(final_plan),
            "recipients_served": int(final_plan["recipient_id"].nunique()) if not final_plan.empty else 0,
            "total_route_distance_km_estimate": round(route_distance, 2),
        },
    }
