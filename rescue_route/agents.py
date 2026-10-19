"""Lightweight, deterministic agent workflow for RescueRoute."""
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


def run_rescue_workflow(
    donations: pd.DataFrame,
    recipients: pd.DataFrame,
    inject_critic_failure: bool = False,
) -> dict[str, Any]:
    """Run intake -> planner -> optimizer -> critic -> optional correction -> report.

    The agents in this first milestone are deterministic Python components, not LLMs.
    They demonstrate tool use, conditional routing, state passing and validation.
    """
    logs: list[AgentLog] = []
    started_at = datetime.now().isoformat(timespec="seconds")

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
            "logs": logs,
            "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
            "summary": {"total_donated_kg": 0, "total_allocated_kg": 0, "demand_fulfilled_pct": 0.0},
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
    except Exception as exc:  # a failed solver should be visible, not disguised
        _log(logs, "Allocation Optimizer", "Stop after solver failure", str(exc), "FAIL")
        return {
            "success": False,
            "status": "solver_failed",
            "input_validation": input_validation,
            "plan": pd.DataFrame(),
            "critic_validation": None,
            "logs": logs,
            "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
            "summary": {"total_donated_kg": total_supply, "total_allocated_kg": 0, "demand_fulfilled_pct": 0.0},
        }
    _log(logs, "Allocation Optimizer", "Return candidate plan",
         f"Solver completed: {len(plan)} allocation rows; {int(plan['allocated_kg'].sum()) if not plan.empty else 0} kg allocated.",
         "PASS", solver=solver_metadata["solver"], objective_value=solver_metadata["objective_value"])

    candidate_plan = corrupt_plan_for_demo(plan) if inject_critic_failure else plan.copy()
    if inject_critic_failure:
        _log(logs, "Demo Fault Injector", "Inject an invalid allocation to test validation.",
             "One allocation quantity was deliberately changed to -1 kg; this is a test fault, not optimizer output.", "WARN")

    _log(logs, "Critic Agent", "Verify candidate plan against original supply, demand, accepted food types and time windows.",
         "Run deterministic constraint validation.")
    critic_result = validate_plan(candidate_plan, donations, recipients)
    if critic_result.valid:
        _log(logs, "Critic Agent", "Approve candidate plan", "All checked constraints passed.", "PASS")
        final_plan = candidate_plan
        final_validation = critic_result
        status = "completed"
    else:
        for error in critic_result.errors:
            _log(logs, "Critic Agent", "Reject candidate plan", error, "FAIL")
        _log(logs, "Correction Agent", "Recompute the allocation plan from trusted source data.",
             "One automatic correction attempt will be made.", "WARN")
        final_plan, solver_metadata = optimize_allocations(donations, recipients)
        retry_result = validate_plan(final_plan, donations, recipients)
        if retry_result.valid:
            _log(logs, "Critic Agent", "Approve corrected plan", "Retry passed all deterministic checks.", "PASS")
            final_validation = retry_result
            status = "completed_after_retry"
        else:
            for error in retry_result.errors:
                _log(logs, "Critic Agent", "Reject corrected plan", error, "FAIL")
            _log(logs, "Orchestrator", "Escalate after retry limit", "Plan did not pass validation after one correction attempt.", "FAIL")
            final_validation = retry_result
            status = "manual_review_required"

    allocated = int(final_plan["allocated_kg"].sum()) if not final_plan.empty else 0
    eligible_supply = max(total_supply, 0)
    total_effective_demand = max(effective_demand, 0)
    # Demand fulfillment is demand-weighted and capped by effective recipient capacity.
    recipient_delivered = final_plan.groupby("recipient_id")["allocated_kg"].sum().to_dict() if not final_plan.empty else {}
    demand_fulfilled = 0.0
    for recipient in recipients.itertuples(index=False):
        cap = min(float(recipient.demand_kg), float(recipient.storage_capacity_kg))
        if cap > 0:
            demand_fulfilled += min(float(recipient_delivered.get(recipient.recipient_id, 0)), cap)
    fulfillment_pct = (100.0 * demand_fulfilled / total_effective_demand) if total_effective_demand else 0.0

    _log(logs, "Report Agent", "Summarize results and unmet demand.",
         f"Final plan allocates {allocated} kg; effective recipient demand/capacity is {total_effective_demand} kg.",
         "PASS" if final_validation.valid else "WARN")

    return {
        "success": bool(final_validation.valid),
        "status": status,
        "input_validation": input_validation,
        "plan": final_plan,
        "critic_validation": final_validation,
        "logs": logs,
        "solver_metadata": solver_metadata,
        "metadata": {"started_at": started_at, "ended_at": datetime.now().isoformat(timespec="seconds")},
        "summary": {
            "total_donated_kg": eligible_supply,
            "total_allocated_kg": allocated,
            "demand_capacity_kg": total_effective_demand,
            "demand_fulfilled_pct": round(fulfillment_pct, 1),
            "allocation_rows": len(final_plan),
            "recipients_served": int(final_plan["recipient_id"].nunique()) if not final_plan.empty else 0,
            "total_route_distance_km_estimate": round(float((final_plan["distance_km"] * 2).sum()), 2) if not final_plan.empty else 0.0,
        },
    }
