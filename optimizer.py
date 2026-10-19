"""Deterministic allocation optimization and plan validation for NourishFlow.

Uses scipy.optimize.milp so allocations are integer kilograms and all constraints
are explicit. Routing optimization is planned as the next milestone.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

DONATION_COLUMNS = {
    "donation_id", "donor_name", "food_type", "quantity_kg", "expiry_hours",
    "x_km", "y_km", "available",
}
RECIPIENT_COLUMNS = {
    "recipient_id", "recipient_name", "demand_kg", "storage_capacity_kg",
    "urgency", "delivery_deadline_hours", "accepts_food_types", "x_km", "y_km",
}
PLAN_COLUMNS = [
    "donation_id", "donor_name", "food_type", "recipient_id", "recipient_name",
    "allocated_kg", "distance_km", "estimated_delivery_hours", "urgency",
    "delivery_deadline_hours", "expiry_hours", "objective_weight",
]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def _food_key(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def validate_inputs(donations: pd.DataFrame, recipients: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    missing_d = sorted(DONATION_COLUMNS - set(donations.columns))
    missing_r = sorted(RECIPIENT_COLUMNS - set(recipients.columns))
    if missing_d:
        errors.append("Donations are missing required columns: " + ", ".join(missing_d))
    if missing_r:
        errors.append("Recipients are missing required columns: " + ", ".join(missing_r))
    if errors:
        return ValidationResult(False, errors, warnings)

    if donations.empty:
        warnings.append("No donations were supplied.")
    if recipients.empty:
        errors.append("At least one recipient row is required.")

    for frame, identifier, label in [
        (donations, "donation_id", "donation"),
        (recipients, "recipient_id", "recipient"),
    ]:
        if frame[identifier].isna().any() or frame[identifier].astype(str).str.strip().eq("").any():
            errors.append(f"Every {label} must have a non-empty {identifier}.")
        if frame[identifier].duplicated().any():
            duplicates = frame.loc[frame[identifier].duplicated(), identifier].astype(str).tolist()
            errors.append(f"Duplicate {label} IDs: {', '.join(duplicates)}.")

    numeric_rules = [
        (donations, ["quantity_kg", "expiry_hours", "x_km", "y_km"], "donation"),
        (recipients, ["demand_kg", "storage_capacity_kg", "urgency", "delivery_deadline_hours", "x_km", "y_km"], "recipient"),
    ]
    for frame, columns, label in numeric_rules:
        for column in columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.isna().any():
                errors.append(f"{label.title()} field '{column}' contains missing or non-numeric values.")
            elif not np.isfinite(converted.to_numpy(dtype=float)).all():
                errors.append(f"{label.title()} field '{column}' contains non-finite values.")

    if not errors:
        if (pd.to_numeric(donations["quantity_kg"]) < 0).any():
            errors.append("Donation quantity_kg cannot be negative.")
        if (pd.to_numeric(donations["expiry_hours"]) <= 0).any():
            errors.append("Donation expiry_hours must be greater than zero.")
        if (pd.to_numeric(recipients["demand_kg"]) < 0).any():
            errors.append("Recipient demand_kg cannot be negative.")
        if (pd.to_numeric(recipients["storage_capacity_kg"]) < 0).any():
            errors.append("Recipient storage_capacity_kg cannot be negative.")
        urgency = pd.to_numeric(recipients["urgency"])
        if ((urgency < 1) | (urgency > 5) | (urgency % 1 != 0)).any():
            errors.append("Recipient urgency must be an integer from 1 to 5.")
        if (pd.to_numeric(recipients["delivery_deadline_hours"]) <= 0).any():
            errors.append("Recipient delivery_deadline_hours must be greater than zero.")
        if donations["food_type"].isna().any() or donations["food_type"].astype(str).str.strip().eq("").any():
            errors.append("Every donation must specify a food_type.")
        if recipients["accepts_food_types"].isna().any() or recipients["accepts_food_types"].astype(str).str.strip().eq("").any():
            errors.append("Every recipient must specify accepts_food_types (use pipe-separated values).")
        if "available" in donations.columns:
            values = donations["available"].astype(str).str.lower().str.strip()
            if not values.isin({"true", "false", "1", "0", "yes", "no"}).all():
                errors.append("Donation available must contain true/false-style values.")

    return ValidationResult(not errors, errors, warnings)


def _distance_km(donation: pd.Series, recipient: pd.Series) -> float:
    # Synthetic coordinates are expressed in a local Cartesian grid measured in km.
    return hypot(float(donation["x_km"]) - float(recipient["x_km"]),
                 float(donation["y_km"]) - float(recipient["y_km"]))


def build_eligibility(donations: pd.DataFrame, recipients: pd.DataFrame,
                      average_speed_kmh: float = 25.0,
                      handling_buffer_hours: float = 0.25) -> tuple[pd.DataFrame, list[str]]:
    """Construct compatible donation-recipient links and explain excluded links."""
    links: list[dict[str, Any]] = []
    exclusions: list[str] = []
    available_col = donations["available"].astype(str).str.lower().str.strip().isin({"true", "1", "yes"})

    for d_index, donation in donations.iterrows():
        if not bool(available_col.loc[d_index]):
            exclusions.append(f"{donation['donation_id']} excluded: donation marked unavailable.")
            continue
        for _, recipient in recipients.iterrows():
            compatible = {_food_key(x) for x in str(recipient["accepts_food_types"]).split("|")}
            food = _food_key(donation["food_type"])
            if food not in compatible:
                exclusions.append(f"{donation['donation_id']} → {recipient['recipient_id']} excluded: food type not accepted.")
                continue
            distance = _distance_km(donation, recipient)
            travel_hours = handling_buffer_hours + distance / average_speed_kmh
            expiry = float(donation["expiry_hours"])
            deadline = float(recipient["delivery_deadline_hours"])
            if travel_hours > expiry:
                exclusions.append(f"{donation['donation_id']} → {recipient['recipient_id']} excluded: estimated delivery exceeds remaining food window.")
                continue
            if travel_hours > deadline:
                exclusions.append(f"{donation['donation_id']} → {recipient['recipient_id']} excluded: estimated delivery misses recipient deadline.")
                continue
            links.append({
                "donation_index": int(d_index),
                "recipient_index": int(recipient.name),
                "donation_id": str(donation["donation_id"]),
                "recipient_id": str(recipient["recipient_id"]),
                "distance_km": distance,
                "estimated_delivery_hours": travel_hours,
                "urgency": int(recipient["urgency"]),
                "expiry_hours": expiry,
                "delivery_deadline_hours": deadline,
            })
    return pd.DataFrame(links), exclusions


def optimize_allocations(donations: pd.DataFrame, recipients: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Maximize urgency-weighted, feasible kilograms delivered, with a distance penalty."""
    links, exclusions = build_eligibility(donations, recipients)
    if links.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS), {
            "solver": "SciPy MILP", "status": "No eligible donation-recipient links",
            "eligible_links": 0, "excluded_links": len(exclusions), "objective_value": 0.0,
            "exclusions": exclusions,
        }

    n = len(links)
    # Each feasible link gets an integer number of kg. Positive scores fill eligible
    # demand; urgency dominates distance, while the distance penalty breaks ties.
    objective_weights = (
        100.0 + links["urgency"].to_numpy(dtype=float) * 25.0
        - links["distance_km"].to_numpy(dtype=float) * 2.0
        + 8.0 / np.maximum(links["expiry_hours"].to_numpy(dtype=float), 0.5)
    )
    c = -objective_weights  # scipy.optimize.milp minimizes

    n_donations, n_recipients = len(donations), len(recipients)
    constraints_count = n_donations + n_recipients
    A = lil_matrix((constraints_count, n), dtype=float)
    upper = np.zeros(constraints_count, dtype=float)

    donation_id_to_index = {str(row["donation_id"]): idx for idx, (_, row) in enumerate(donations.iterrows())}
    recipient_id_to_index = {str(row["recipient_id"]): idx for idx, (_, row) in enumerate(recipients.iterrows())}

    for variable_idx, link in links.iterrows():
        A[donation_id_to_index[link["donation_id"]], variable_idx] = 1.0
        A[n_donations + recipient_id_to_index[link["recipient_id"]], variable_idx] = 1.0
    for idx, row in enumerate(donations.itertuples(index=False)):
        upper[idx] = float(row.quantity_kg)
    for idx, row in enumerate(recipients.itertuples(index=False)):
        upper[n_donations + idx] = min(float(row.demand_kg), float(row.storage_capacity_kg))

    result = milp(
        c=c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.array([
            min(float(donations.loc[donation_id_to_index[link["donation_id"]], "quantity_kg"]),
                float(recipients.loc[recipient_id_to_index[link["recipient_id"]], "demand_kg"]),
                float(recipients.loc[recipient_id_to_index[link["recipient_id"]], "storage_capacity_kg"]))
            for _, link in links.iterrows()
        ])),
        constraints=LinearConstraint(A.tocsr(), lb=np.full(constraints_count, -np.inf), ub=upper),
        options={"time_limit": 10.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Allocation optimizer failed: {result.message}")

    allocations = np.rint(result.x).astype(int)
    plan_records: list[dict[str, Any]] = []
    for var_idx, kg in enumerate(allocations):
        if kg <= 0:
            continue
        link = links.iloc[var_idx]
        drow = donations.loc[donation_id_to_index[link["donation_id"]]]
        rrow = recipients.loc[recipient_id_to_index[link["recipient_id"]]]
        plan_records.append({
            "donation_id": str(drow["donation_id"]),
            "donor_name": str(drow["donor_name"]),
            "food_type": str(drow["food_type"]),
            "recipient_id": str(rrow["recipient_id"]),
            "recipient_name": str(rrow["recipient_name"]),
            "allocated_kg": int(kg),
            "distance_km": round(float(link["distance_km"]), 2),
            "estimated_delivery_hours": round(float(link["estimated_delivery_hours"]), 2),
            "urgency": int(rrow["urgency"]),
            "delivery_deadline_hours": float(rrow["delivery_deadline_hours"]),
            "expiry_hours": float(drow["expiry_hours"]),
            "objective_weight": round(float(objective_weights[var_idx]), 2),
        })

    plan = pd.DataFrame(plan_records, columns=PLAN_COLUMNS)
    metadata = {
        "solver": "SciPy MILP (mixed-integer linear optimization)",
        "status": str(result.message),
        "eligible_links": int(len(links)),
        "excluded_links": int(len(exclusions)),
        "objective_value": round(float(-result.fun), 2),
        "exclusions": exclusions,
    }
    return plan, metadata


def validate_plan(plan: pd.DataFrame, donations: pd.DataFrame, recipients: pd.DataFrame) -> ValidationResult:
    """Critic checks allocation invariants against original input data."""
    errors: list[str] = []
    warnings: list[str] = []
    if plan.empty:
        warnings.append("No feasible allocations were produced; check eligibility, timing and available supply.")
        return ValidationResult(True, errors, warnings)

    if (pd.to_numeric(plan["allocated_kg"], errors="coerce").isna()).any():
        errors.append("Plan contains non-numeric allocated_kg values.")
        return ValidationResult(False, errors, warnings)
    if (plan["allocated_kg"] <= 0).any():
        errors.append("Every plan row must allocate a positive quantity.")
    if ((plan["allocated_kg"] % 1) != 0).any():
        errors.append("Allocations must be whole kilograms in this prototype.")

    donation_by_id = donations.set_index("donation_id", drop=False)
    recipient_by_id = recipients.set_index("recipient_id", drop=False)
    for donation_id, group in plan.groupby("donation_id"):
        if donation_id not in donation_by_id.index:
            errors.append(f"Plan references unknown donation {donation_id}.")
        elif group["allocated_kg"].sum() > float(donation_by_id.loc[donation_id, "quantity_kg"]):
            errors.append(f"Allocation exceeds the available quantity for donation {donation_id}.")
    for recipient_id, group in plan.groupby("recipient_id"):
        if recipient_id not in recipient_by_id.index:
            errors.append(f"Plan references unknown recipient {recipient_id}.")
        else:
            recipient = recipient_by_id.loc[recipient_id]
            cap = min(float(recipient["demand_kg"]), float(recipient["storage_capacity_kg"]))
            if group["allocated_kg"].sum() > cap:
                errors.append(f"Allocation exceeds demand or storage capacity for recipient {recipient_id}.")

    for _, row in plan.iterrows():
        if row["donation_id"] not in donation_by_id.index or row["recipient_id"] not in recipient_by_id.index:
            continue
        donation = donation_by_id.loc[row["donation_id"]]
        recipient = recipient_by_id.loc[row["recipient_id"]]
        accepted = {_food_key(x) for x in str(recipient["accepts_food_types"]).split("|")}
        if _food_key(donation["food_type"]) not in accepted:
            errors.append(f"Recipient {row['recipient_id']} does not accept {donation['food_type']}.")
        if float(row["estimated_delivery_hours"]) > float(donation["expiry_hours"]):
            errors.append(f"Delivery from {row['donation_id']} exceeds its remaining food window.")
        if float(row["estimated_delivery_hours"]) > float(recipient["delivery_deadline_hours"]):
            errors.append(f"Delivery to {row['recipient_id']} misses its deadline.")

    # Duplicate pair rows are not needed and can hide aggregation errors.
    if plan.duplicated(["donation_id", "recipient_id"]).any():
        warnings.append("Plan contains repeated donation-recipient pairs; these should normally be consolidated.")
    return ValidationResult(not errors, errors, warnings)


def corrupt_plan_for_demo(plan: pd.DataFrame) -> pd.DataFrame:
    """Deliberately inject a bad row to exercise Critic rejection and repair."""
    corrupted = plan.copy()
    if corrupted.empty:
        # An explicit invalid row ensures the Critic test can demonstrate rejection.
        corrupted = pd.DataFrame([{
            "donation_id": "UNKNOWN", "donor_name": "Injected test row", "food_type": "Unknown",
            "recipient_id": "UNKNOWN", "recipient_name": "Injected test row", "allocated_kg": -1,
            "distance_km": 0.0, "estimated_delivery_hours": 0.0, "urgency": 1,
            "delivery_deadline_hours": 1.0, "expiry_hours": 1.0, "objective_weight": 0.0,
        }], columns=PLAN_COLUMNS)
    else:
        corrupted.loc[corrupted.index[0], "allocated_kg"] = -1
    return corrupted
