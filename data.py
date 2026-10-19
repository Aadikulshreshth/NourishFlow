"""Synthetic, reproducible data for the NourishFlow proof of concept."""
from __future__ import annotations

import pandas as pd


def sample_donations() -> pd.DataFrame:
    return pd.DataFrame([
        {"donation_id": "D01", "donor_name": "GreenBasket Market", "food_type": "Produce", "quantity_kg": 120, "expiry_hours": 6.0, "x_km": 2.0, "y_km": 4.0, "available": True},
        {"donation_id": "D02", "donor_name": "DailyDairy Cooperative", "food_type": "Dairy", "quantity_kg": 70, "expiry_hours": 3.0, "x_km": 8.0, "y_km": 3.0, "available": True},
        {"donation_id": "D03", "donor_name": "GrainShare Warehouse", "food_type": "Grains", "quantity_kg": 160, "expiry_hours": 72.0, "x_km": 3.0, "y_km": 9.0, "available": True},
        {"donation_id": "D04", "donor_name": "City Canteen", "food_type": "Prepared meals", "quantity_kg": 55, "expiry_hours": 2.5, "x_km": 7.0, "y_km": 8.0, "available": True},
        {"donation_id": "D05", "donor_name": "FarmLink Cooperative", "food_type": "Produce", "quantity_kg": 90, "expiry_hours": 12.0, "x_km": 10.0, "y_km": 6.0, "available": True},
    ])


def sample_recipients() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipient_id": "R01", "recipient_name": "Hope Community Kitchen", "demand_kg": 110, "storage_capacity_kg": 120, "urgency": 5, "delivery_deadline_hours": 4.0, "accepts_food_types": "Produce|Grains|Prepared meals", "x_km": 3.0, "y_km": 3.0},
        {"recipient_id": "R02", "recipient_name": "Sunrise Shelter", "demand_kg": 80, "storage_capacity_kg": 90, "urgency": 4, "delivery_deadline_hours": 5.0, "accepts_food_types": "Produce|Dairy|Grains|Prepared meals", "x_km": 8.0, "y_km": 5.0},
        {"recipient_id": "R03", "recipient_name": "Community Support Hub", "demand_kg": 140, "storage_capacity_kg": 160, "urgency": 3, "delivery_deadline_hours": 12.0, "accepts_food_types": "Produce|Grains", "x_km": 4.0, "y_km": 10.0},
        {"recipient_id": "R04", "recipient_name": "Night Relief Centre", "demand_kg": 65, "storage_capacity_kg": 70, "urgency": 5, "delivery_deadline_hours": 2.0, "accepts_food_types": "Produce|Dairy|Prepared meals", "x_km": 7.0, "y_km": 9.0},
        {"recipient_id": "R05", "recipient_name": "Youth Support Centre", "demand_kg": 90, "storage_capacity_kg": 95, "urgency": 2, "delivery_deadline_hours": 10.0, "accepts_food_types": "Produce|Dairy|Grains", "x_km": 11.0, "y_km": 7.0},
    ])


def demo_scenario(name: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return copies of sample data configured for a named demonstration case."""
    donations = sample_donations()
    recipients = sample_recipients()
    note = "Baseline synthetic donations and recipient demand."

    if name == "Time-critical food":
        # A tighter deadline forces the workflow to reconsider eligibility.
        mask = donations["donation_id"] == "D04"
        donations.loc[mask, "expiry_hours"] = 0.35
        recipients.loc[recipients["recipient_id"] == "R04", "delivery_deadline_hours"] = 0.4
        note = "Prepared meals have a very short remaining window; the optimizer must exclude recipient links that cannot be reached in time."
    elif name == "Donation disruption":
        donations = donations.loc[donations["donation_id"] != "D03"].reset_index(drop=True)
        note = "D03 is unavailable. The optimizer must rebuild allocations using the remaining donations."
    elif name == "Supply shortage":
        donations["quantity_kg"] = (donations["quantity_kg"] * 0.4).round().astype(int)
        note = "Available quantities are reduced to 40% to demonstrate unmet demand and prioritization under scarcity."
    elif name == "Invalid input":
        donations = donations.drop(columns=["quantity_kg"])
        note = "The required quantity_kg column is deliberately removed. Intake validation should reject this input."

    return donations, recipients, note
