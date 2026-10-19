from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from rescue_route.agents import run_rescue_workflow
from rescue_route.data import demo_scenario, sample_donations, sample_recipients
from rescue_route.optimizer import validate_inputs

st.set_page_config(page_title="RescueRoute", page_icon="♻️", layout="wide")

st.title("RescueRoute")
st.subheader("Multi-agent food rescue and allocation optimization")
st.write(
    "A demonstrator for allocating surplus food to eligible community recipients under supply, "
    "demand, storage, urgency and delivery-window constraints. It uses synthetic data by default."
)
st.info(
    "This is a planning simulation, not an operational food-safety or dispatch system. "
    "Food eligibility, safe handling and recipient approval must be verified by qualified people."
)

with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox(
        "Choose a demonstration case",
        ["Baseline", "Time-critical food", "Donation disruption", "Supply shortage", "Invalid input"],
    )
    inject_fault = st.checkbox("Inject a fault to test Critic + retry", value=False)
    st.caption("The fault injection deliberately corrupts one proposed allocation so the Critic can reject it.")
    st.divider()
    st.header("Upload your data (optional)")
    donation_upload = st.file_uploader("Donations CSV", type=["csv"], key="donations")
    recipient_upload = st.file_uploader("Recipients CSV", type=["csv"], key="recipients")

scenario_key = {
    "Baseline": "Baseline",
    "Time-critical food": "Time-critical food",
    "Donation disruption": "Donation disruption",
    "Supply shortage": "Supply shortage",
    "Invalid input": "Invalid input",
}[scenario]
donations, recipients, scenario_note = demo_scenario(scenario_key)

if donation_upload is not None:
    try:
        donations = pd.read_csv(donation_upload)
        scenario_note = "Custom donations CSV uploaded."
    except Exception as exc:
        st.error(f"Could not read donations CSV: {exc}")
if recipient_upload is not None:
    try:
        recipients = pd.read_csv(recipient_upload)
        scenario_note += " Custom recipients CSV uploaded."
    except Exception as exc:
        st.error(f"Could not read recipients CSV: {exc}")

st.caption(scenario_note)
left, right = st.columns(2)
with left:
    st.markdown("### Donation inventory")
    st.dataframe(donations, use_container_width=True, hide_index=True)
with right:
    st.markdown("### Recipient requirements")
    st.dataframe(recipients, use_container_width=True, hide_index=True)

if scenario != "Invalid input" or donation_upload is not None:
    precheck = validate_inputs(donations, recipients)
    if precheck.valid:
        st.success("Input schema check passed.")
    else:
        st.warning("Input validation will reject this scenario.")
        for error in precheck.errors:
            st.write(f"- {error}")

run = st.button("Run RescueRoute workflow", type="primary", use_container_width=True)
if run:
    result = run_rescue_workflow(donations.copy(), recipients.copy(), inject_critic_failure=inject_fault)
    st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.markdown("## Workflow results")
    if result["status"] == "rejected":
        st.error("Workflow stopped at data intake because the input did not pass validation.")
        for error in result["input_validation"].errors:
            st.write(f"- {error}")
    elif result["status"] == "solver_failed":
        st.error("The optimization solver failed. See the execution log below.")
    else:
        summary = result["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Allocated food", f"{summary['total_allocated_kg']:,} kg")
        m2.metric("Effective demand/capacity", f"{summary['demand_capacity_kg']:,} kg")
        m3.metric("Demand fulfilled", f"{summary['demand_fulfilled_pct']}%")
        m4.metric("Recipients served", str(summary["recipients_served"]))

        if result["status"] == "completed_after_retry":
            st.success("Critic rejected the injected fault; Correction Agent regenerated the plan and the retry passed.")
        elif result["status"] == "completed":
            st.success("Critic approved the plan.")
        elif result["status"] == "manual_review_required":
            st.error("The corrected plan did not pass validation. Manual review is required.")

        st.markdown("### Allocation plan")
        plan = result["plan"]
        if plan.empty:
            st.warning("No eligible allocations were found for this scenario.")
        else:
            display_cols = ["donation_id", "food_type", "recipient_name", "allocated_kg", "distance_km", "estimated_delivery_hours", "urgency"]
            st.dataframe(plan[display_cols], use_container_width=True, hide_index=True)
            st.bar_chart(plan.groupby("recipient_name", as_index=True)["allocated_kg"].sum())
            st.download_button(
                "Download allocation plan CSV",
                plan.to_csv(index=False).encode("utf-8"),
                file_name="rescueroute_allocation_plan.csv",
                mime="text/csv",
            )

        with st.expander("Solver metadata and excluded links"):
            st.json({k: v for k, v in result.get("solver_metadata", {}).items() if k != "exclusions"})
            exclusions = result.get("solver_metadata", {}).get("exclusions", [])
            st.write("Eligibility exclusions (sampled up to 50):")
            st.write(exclusions[:50] if exclusions else "None")

        critic = result.get("critic_validation")
        if critic:
            st.markdown("### Critic validation")
            if critic.valid:
                st.success("Final plan passed validation.")
            else:
                st.error("Final plan failed validation.")
            for warning in critic.warnings:
                st.warning(warning)
            for error in critic.errors:
                st.write(f"- {error}")

    st.markdown("### Agent execution log")
    log_rows = [{
        "Stage": entry.stage,
        "Action": entry.action,
        "Result": entry.result,
        "Status": entry.status,
    } for entry in result["logs"]]
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    log_csv = pd.DataFrame(log_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Download execution log CSV", log_csv, "rescueroute_execution_log.csv", "text/csv")

with st.expander("CSV schema and sample downloads"):
    st.markdown("**Donations required columns:** `donation_id`, `donor_name`, `food_type`, `quantity_kg`, `expiry_hours`, `x_km`, `y_km`, `available`.")
    st.markdown("**Recipients required columns:** `recipient_id`, `recipient_name`, `demand_kg`, `storage_capacity_kg`, `urgency`, `delivery_deadline_hours`, `accepts_food_types`, `x_km`, `y_km`.")
    st.markdown("`accepts_food_types` must be a pipe-separated list, e.g. `Produce|Grains|Prepared meals`. Coordinates are synthetic local-grid kilometres for demonstration.")
    st.download_button("Download sample donations CSV", sample_donations().to_csv(index=False).encode("utf-8"), "donations_sample.csv", "text/csv")
    st.download_button("Download sample recipients CSV", sample_recipients().to_csv(index=False).encode("utf-8"), "recipients_sample.csv", "text/csv")

st.divider()
st.caption("Milestone 1: deterministic multi-agent workflow + mixed-integer allocation optimization. Route optimization and disruption replanning are planned for the next milestone.")
