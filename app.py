from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rescue_route.agents import run_rescue_workflow
from rescue_route.data import demo_scenario, sample_donations, sample_recipients
from rescue_route.optimizer import validate_inputs

st.set_page_config(
    page_title="RescueRoute | Food Rescue Optimizer",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --rr-ink: #17352d;
        --rr-muted: #61766d;
        --rr-green: #176b50;
        --rr-green-dark: #103d31;
        --rr-mint: #dff3e7;
        --rr-lime: #d5ed8a;
        --rr-orange: #f5a65b;
        --rr-paper: #f5f8f3;
        --rr-line: #dce7de;
    }
    .stApp {
        background:
            radial-gradient(circle at 85% 2%, rgba(213,237,138,.20), transparent 22rem),
            linear-gradient(180deg, #f8faf6 0%, #f1f6f0 100%);
        color: var(--rr-ink);
    }
    [data-testid="stHeader"] { background: rgba(248,250,246,.82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123d31 0%, #0d2f27 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #f0f7f1; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.08);
        border: 1px dashed rgba(230,245,234,.42);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #f0f7f1; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        font-weight: 600;
    }
    .block-container { max-width: 1500px; padding-top: 1.45rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--rr-ink); letter-spacing: -0.025em; }
    h2 { margin-top: .35rem; }
    p, li { color: #38564b; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--rr-line);
        border-radius: 16px;
        padding: 15px 17px;
        box-shadow: 0 5px 18px rgba(24, 58, 43, .04);
    }
    [data-testid="stMetricLabel"] { color: var(--rr-muted); font-weight: 650; }
    [data-testid="stMetricValue"] { color: var(--rr-ink); font-weight: 780; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 2.7rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #187653 0%, #0f4b3b 100%);
        border: 0;
        color: #fff;
        box-shadow: 0 7px 16px rgba(23,107,80,.18);
    }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(23,75,55,.12);
    }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        font-weight: 650;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--rr-line); border-radius: 12px; overflow: hidden; }
    [data-testid="stExpander"] {
        border: 1px solid var(--rr-line);
        border-radius: 12px;
        background: rgba(255,255,255,.74);
    }
    .rr-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(118deg, #123e31 0%, #176b50 58%, #24815d 100%);
        border-radius: 23px;
        padding: 29px 32px 27px;
        margin: 0 0 20px 0;
        color: #fff;
        box-shadow: 0 14px 32px rgba(17,67,49,.16);
    }
    .rr-hero:after {
        content: "";
        position: absolute;
        width: 280px; height: 280px;
        border: 1px solid rgba(255,255,255,.15);
        border-radius: 50%;
        right: -60px; top: -135px;
        box-shadow: 0 0 0 35px rgba(255,255,255,.035), 0 0 0 75px rgba(255,255,255,.025);
    }
    .rr-brandline { display:flex; align-items:center; gap:9px; margin-bottom:12px; position:relative; z-index:1; }
    .rr-brandmark {
        display:inline-flex; align-items:center; justify-content:center;
        width:33px; height:33px; border-radius:10px;
        background: var(--rr-lime); color:#153c2e; font-weight:900; font-size:18px;
    }
    .rr-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:11px; font-weight:800; color:#d5edc7; }
    .rr-hero h1 { color:#fff; font-size: clamp(2.0rem, 4vw, 3.15rem); line-height:1.02; margin:0; position:relative; z-index:1; }
    .rr-hero p { color:rgba(255,255,255,.86); max-width:760px; font-size:1rem; line-height:1.55; margin:12px 0 17px; position:relative; z-index:1; }
    .rr-pills { display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1; }
    .rr-pill { border:1px solid rgba(255,255,255,.24); background:rgba(255,255,255,.10); border-radius:999px; padding:6px 10px; font-size:12px; font-weight:650; color:#f6fff5; }
    .rr-section { margin: 21px 0 10px; }
    .rr-eyebrow { color:var(--rr-green); text-transform:uppercase; letter-spacing:.13em; font-size:10px; font-weight:850; margin-bottom:4px; }
    .rr-section-title { color:var(--rr-ink); font-size:1.34rem; font-weight:800; letter-spacing:-.025em; margin:0; }
    .rr-section-sub { color:var(--rr-muted); font-size:.91rem; margin-top:4px; }
    .rr-card {
        background:rgba(255,255,255,.88);
        border:1px solid var(--rr-line);
        border-radius:16px;
        padding:16px 17px;
        box-shadow:0 5px 18px rgba(24,58,43,.035);
        height:100%;
    }
    .rr-card-label { color:var(--rr-muted); font-size:.79rem; font-weight:750; margin-bottom:8px; }
    .rr-card-value { color:var(--rr-ink); font-weight:850; font-size:1.75rem; line-height:1.1; letter-spacing:-.04em; }
    .rr-card-note { color:#6a8076; font-size:.76rem; margin-top:8px; line-height:1.35; }
    .rr-card-accent { width:35px; height:4px; border-radius:20px; margin-bottom:11px; background:var(--rr-green); }
    .rr-card-accent.orange { background:var(--rr-orange); }
    .rr-card-accent.lime { background:#93bd47; }
    .rr-card-accent.blue { background:#68a7bc; }
    .rr-info {
        background:#eaf5ed; border:1px solid #cee6d4; border-left:4px solid #32805c;
        border-radius:12px; padding:12px 15px; color:#244e3e; font-size:.89rem; line-height:1.5;
    }
    .rr-warn {
        background:#fff4e7; border:1px solid #f5d4a9; border-left:4px solid #e7a14d;
        border-radius:12px; padding:12px 15px; color:#72461e; font-size:.89rem; line-height:1.5;
    }
    .rr-step { border:1px solid var(--rr-line); background:#fff; border-radius:13px; padding:14px 15px; margin-bottom:10px; }
    .rr-step-head { display:flex; gap:11px; align-items:flex-start; }
    .rr-step-num { display:flex; align-items:center; justify-content:center; width:31px; height:31px; min-width:31px; background:var(--rr-mint); color:var(--rr-green-dark); border-radius:10px; font-weight:850; }
    .rr-step-title { color:var(--rr-ink); font-weight:800; margin:2px 0 4px; }
    .rr-step-body { color:#587065; font-size:.88rem; line-height:1.5; }
    .rr-status { display:inline-block; border-radius:99px; padding:4px 9px; font-size:10px; font-weight:850; letter-spacing:.06em; }
    .rr-status.pass { color:#155c3e; background:#ddf5e5; }
    .rr-status.fail { color:#8d2f2f; background:#ffe4e1; }
    .rr-status.warn { color:#86551c; background:#ffefcf; }
    .rr-status.info { color:#345d81; background:#e5f0ff; }
    .rr-footer { border-top:1px solid var(--rr-line); margin-top:32px; padding-top:15px; color:#73867d; font-size:.78rem; }
    @media (max-width: 800px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .rr-hero { padding:22px 20px; }
        .rr-hero p { font-size:.92rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_heading(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="rr-section"><div class="rr-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="rr-section-title">{html.escape(title)}</div>'
        f'<div class="rr-section-sub">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, accent: str = "green") -> None:
    st.markdown(
        f'<div class="rr-card"><div class="rr-card-accent {html.escape(accent)}"></div>'
        f'<div class="rr-card-label">{html.escape(label)}</div>'
        f'<div class="rr-card-value">{html.escape(value)}</div>'
        f'<div class="rr-card-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    key = status.lower()
    cls = "pass" if key == "pass" else "fail" if key == "fail" else "warn" if key == "warn" else "info"
    return f'<span class="rr-status {cls}">{html.escape(status.upper())}</span>'


def is_available(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def numeric_total(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if mask is not None:
        values = values[mask]
    if values.isna().any():
        return None
    return float(values.sum())


def frame_signature(donations: pd.DataFrame, recipients: pd.DataFrame, fault: bool) -> str:
    return "|".join([
        donations.to_csv(index=False),
        recipients.to_csv(index=False),
        str(bool(fault)),
    ])


def plot_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#294b3d", size=12),
        margin=dict(l=12, r=12, t=28, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6eee7", zeroline=False, linecolor="#dce7de")
    fig.update_yaxes(showgrid=True, gridcolor="#e6eee7", zeroline=False, linecolor="#dce7de")
    return fig


SCENARIO_DESCRIPTIONS = {
    "Baseline": "Default example: all five donations are listed as available, with mixed food categories, deadlines and recipient needs.",
    "Time-critical food": "Prepared meals have a very short remaining time window, forcing the optimizer to remove links that arrive too late.",
    "Donation disruption": "Donation D03 is removed to simulate a source becoming unavailable before allocation.",
    "Supply shortage": "Donation quantities are reduced to 40% to demonstrate how the plan responds when there is not enough supply to meet all needs.",
    "Invalid input": "The quantity_kg field is intentionally removed so the Data Intake Agent can demonstrate schema rejection.",
}

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:4px 0 20px;">'
        '<div style="width:38px;height:38px;border-radius:12px;background:#d5ed8a;color:#133d30;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:19px;">R</div>'
        '<div><div style="font-weight:850;font-size:18px;color:#f5fff6;">RescueRoute</div>'
        '<div style="font-size:11px;color:#b9d4c5;letter-spacing:.04em;">FOOD RESCUE OPTIMIZER</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Simulation controls")
    scenario = st.selectbox(
        "Choose a scenario",
        ["Baseline", "Time-critical food", "Donation disruption", "Supply shortage", "Invalid input"],
        help="Each scenario changes synthetic data to test a different decision path.",
    )
    inject_fault = st.checkbox(
        "Demonstrate Critic + retry",
        value=False,
        help="Deliberately corrupts one candidate allocation so the Critic rejects it and the workflow tries again.",
    )
    st.caption("Fault injection is a test feature. It does not alter the optimizer's original output.")
    st.divider()
    st.markdown("### Bring your own data")
    donation_upload = st.file_uploader("Donations CSV", type=["csv"], key="donations", help="Upload a CSV using the required schema shown in the Data guide tab.")
    recipient_upload = st.file_uploader("Recipients CSV", type=["csv"], key="recipients", help="Upload a CSV using the required schema shown in the Data guide tab.")
    st.divider()
    st.markdown("<div style='font-size:12px;color:#c2d9cb;line-height:1.5;'>Demo data is synthetic. Coordinates and travel times are illustrative assumptions, not live maps or dispatch instructions.</div>", unsafe_allow_html=True)

scenario_key = scenario
donations, recipients, scenario_note = demo_scenario(scenario_key)
upload_errors: list[str] = []

if donation_upload is not None:
    try:
        donations = pd.read_csv(donation_upload)
        scenario_note = "A custom donations CSV is being used with the selected recipient dataset unless a recipients CSV is also uploaded."
    except Exception as exc:
        upload_errors.append(f"Could not read donations CSV: {exc}")
if recipient_upload is not None:
    try:
        recipients = pd.read_csv(recipient_upload)
        scenario_note += " A custom recipients CSV is also being used."
    except Exception as exc:
        upload_errors.append(f"Could not read recipients CSV: {exc}")

# -----------------------------------------------------------------------------
# Header and input overview
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="rr-hero">
      <div class="rr-brandline"><span class="rr-brandmark">↗</span><span class="rr-kicker">Multi-agent decision intelligence</span></div>
      <h1>RescueRoute</h1>
      <p>Turn surplus food into a feasible rescue plan. Allocate donations to the right community recipients using urgency, food compatibility, supply limits and delivery-time constraints—then validate every proposed allocation.</p>
      <div class="rr-pills"><span class="rr-pill">♻ Food recovery</span><span class="rr-pill">▦ Mixed-integer optimization</span><span class="rr-pill">✓ Critic validation</span><span class="rr-pill">⚙ Synthetic demo</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if upload_errors:
    for error in upload_errors:
        st.error(error)

cols = st.columns([1.45, 1, 1, 1])
active_supply = None
if "available" in donations.columns:
    active_supply = numeric_total(donations, "quantity_kg", is_available(donations["available"]))
eligible_demand = None
if {"demand_kg", "storage_capacity_kg"}.issubset(recipients.columns):
    demand = pd.to_numeric(recipients["demand_kg"], errors="coerce")
    capacity = pd.to_numeric(recipients["storage_capacity_kg"], errors="coerce")
    if not demand.isna().any() and not capacity.isna().any():
        eligible_demand = float(pd.concat([demand, capacity], axis=1).min(axis=1).sum())

with cols[0]:
    metric_card("Food currently available", f"{active_supply:,.0f} kg" if active_supply is not None else "—", "Sum of donations marked available", "green")
with cols[1]:
    metric_card("Donation sources", str(len(donations)), "Rows in the current donation table", "lime")
with cols[2]:
    metric_card("Recipient organizations", str(len(recipients)), "Rows in the recipient table", "blue")
with cols[3]:
    metric_card("Effective recipient capacity", f"{eligible_demand:,.0f} kg" if eligible_demand is not None else "—", "Sum of min(demand, storage capacity)", "orange")

st.markdown(
    f'<div class="rr-info" style="margin-top:14px;"><strong>Selected scenario:</strong> {html.escape(scenario)} — {html.escape(scenario_note)}<br>'
    '<strong>Important:</strong> these examples use synthetic organizations, quantities and local-grid coordinates. This milestone optimizes allocations; it does not yet optimize complete multi-stop vehicle routes.</div>',
    unsafe_allow_html=True,
)

precheck = validate_inputs(donations, recipients)
if precheck.valid:
    st.markdown(f'<div style="margin-top:10px;">{status_badge("PASS")} <span style="color:#5f766b;font-size:13px;margin-left:7px;">Input schema and basic value checks look valid. Run the workflow to create a plan.</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div style="margin-top:10px;">{status_badge("WARN")} <span style="color:#805321;font-size:13px;margin-left:7px;">Input validation has found an issue; running the workflow will show the rejection details.</span></div>', unsafe_allow_html=True)

run_col, spacer_col = st.columns([1.3, 4])
with run_col:
    run = st.button("▶  Run RescueRoute", type="primary", use_container_width=True, disabled=bool(upload_errors))
with spacer_col:
    st.markdown("<div style='padding:12px 0 0;color:#6a8076;font-size:13px;'>The optimizer will build feasible donor-to-recipient links, solve the allocation model, then send the result to the Critic.</div>", unsafe_allow_html=True)

current_signature = frame_signature(donations, recipients, inject_fault)
if run:
    with st.spinner("Validating data, optimizing allocations and checking constraints…"):
        result = run_rescue_workflow(donations.copy(), recipients.copy(), inject_critic_failure=inject_fault)
    st.session_state["last_result"] = result
    st.session_state["last_signature"] = current_signature
    st.session_state["last_scenario"] = scenario

saved_result = st.session_state.get("last_result")
result_is_current = saved_result is not None and st.session_state.get("last_signature") == current_signature

# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------
tab_results, tab_trace, tab_data, tab_guide = st.tabs([
    "📊  Results & impact",
    "🧭  Agent execution trace",
    "🗂️  Input data",
    "🧠  How it works",
])

with tab_results:
    section_heading("Optimization output", "Allocation results", "Understand how much food can be assigned, which recipients are served, and whether the plan passes validation.")
    if not result_is_current:
        st.markdown(
            '<div class="rr-warn"><strong>No current result yet.</strong> Select a scenario in the sidebar and click <em>Run RescueRoute</em>. If you change the scenario, uploaded data or Critic toggle, run it again so the displayed result matches those inputs.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### What you will see here")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown('<div class="rr-card"><div class="rr-card-accent"></div><div class="rr-card-label">1 · Allocation plan</div><div style="font-weight:800;color:#17352d;">Donor → Recipient</div><div class="rr-card-note">Kilograms assigned on each feasible pair.</div></div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div class="rr-card"><div class="rr-card-accent orange"></div><div class="rr-card-label">2 · Impact metrics</div><div style="font-weight:800;color:#17352d;">Demand coverage</div><div class="rr-card-note">Allocated food compared with recipient demand and storage capacity.</div></div>', unsafe_allow_html=True)
        with p3:
            st.markdown('<div class="rr-card"><div class="rr-card-accent blue"></div><div class="rr-card-label">3 · Validation</div><div style="font-weight:800;color:#17352d;">Critic verdict</div><div class="rr-card-note">The proposed plan is checked against explicit constraints.</div></div>', unsafe_allow_html=True)
    else:
        result = saved_result
        if result["status"] == "rejected":
            st.markdown('<div class="rr-warn"><strong>Workflow rejected the input.</strong> The system stopped before optimization because the required schema or input values did not pass validation.</div>', unsafe_allow_html=True)
            for error in result["input_validation"].errors:
                st.error(error)
        elif result["status"] == "solver_failed":
            st.error("The optimization solver failed. Open the execution trace to inspect the recorded failure.")
        else:
            summary = result["summary"]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Allocated food", f"{summary['total_allocated_kg']:,} kg", "Integer-kilogram allocation total", "green")
            with k2:
                metric_card("Effective demand", f"{summary['demand_capacity_kg']:,} kg", "Recipient demand capped by storage capacity", "blue")
            with k3:
                metric_card("Demand covered", f"{summary['demand_fulfilled_pct']:.1f}%", "Share of effective demand allocated", "lime")
            with k4:
                metric_card("Recipients served", str(summary["recipients_served"]), f"{summary['allocation_rows']} donor-recipient allocation rows", "orange")

            if result["status"] == "completed_after_retry":
                st.markdown('<div class="rr-info" style="margin-top:15px;"><strong>Critic + retry succeeded.</strong> The Critic rejected a deliberately corrupted candidate. The Correction Agent reran the optimizer from source data, and the corrected plan passed validation.</div>', unsafe_allow_html=True)
            elif result["status"] == "completed":
                st.markdown('<div class="rr-info" style="margin-top:15px;"><strong>Critic approved the allocation.</strong> All implemented plan constraints passed deterministic validation.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rr-warn" style="margin-top:15px;"><strong>Manual review required.</strong> The plan did not pass validation after the retry limit.</div>', unsafe_allow_html=True)

            plan = result["plan"]
            section_heading("Allocation plan", "Where the food goes", "Each row is one donor-to-recipient allocation link, not a complete vehicle route.")
            if plan.empty:
                st.warning("No eligible allocations were found. Review the constraints and the excluded-link list.")
            else:
                # Flow visualization: donation source -> recipient organization.
                donor_labels = plan[["donation_id", "donor_name"]].drop_duplicates().sort_values("donation_id")
                recipient_labels = plan[["recipient_id", "recipient_name"]].drop_duplicates().sort_values("recipient_id")
                node_labels = [f"{r.donation_id} · {r.donor_name}" for r in donor_labels.itertuples(index=False)] + [f"{r.recipient_id} · {r.recipient_name}" for r in recipient_labels.itertuples(index=False)]
                donor_node = {r.donation_id: i for i, r in enumerate(donor_labels.itertuples(index=False))}
                recipient_node = {r.recipient_id: len(donor_labels) + i for i, r in enumerate(recipient_labels.itertuples(index=False))}
                source = [donor_node[x] for x in plan["donation_id"]]
                target = [recipient_node[x] for x in plan["recipient_id"]]
                values = plan["allocated_kg"].astype(float).tolist()
                hover = [f"{r.donation_id} → {r.recipient_id}<br>{r.food_type}<br>{int(r.allocated_kg)} kg" for r in plan.itertuples(index=False)]
                flow = go.Figure(data=[go.Sankey(
                    arrangement="snap",
                    node=dict(pad=20, thickness=17, line=dict(color="#d5e3d8", width=1), label=node_labels,
                              color=["#247455"] * len(donor_labels) + ["#9bbd58"] * len(recipient_labels)),
                    link=dict(source=source, target=target, value=values, label=hover,
                              color=["rgba(54,133,91,0.32)"] * len(values),
                              customdata=hover, hovertemplate="%{customdata}<extra></extra>"),
                )])
                flow.update_layout(title="Food flow · donor sources to recipient organizations", height=410,
                                   paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#294b3d", size=11),
                                   margin=dict(l=5, r=5, t=48, b=5))
                st.plotly_chart(flow, use_container_width=True)

                delivered = plan.groupby("recipient_id", as_index=False).agg(
                    recipient_name=("recipient_name", "first"), allocated_kg=("allocated_kg", "sum")
                )
                if {"demand_kg", "storage_capacity_kg"}.issubset(recipients.columns):
                    demand_df = recipients[["recipient_id", "recipient_name", "demand_kg", "storage_capacity_kg"]].copy()
                    demand_df["effective_demand_kg"] = demand_df[["demand_kg", "storage_capacity_kg"]].min(axis=1)
                    allocated_only = delivered[["recipient_id", "allocated_kg"]]
                    delivered = demand_df.merge(allocated_only, on="recipient_id", how="left")
                    delivered["allocated_kg"] = delivered["allocated_kg"].fillna(0)
                    bar = go.Figure()
                    bar.add_bar(name="Effective demand", x=delivered["recipient_name"], y=delivered["effective_demand_kg"], marker_color="#d8e7d9")
                    bar.add_bar(name="Allocated", x=delivered["recipient_name"], y=delivered["allocated_kg"], marker_color="#25835d")
                    bar.update_layout(barmode="group", title="Recipient demand vs. allocated food", xaxis_title="Recipient", yaxis_title="Kilograms")
                    st.plotly_chart(plot_style(bar, 350), use_container_width=True)

                display_cols = ["donation_id", "food_type", "recipient_name", "allocated_kg", "distance_km", "estimated_delivery_hours", "urgency"]
                st.dataframe(plan[display_cols], use_container_width=True, hide_index=True)
                st.caption("`distance_km` is straight-line distance in the synthetic coordinate grid. `estimated_delivery_hours` uses distance ÷ 25 km/h + a 0.25-hour handling buffer. These are estimates, not road-route ETAs.")
                dl1, dl2 = st.columns([1, 2])
                with dl1:
                    st.download_button(
                        "↓ Download allocation CSV",
                        plan.to_csv(index=False).encode("utf-8"),
                        file_name="rescueroute_allocation_plan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl2:
                    st.caption(f"Solver: {result.get('solver_metadata', {}).get('solver', 'Not available')} · {result.get('solver_metadata', {}).get('eligible_links', 0)} feasible donor-recipient links considered.")

            critic = result.get("critic_validation")
            section_heading("Quality gate", "Critic validation", "A solver result is treated as a candidate plan until rule checks pass.")
            if critic:
                if critic.valid:
                    st.success("Final plan passed all implemented deterministic checks.")
                else:
                    st.error("Final plan failed validation.")
                for warning in critic.warnings:
                    st.warning(warning)
                for error in critic.errors:
                    st.error(error)

            with st.expander("Solver details and excluded allocation links"):
                metadata = result.get("solver_metadata", {})
                shown_meta = {k: v for k, v in metadata.items() if k != "exclusions"}
                st.json(shown_meta)
                exclusions = metadata.get("exclusions", [])
                st.markdown(f"**Excluded links:** {len(exclusions)}")
                if exclusions:
                    st.dataframe(pd.DataFrame({"Eligibility explanation": exclusions[:100]}), use_container_width=True, hide_index=True)
                else:
                    st.write("No donation-recipient links were excluded by the eligibility checks.")

with tab_trace:
    section_heading("Transparent execution", "Agent execution trace", "Every stage records its action and outcome. Expand an entry to understand the decision made.")
    if not result_is_current:
        st.info("Run the workflow to generate the execution trace for the selected scenario.")
    else:
        result = saved_result
        logs = result.get("logs", [])
        counts = {key: sum(1 for entry in logs if entry.status == key) for key in ["PASS", "WARN", "FAIL", "INFO"]}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Passed", counts["PASS"])
        c2.metric("Warnings", counts["WARN"])
        c3.metric("Failures / rejected checks", counts["FAIL"])
        c4.metric("Information events", counts["INFO"])
        for idx, entry in enumerate(logs, start=1):
            with st.expander(f"{idx:02d}  ·  {entry.stage}  ·  {entry.action}", expanded=(idx == 1 or entry.status == "FAIL")):
                st.markdown(status_badge(entry.status), unsafe_allow_html=True)
                st.markdown(f"**Outcome:** {entry.result}")
                if entry.details:
                    st.markdown("**Technical details**")
                    st.json(entry.details)
        log_rows = [{"Stage": entry.stage, "Action": entry.action, "Result": entry.result, "Status": entry.status} for entry in logs]
        log_frame = pd.DataFrame(log_rows)
        st.download_button("↓ Download execution log CSV", log_frame.to_csv(index=False).encode("utf-8"), "rescueroute_execution_log.csv", "text/csv")

        section_heading("What the agents mean", "Roles in this milestone", "These are deterministic Python agents/components, not LLM-driven autonomous agents.")
        role_data = [
            ("Data Intake Agent", "Validates required columns, identifiers, numeric fields and valid ranges. Rejects invalid inputs before the solver runs."),
            ("Planner Agent", "Profiles the available supply, effective recipient demand and urgency context. In this milestone the plan is a fixed workflow, with routing decisions controlled by validation outcomes."),
            ("Allocation Optimizer", "Calls SciPy's mixed-integer linear optimizer to assign whole kilograms across feasible donation-recipient pairs."),
            ("Critic Agent", "Checks the returned candidate against supply, demand, storage, food acceptance and estimated time-window constraints."),
            ("Correction Agent", "If the Critic rejects a candidate, reruns optimization using the original input data and permits one retry."),
            ("Report Agent", "Calculates summary metrics and records a final result for the UI and downloadable reports."),
        ]
        for title, body in role_data:
            st.markdown(f'<div class="rr-step"><div class="rr-step-head"><div class="rr-step-num">✓</div><div><div class="rr-step-title">{html.escape(title)}</div><div class="rr-step-body">{html.escape(body)}</div></div></div></div>', unsafe_allow_html=True)

with tab_data:
    section_heading("Inspect inputs", "Data explorer", "The sample tables are fictional, intentionally small, and designed to exercise allocation constraints. Uploaded CSVs replace the corresponding sample table.")
    dtab, rtab, map_tab, dict_tab = st.tabs(["Donation inventory", "Recipient requirements", "Synthetic coordinate view", "Field definitions"])
    with dtab:
        st.markdown("**What this table represents:** food available from markets, warehouses, farms or canteens. Each row is one donation offer.")
        st.dataframe(donations, use_container_width=True, hide_index=True)
        if "food_type" in donations.columns and "quantity_kg" in donations.columns:
            food_chart_data = donations.copy()
            food_chart_data["quantity_kg"] = pd.to_numeric(food_chart_data["quantity_kg"], errors="coerce")
            food_chart = food_chart_data.groupby("food_type", as_index=False)["quantity_kg"].sum().sort_values("quantity_kg", ascending=True)
            fig_food = go.Figure(go.Bar(x=food_chart["quantity_kg"], y=food_chart["food_type"], orientation="h", marker_color="#25835d"))
            fig_food.update_layout(title="Donation quantities by food category", xaxis_title="Kilograms", yaxis_title="Food type")
            st.plotly_chart(plot_style(fig_food, 280), use_container_width=True)
    with rtab:
        st.markdown("**What this table represents:** community kitchens, shelters and other recipient organizations that have demand, storage limits and accepted food categories.")
        st.dataframe(recipients, use_container_width=True, hide_index=True)
        if {"recipient_name", "demand_kg", "storage_capacity_kg", "urgency"}.issubset(recipients.columns):
            rchart = recipients.copy()
            for c in ["demand_kg", "storage_capacity_kg", "urgency"]:
                rchart[c] = pd.to_numeric(rchart[c], errors="coerce")
            fig_r = go.Figure()
            fig_r.add_bar(name="Demand", x=rchart["recipient_name"], y=rchart["demand_kg"], marker_color="#b9d6c1")
            fig_r.add_bar(name="Storage capacity", x=rchart["recipient_name"], y=rchart["storage_capacity_kg"], marker_color="#25835d")
            fig_r.update_layout(barmode="group", title="Recipient demand vs storage capacity", xaxis_title="Recipient", yaxis_title="Kilograms")
            st.plotly_chart(plot_style(fig_r, 320), use_container_width=True)
    with map_tab:
        st.markdown('<div class="rr-warn"><strong>Not a geographic map.</strong> x_km and y_km are made-up Cartesian coordinates so the prototype can estimate straight-line distance. They are not GPS coordinates, actual Pune locations or road travel distances.</div>', unsafe_allow_html=True)
        if {"x_km", "y_km"}.issubset(donations.columns) and {"x_km", "y_km"}.issubset(recipients.columns):
            fig_xy = go.Figure()
            fig_xy.add_scatter(x=donations["x_km"], y=donations["y_km"], mode="markers+text", text=donations.get("donation_id"), textposition="top center", name="Donors", marker=dict(size=13, color="#25835d", symbol="circle"))
            fig_xy.add_scatter(x=recipients["x_km"], y=recipients["y_km"], mode="markers+text", text=recipients.get("recipient_id"), textposition="bottom center", name="Recipients", marker=dict(size=13, color="#df9b4b", symbol="diamond"))
            fig_xy.update_layout(title="Illustrative local-grid positions", xaxis_title="x coordinate (km)", yaxis_title="y coordinate (km)")
            st.plotly_chart(plot_style(fig_xy, 400), use_container_width=True)
        else:
            st.info("Coordinate fields are missing from the selected CSV.")
    with dict_tab:
        donation_dictionary = pd.DataFrame([
            ("donation_id", "Unique identifier for a donation row, e.g. D01."),
            ("donor_name", "Fictional organization supplying the food."),
            ("food_type", "Category of food offered; must match the recipient's accepted types."),
            ("quantity_kg", "Total kilograms offered by that donation."),
            ("expiry_hours", "Remaining allowable planning window in hours; simplified and not a food-safety determination."),
            ("available", "Whether the optimizer may use the donation (true/false style value)."),
            ("x_km, y_km", "Synthetic two-dimensional coordinates used to estimate straight-line distance."),
        ], columns=["Donation field", "Meaning"])
        recipient_dictionary = pd.DataFrame([
            ("recipient_id", "Unique identifier for a receiving organization, e.g. R01."),
            ("recipient_name", "Fictional recipient organization."),
            ("demand_kg", "How many kilograms the recipient requests."),
            ("storage_capacity_kg", "Maximum kilograms the recipient can store."),
            ("urgency", "Integer from 1 (lower priority) to 5 (higher priority) in this simulation."),
            ("delivery_deadline_hours", "Latest modeled delivery time for the recipient."),
            ("accepts_food_types", "Pipe-separated list of food categories accepted by the recipient."),
            ("x_km, y_km", "Synthetic recipient coordinates used for straight-line distance estimation."),
        ], columns=["Recipient field", "Meaning"])
        st.markdown("**Donation data**")
        st.dataframe(donation_dictionary, use_container_width=True, hide_index=True)
        st.markdown("**Recipient data**")
        st.dataframe(recipient_dictionary, use_container_width=True, hide_index=True)
        st.markdown("**Calculated fields in the output**")
        st.dataframe(pd.DataFrame([
            ("allocated_kg", "Whole-kilogram quantity assigned on this donor-recipient link."),
            ("distance_km", "Euclidean distance between the synthetic coordinates."),
            ("estimated_delivery_hours", "0.25-hour handling buffer + straight-line distance / 25 km/h assumed speed."),
            ("objective_weight", "Per-kilogram score used by the optimization objective to balance urgency, distance and shorter remaining expiry."),
        ], columns=["Output field", "Meaning"]), use_container_width=True, hide_index=True)

with tab_guide:
    section_heading("Methodology", "How the workflow works", "A step-by-step explanation of where the data goes, what each stage decides, and how the optimizer chooses an allocation.")
    workflow_steps = [
        ("1", "Data Intake Agent · validate input", "Checks that the donations and recipients tables contain all required columns. It also checks for empty or duplicate IDs, missing/non-numeric values, non-finite numbers, negative quantities/capacities, valid urgency values and available flags. If required fields fail, the workflow stops before optimization."),
        ("2", "Planner Agent · profile the problem", "Counts donation rows and recipients, sums donations marked available and estimates effective demand as min(requested demand, storage capacity) for each recipient. It records these values in the execution log."),
        ("3", "Eligibility builder · create candidate links", "For every available donation-recipient pair, checks whether the recipient accepts that food. Computes straight-line distance from the synthetic coordinates and estimated delivery time = 0.25-hour handling buffer + distance / 25 km/h. It excludes links that miss the donation window or recipient deadline and records an explanation for each exclusion."),
        ("4", "Allocation Optimizer · solve the MILP", "Each feasible donor-recipient link gets an integer decision variable measured in kg. SciPy's mixed-integer linear programming solver maximizes the sum of per-kilogram scores. The constraints prevent a donation from being overused and prevent a recipient from receiving more than the lesser of its demand and storage capacity."),
        ("5", "Critic Agent · validate the candidate", "Checks that allocations are positive whole kilograms; donation and recipient IDs exist; supply, demand and storage limits are respected; the recipient accepts the food type; and modeled delivery time meets both the food window and the recipient deadline."),
        ("6", "Correction Agent · retry only if rejected", "If the Critic finds a violation, the system regenerates the plan from the trusted original inputs and retries validation once. If that retry fails, the result is marked for manual review instead of being treated as accepted."),
        ("7", "Report Agent · summarize and export", "Computes total allocated kilograms, effective demand, demand coverage and recipients served. The UI displays the allocation flow, tabular plan, exclusions, validation result and downloadable CSV files."),
    ]
    for num, title, body in workflow_steps:
        st.markdown(f'<div class="rr-step"><div class="rr-step-head"><div class="rr-step-num">{html.escape(num)}</div><div><div class="rr-step-title">{html.escape(title)}</div><div class="rr-step-body">{html.escape(body)}</div></div></div></div>', unsafe_allow_html=True)

    section_heading("Optimization model", "How the allocation score works", "The numbers below are transparent demonstration weights, not a trained prediction model or official food distribution policy.")
    st.latex(r"\text{score}_{ij}=100+25u_j-2d_{ij}+\frac{8}{\max(e_i,0.5)}")
    st.markdown(r"""
    - \(u_j\): recipient urgency (1–5); higher urgency increases the score per kilogram.
    - \(d_{ij}\): straight-line donor-to-recipient distance in the synthetic coordinate grid; distance reduces the score.
    - \(e_i\): donor's remaining expiry window in hours; a shorter window adds urgency to the score, with a floor at 0.5 hours to limit the bonus.
    - Decision variables are **whole kilograms** assigned to eligible donor-recipient links.

    The model is constrained by total supply for each donation and by `min(demand_kg, storage_capacity_kg)` for each recipient. All score weights are chosen for this educational prototype and can be adjusted for experiments.
    """)

    section_heading("Assumptions and scope", "What this milestone does—and does not—claim", "Knowing the limits is important for a technically accurate viva.")
    st.markdown("""
    <div class="rr-info"><strong>Implemented now:</strong> deterministic intake validation, eligibility filtering, mixed-integer allocation, Critic validation, one correction retry, scenario switching, execution logs and downloadable results.</div>
    <div class="rr-warn" style="margin-top:10px;"><strong>Not implemented yet:</strong> multi-stop vehicle route optimization, live road traffic/time estimates, live location maps, real donor/recipient verification and dynamic replanning after an actual vehicle failure. Those belong in the next milestone.</div>
    <div class="rr-warn" style="margin-top:10px;"><strong>Food safety:</strong> `expiry_hours` is a simplified scenario input, not a determination that food is safe to donate or consume. Real-world use would require verified food handling, temperature, quality, regulatory and human approval checks.</div>
    """, unsafe_allow_html=True)

with st.expander("CSV schema and download starter datasets"):
    st.markdown("**Donation columns:** `donation_id`, `donor_name`, `food_type`, `quantity_kg`, `expiry_hours`, `x_km`, `y_km`, `available`.")
    st.markdown("**Recipient columns:** `recipient_id`, `recipient_name`, `demand_kg`, `storage_capacity_kg`, `urgency`, `delivery_deadline_hours`, `accepts_food_types`, `x_km`, `y_km`.")
    st.markdown("`accepts_food_types` must be pipe-separated, e.g. `Produce|Grains|Prepared meals`. Coordinates are synthetic local-grid kilometres.")
    st.download_button("Download sample donations CSV", sample_donations().to_csv(index=False).encode("utf-8"), "donations_sample.csv", "text/csv")
    st.download_button("Download sample recipients CSV", sample_recipients().to_csv(index=False).encode("utf-8"), "recipients_sample.csv", "text/csv")

st.markdown(
    '<div class="rr-footer">RescueRoute · Milestone 1 · Deterministic multi-agent workflow + SciPy mixed-integer allocation optimization. Next: OR-Tools route optimization and disruption replanning.</div>',
    unsafe_allow_html=True,
)
