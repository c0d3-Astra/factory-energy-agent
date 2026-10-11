"""Factory Energy Bill Agent — Streamlit UI
Talks to the API over HTTP only. Four pages:
  Ask      — question box, bridge waterfall, rich charts
  Data     — health, versions, upload, validation
  Explore  — meter breakdown, idle heatmap, leak evidence
  Eval     — evaluation report
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

CAUSE_LABEL = {
    "tariff": "Tariff (price)",
    "volume": "Volume",
    "product_mix": "Product mix",
    "weather": "Weather",
    "waste": "Waste (controllable)",
}
CAUSE_COLOR = {
    "tariff": "#3b82f6",       # blue
    "volume": "#8b5cf6",       # violet
    "product_mix": "#06b6d4",  # cyan
    "weather": "#10b981",      # emerald
    "waste": "#f59e0b",        # amber — controllable
}
COLORS = {
    "increase": "#c2410c",   # red-orange
    "decrease": "#15803d",   # green
    "neutral": "#334155",    # dark slate
    "waste": "#f59e0b",
    "production": "#3b82f6",
    "idle": "#f97316",
}
EXAMPLES = [
    "Why did the bill go up 15%?",
    "Which assets use the most energy?",
    "How much energy is used when nothing is produced?",
    "What should we do first?",
    "Which action pays back fastest?",
    "What if the tariff had not changed?",
    "What will the tariff be next year?",
    "What changed after the correction?",
]

st.set_page_config(
    page_title="Factory Energy Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: #1e293b; border-radius: 10px; padding: 16px 20px;
        border-left: 4px solid #3b82f6; margin-bottom: 8px;
    }
    .metric-card.amber { border-left-color: #f59e0b; }
    .metric-card.green { border-left-color: #10b981; }
    .metric-card.red   { border-left-color: #ef4444; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #1e293b; padding-bottom: 4px;
    }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-blue   { background: #1d4ed8; color: #fff; }
    .badge-amber  { background: #b45309; color: #fff; }
    .badge-green  { background: #15803d; color: #fff; }
    .badge-red    { background: #b91c1c; color: #fff; }
    .badge-slate  { background: #475569; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def call(method: str, path: str, **kw):
    try:
        r = requests.request(method, API + path, timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API}. Is it running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


def fmt_range(r):
    if not r:
        return "n/a"
    return f"{r[0]:,.0f} – {r[1]:,.0f}"


def answer_badge(atype: str) -> str:
    mapping = {
        "decomposition": ("blue", "Decomposition"),
        "recommendation": ("green", "Recommendation"),
        "calculated": ("slate", "Calculated"),
        "not_in_data": ("amber", "Not in data"),
        "clarification_needed": ("amber", "Clarification needed"),
        "refused": ("red", "Refused"),
    }
    cls, label = mapping.get(atype, ("slate", atype))
    return f'<span class="badge badge-{cls}">{label}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_bridge_waterfall(bridge: list, total: float, period: str, ref: str):
    """Waterfall chart: pct-point contribution per cause with range whiskers."""
    rows = sorted(bridge, key=lambda r: (r["cause"] == "waste", -r["pct_points"]))
    causes = [CAUSE_LABEL[r["cause"]] for r in rows] + ["Total change"]
    values = [r["pct_points"] for r in rows] + [total]
    measures = ["relative"] * len(rows) + ["total"]
    colors_bar = [CAUSE_COLOR.get(r["cause"], "#64748b") for r in rows] + [COLORS["neutral"]]
    text_labels = [
        f"{r['pct_points']:+.1f} pp<br>({r['range'][0]:.1f}–{r['range'][1]:.1f})"
        for r in rows
    ] + [f"{total:+.1f}%"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=causes, y=values,
        text=text_labels, textposition="outside",
        connector={"line": {"color": "#334155", "width": 1}},
        increasing={"marker": {"color": COLORS["increase"]}},
        decreasing={"marker": {"color": COLORS["decrease"]}},
        totals={"marker": {"color": COLORS["neutral"]}},
    ))

    # Range whiskers overlay
    error_x = [(r["range"][1] - r["range"][0]) / 2 for r in rows]
    fig.add_trace(go.Scatter(
        x=[CAUSE_LABEL[r["cause"]] for r in rows],
        y=[r["pct_points"] for r in rows],
        mode="markers",
        marker=dict(size=8, color=[CAUSE_COLOR.get(r["cause"], "#64748b") for r in rows], symbol="diamond"),
        error_y=dict(type="data", array=error_x, color="#94a3b8", thickness=2, width=6),
        name="90% range",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(text=f"Bill change {ref} → {period}: {total:+.1f}% — causes add up exactly", font_size=14),
        yaxis_title="Percentage points of reference bill",
        legend=dict(orientation="h", y=-0.15),
        height=460, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    fig.add_annotation(
        text="Each bar = share of last quarter's bill. Bars add up to total. Ranges do not add up.",
        xref="paper", yref="paper", x=0, y=-0.22, showarrow=False,
        font=dict(size=11, color="#64748b"), align="left",
    )
    return fig


def chart_cause_pie(bridge: list):
    """Pie chart: absolute cost share by cause."""
    labels = [CAUSE_LABEL[r["cause"]] for r in bridge]
    values = [abs(r["money"]) for r in bridge]
    colors = [CAUSE_COLOR.get(r["cause"], "#64748b") for r in bridge]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Cost impact: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        hole=0.42,
    ))
    fig.update_layout(
        title="Cost impact share by cause",
        legend=dict(orientation="v", x=1.05),
        height=380, margin=dict(t=60, b=20, l=0, r=120),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    fig.add_annotation(text="Cost<br>split", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#94a3b8"), xanchor="center")
    return fig


def chart_waste_breakdown(bridge: list):
    """Bar chart: waste split by meter from bridge components."""
    waste_row = next((r for r in bridge if r["cause"] == "waste"), None)
    if not waste_row or not waste_row.get("components"):
        return None
    comps = waste_row["components"]
    meters = list(comps.keys())
    vals = list(comps.values())
    colors = [COLORS["waste"] if m != "unmetered" else "#64748b" for m in meters]
    fig = go.Figure(go.Bar(
        x=meters, y=vals,
        marker_color=colors,
        text=[f"{v:.2f} pp" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Waste: %{y:.2f} pp<extra></extra>",
    ))
    fig.update_layout(
        title="Waste (controllable) by sub-meter",
        xaxis_title="Sub-meter", yaxis_title="Percentage points",
        height=340, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    return fig


def chart_savings_bar(recs: list):
    """Grouped bar chart: savings per recommended action (kWh, money, CO2e)."""
    if not recs:
        return None
    labels = [f"Action {i+1}: {r['action'][:35]}..." if len(r['action']) > 35 else f"Action {i+1}: {r['action']}"
              for i, r in enumerate(recs)]
    kwh_lo = [r["saving_kwh_per_year"][0] / 1000 for r in recs]
    kwh_hi = [r["saving_kwh_per_year"][1] / 1000 for r in recs]
    money_lo = [r["saving_money_per_year"][0] for r in recs]
    money_hi = [r["saving_money_per_year"][1] for r in recs]
    co2_lo = [r["saving_co2e_t_per_year"][0] for r in recs]
    co2_hi = [r["saving_co2e_t_per_year"][1] for r in recs]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Savings (MWh/yr)", "Savings (currency/yr)", "CO2e (t/yr)"),
                        horizontal_spacing=0.08)

    for i, (label, lo, hi) in enumerate(zip(labels, kwh_lo, kwh_hi)):
        fig.add_trace(go.Bar(
            x=[label], y=[hi], name=label if i == 0 else None,
            marker_color=COLORS["decrease"],
            error_y=dict(type="data", array=[(hi - lo) / 2], color="#94a3b8"),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>{lo:.0f}–{hi:.0f} MWh/yr<extra></extra>",
        ), row=1, col=1)
    for i, (label, lo, hi) in enumerate(zip(labels, money_lo, money_hi)):
        fig.add_trace(go.Bar(
            x=[label], y=[hi], marker_color="#6366f1",
            error_y=dict(type="data", array=[(hi - lo) / 2], color="#94a3b8"),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>{lo:,.0f}–{hi:,.0f}/yr<extra></extra>",
        ), row=1, col=2)
    for i, (label, lo, hi) in enumerate(zip(labels, co2_lo, co2_hi)):
        fig.add_trace(go.Bar(
            x=[label], y=[hi], marker_color="#14b8a6",
            error_y=dict(type="data", array=[(hi - lo) / 2], color="#94a3b8"),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>{lo:.0f}–{hi:.0f} t CO2e/yr<extra></extra>",
        ), row=1, col=3)

    fig.update_layout(
        title="Potential savings per recommended action (ranges shown as error bars)",
        height=420, margin=dict(t=80, b=60),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    fig.update_xaxes(tickangle=25, tickfont_size=10)
    return fig


def chart_payback_bar(recs: list):
    """Horizontal bar chart: payback months per action."""
    if not recs:
        return None
    labels = [f"A{i+1}: {r['action'][:40]}" for i, r in enumerate(recs)]
    mid = [(r["payback_months"][0] + r["payback_months"][1]) / 2 if r.get("payback_months") else 0 for r in recs]
    lo = [r["payback_months"][0] if r.get("payback_months") else 0 for r in recs]
    hi = [r["payback_months"][1] if r.get("payback_months") else 0 for r in recs]

    ease_color = {"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"}
    colors = [ease_color.get(r.get("ease", "medium"), "#64748b") for r in recs]

    fig = go.Figure(go.Bar(
        y=labels, x=mid, orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=[(h - l) / 2 for l, h in zip(lo, hi)], color="#94a3b8"),
        text=[f"{l}–{h} mo" for l, h in zip(lo, hi)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Payback: %{x:.1f} months (midpoint)<extra></extra>",
    ))
    fig.update_layout(
        title="Payback period by action — sorted fastest first (colour = ease: green easy / amber medium / red hard)",
        xaxis_title="Months to payback",
        height=max(300, 80 * len(recs)), margin=dict(t=60, b=20, l=20, r=120),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    return fig


def chart_asset_pie(assets: list):
    """Pie chart: kWh share by sub-meter."""
    if not assets:
        return None
    df = pd.DataFrame(assets)
    if "kwh" not in df.columns and "idle_kwh" not in df.columns:
        return None
    kwh_col = "kwh" if "kwh" in df.columns else "idle_kwh"
    meter_col = "meter" if "meter" in df.columns else df.columns[0]

    palette = px.colors.qualitative.Set2
    colors = [palette[i % len(palette)] for i in range(len(df))]
    fig = go.Figure(go.Pie(
        labels=df[meter_col], values=df[kwh_col],
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} kWh<br>%{percent}<extra></extra>",
        hole=0.38,
    ))
    fig.update_layout(
        title="Energy share by sub-meter",
        height=380, margin=dict(t=60, b=20, l=0, r=120),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    return fig


def chart_idle_bar(assets: list):
    """Grouped bar chart: production vs idle kWh per asset."""
    if not assets:
        return None
    df = pd.DataFrame(assets)
    required = {"meter", "idle_kwh"}
    if not required.issubset(df.columns):
        # try generic assets table
        return None
    df["production_kwh"] = df.get("production_kwh", pd.Series([None] * len(df)))

    fig = go.Figure()
    if "production_kwh" in df.columns and df["production_kwh"].notna().any():
        fig.add_trace(go.Bar(
            name="Production hours",
            x=df["meter"], y=df["production_kwh"],
            marker_color=COLORS["production"],
            hovertemplate="<b>%{x}</b><br>Production kWh: %{y:,.0f}<extra></extra>",
        ))
    fig.add_trace(go.Bar(
        name="Idle / non-production",
        x=df["meter"], y=df["idle_kwh"],
        marker_color=COLORS["idle"],
        hovertemplate="<b>%{x}</b><br>Idle kWh: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title="Production vs idle energy by asset",
        xaxis_title="Sub-meter / asset",
        yaxis_title="kWh",
        height=380, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def chart_idle_ratio_bar(assets: list):
    """Horizontal bar: idle ratio per asset."""
    if not assets:
        return None
    df = pd.DataFrame(assets)
    if "idle_ratio" not in df.columns:
        return None
    meter_col = "meter" if "meter" in df.columns else df.columns[0]
    df_sorted = df.sort_values("idle_ratio", ascending=True)
    colors = [COLORS["waste"] if v > 0.3 else COLORS["decrease"] for v in df_sorted["idle_ratio"]]

    fig = go.Figure(go.Bar(
        y=df_sorted[meter_col], x=df_sorted["idle_ratio"] * 100,
        orientation="h", marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df_sorted["idle_ratio"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Idle ratio: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=30, line_dash="dash", line_color="#f59e0b",
                  annotation_text="30% threshold", annotation_position="top right")
    fig.update_layout(
        title="Idle ratio per asset (% of energy used outside production hours)",
        xaxis_title="Idle ratio (%)",
        height=max(280, 55 * len(df)), margin=dict(t=60, b=20, l=10, r=80),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    return fig


def chart_energy_line(result_rows: list, title: str = "Energy over time"):
    """Line chart from rows with date/ts and kwh columns."""
    if not result_rows:
        return None
    df = pd.DataFrame(result_rows)
    date_col = next((c for c in ["date", "ts", "local_date", "period"] if c in df.columns), None)
    if not date_col:
        return None
    kwh_cols = [c for c in df.columns if "kwh" in c.lower() or "energy" in c.lower()]
    if not kwh_cols:
        return None

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, col in enumerate(kwh_cols):
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[col],
            mode="lines+markers", name=col,
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:,.0f}} kWh<extra></extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title="Date / Period", yaxis_title="kWh",
        height=380, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_comparison_bar(changes: list):
    """Bar chart: before vs after for diff version changes."""
    if not changes:
        return None
    df = pd.DataFrame(changes)
    if not {"item", "old", "new"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["old", "new"])
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=df["item"], y=df["old"], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="After",  x=df["item"], y=df["new"], marker_color="#06b6d4"))
    fig.update_layout(
        barmode="group", title="Before vs after data update",
        xaxis_title="Item", yaxis_title="Value",
        height=380, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_period_bar(bridge: list, period: str, ref: str):
    """Grouped bar: money impact by cause for both periods."""
    causes = [CAUSE_LABEL[r["cause"]] for r in bridge]
    money = [r["money"] for r in bridge]
    colors = [CAUSE_COLOR.get(r["cause"], "#64748b") for r in bridge]
    fig = go.Figure(go.Bar(
        x=causes, y=money,
        marker_color=colors,
        text=[f"{m:,.0f}" for m in money],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Cost impact: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Cost impact per cause ({ref} → {period})",
        xaxis_title="Cause", yaxis_title="Cost impact (currency)",
        height=360, margin=dict(t=60, b=20),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Factory Energy Agent")
    st.caption("HCLTech AI Engineer Hackathon · Track D")

    st.markdown("---")
    st.markdown("### System status")
    h = call("GET", "/health")
    if h:
        api_ok = h.get("api") == "ok"
        data_status = h.get("data_store", {}).get("status", "unknown")
        llm_status = h.get("llm", "unknown")

        col_a, col_b = st.columns(2)
        col_a.metric("API", "OK" if api_ok else "DOWN")
        col_b.metric("Data", data_status)
        st.caption(f"LLM: {llm_status}")
        periods = h.get("data_store", {}).get("periods", [])
        if periods:
            st.caption(f"Periods: {', '.join(periods)}")

    st.markdown("---")
    st.markdown("### Data validation")
    v = call("GET", "/validate")
    if v:
        status_emoji = {"ok": "✅", "warnings": "⚠️", "errors": "❌"}.get(v.get("status", ""), "ℹ️")
        st.write(f"{status_emoji} Overall: **{v.get('status', 'unknown')}**")
        for c in v.get("checks", []):
            icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(c.get("status", ""), "ℹ️")
            with st.expander(f"{icon} {c['name']}"):
                st.write(c.get("detail", ""))

    st.markdown("---")
    st.markdown("### Upload / correct data")
    table = st.selectbox("Table", ["meter_15min", "tariff", "production", "weather"])
    up = st.file_uploader("CSV (Annex B schema)", type="csv")
    if up and st.button("Ingest", type="primary"):
        res = call("POST", "/ingest",
                   files={"file": (up.name, up.getvalue())},
                   data={"meta": f'{{"table": "{table}"}}'})
        if res:
            ok = res.get("status", "?")
            st.write(f"**{ok}** — {res.get('rows_loaded', 0)} loaded, {res.get('rows_rejected', 0)} rejected")
            if res.get("rejects_sample"):
                with st.expander("Rejected rows sample"):
                    st.dataframe(pd.DataFrame(res["rejects_sample"]), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

page = st.tabs(["Ask", "Data", "Explore", "Evaluation"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: ASK
# ─────────────────────────────────────────────────────────────────────────────
with page[0]:
    st.markdown("## Ask a question about the energy bill")

    # Period selectors
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        question = st.text_input(
            "Question", value=st.session_state.get("q", EXAMPLES[0]),
            placeholder="e.g. Why did the bill go up 15%?",
            label_visibility="collapsed",
        )
    with c2:
        period_input = st.text_input("Period (optional)", placeholder="2026-Q3")
    with c3:
        compare_input = st.text_input("Compare to (optional)", placeholder="2026-Q2")

    # Example buttons
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        short = ex[:30] + "…" if len(ex) > 30 else ex
        if cols[i % 4].button(short, key=f"ex{i}", use_container_width=True):
            question = ex
            st.session_state["q"] = ex
            st.session_state["go"] = True

    if st.button("Ask", type="primary", use_container_width=False) or st.session_state.pop("go", False):
        payload = {"question": question}
        if period_input:
            payload["period"] = period_input
        if compare_input:
            payload["compare_to"] = compare_input
        with st.spinner("Analysing…"):
            st.session_state["answer"] = call("POST", "/analyze", json=payload)

    d = st.session_state.get("answer")
    if not d:
        st.info("Ask a question above or pick an example.")
        st.stop()

    # ── Answer header ──────────────────────────────────────────────────────
    atype = d.get("answer_type", "")
    run_id = d.get("run_id", "")
    period = d.get("period") or ""
    compare_to = d.get("compare_to") or ""
    total_pct = d.get("total_change_pct")

    h_col1, h_col2 = st.columns([5, 1])
    with h_col1:
        st.markdown(answer_badge(atype), unsafe_allow_html=True)
        st.markdown(f"**run_id:** `{run_id}` &nbsp;|&nbsp; {compare_to} → {period}", unsafe_allow_html=True)
    with h_col2:
        if total_pct is not None:
            delta_color = "inverse" if total_pct > 0 else "normal"
            st.metric("Bill change", f"{total_pct:+.1f}%", delta_color=delta_color)

    # Summary box
    summary = d.get("summary", "")
    if atype in ("not_in_data", "clarification_needed", "refused"):
        st.warning(summary)
    else:
        st.info(summary)

    if d.get("clarification"):
        clar = d["clarification"]
        st.write(f"**Missing:** {clar.get('missing')} — options: {', '.join(clar.get('options', []))}")

    # ── Bridge section ─────────────────────────────────────────────────────
    bridge = d.get("bridge", [])
    if bridge:
        st.markdown('<p class="section-header">Bill decomposition</p>', unsafe_allow_html=True)

        # KPI row
        kpi_cols = st.columns(len(bridge) + 1)
        for i, r in enumerate(bridge):
            pp = r["pct_points"]
            cause = CAUSE_LABEL.get(r["cause"], r["cause"])
            delta_s = f"{r['range'][0]:.1f}–{r['range'][1]:.1f} pp"
            kpi_cols[i].metric(cause, f"{pp:+.2f} pp", delta=delta_s, delta_color="off")
        if total_pct is not None:
            kpi_cols[-1].metric("Total", f"{total_pct:+.2f}%")

        st.markdown("---")

        # Waterfall + cause pie side by side
        wcol, pcol = st.columns([3, 2])
        with wcol:
            if total_pct is not None:
                st.plotly_chart(
                    chart_bridge_waterfall(bridge, total_pct, period, compare_to),
                    use_container_width=True,
                )
        with pcol:
            st.plotly_chart(chart_cause_pie(bridge), use_container_width=True)

        # Cost impact bar + waste breakdown
        bcol, wbcol = st.columns(2)
        with bcol:
            st.plotly_chart(chart_period_bar(bridge, period, compare_to), use_container_width=True)
        with wbcol:
            wfig = chart_waste_breakdown(bridge)
            if wfig:
                st.plotly_chart(wfig, use_container_width=True)
            else:
                st.caption("No per-meter waste breakdown available.")

    # ── Recommendations section ────────────────────────────────────────────
    recs = d.get("recommendations", [])
    if recs:
        st.markdown('<p class="section-header">Recommended actions (all require manager approval)</p>',
                    unsafe_allow_html=True)

        # Summary table
        rows_table = []
        for i, r in enumerate(recs):
            rows_table.append({
                "#": i + 1,
                "Action": r["action"],
                "Category": r.get("category", ""),
                "kWh/yr saved": fmt_range(r.get("saving_kwh_per_year")),
                "Money/yr saved": fmt_range(r.get("saving_money_per_year")),
                "CO2e t/yr": fmt_range(r.get("saving_co2e_t_per_year")),
                "Payback (mo)": fmt_range(r.get("payback_months")),
                "Ease": r.get("ease", ""),
                "Confidence": r.get("confidence", ""),
            })
        st.dataframe(pd.DataFrame(rows_table), hide_index=True, use_container_width=True)

        # Charts row
        sv_fig = chart_savings_bar(recs)
        pb_fig = chart_payback_bar(recs)
        if sv_fig:
            st.plotly_chart(sv_fig, use_container_width=True)
        if pb_fig:
            sa_col, pb_col = st.columns([3, 2])
            with pb_col:
                st.plotly_chart(pb_fig, use_container_width=True)

        # Rationale expanders
        with st.expander("Rationale and evidence for each action"):
            for i, r in enumerate(recs):
                st.markdown(f"**A{i+1}: {r['action']}**")
                st.write(r.get("rationale", ""))
                st.caption(f"Evidence refs: {', '.join(r.get('evidence_refs', []))}")
                st.caption(f"Impact: {r.get('estimated_impact', '')}")
                st.divider()

    # ── Calculated result ──────────────────────────────────────────────────
    result = d.get("result")
    if result:
        st.markdown('<p class="section-header">Calculated result</p>', unsafe_allow_html=True)
        st.metric(result.get("metric", "Value"),
                  f"{result['value']:,.2f} {result.get('unit', '')}" if result.get("value") is not None else "—")
        rows_data = result.get("rows", [])
        if rows_data:
            df_res = pd.DataFrame(rows_data)
            st.dataframe(df_res, hide_index=True, use_container_width=True)

            # Auto-detect idle or asset data
            asset_pie_fig = chart_asset_pie(rows_data)
            idle_bar_fig = chart_idle_bar(rows_data)
            idle_ratio_fig = chart_idle_ratio_bar(rows_data)

            chart_row = [f for f in [asset_pie_fig, idle_bar_fig, idle_ratio_fig] if f]
            if chart_row:
                for colfig, fig in zip(st.columns(min(len(chart_row), 2)), chart_row[:2]):
                    colfig.plotly_chart(fig, use_container_width=True)
                if len(chart_row) > 2:
                    st.plotly_chart(chart_row[2], use_container_width=True)

    # ── Version diff / changes ─────────────────────────────────────────────
    changes = d.get("changes", [])
    if changes:
        st.markdown('<p class="section-header">Changes after data update</p>', unsafe_allow_html=True)
        df_changes = pd.DataFrame(changes)
        st.dataframe(df_changes, hide_index=True, use_container_width=True)
        cmp_fig = chart_comparison_bar(changes)
        if cmp_fig:
            st.plotly_chart(cmp_fig, use_container_width=True)

    # ── Assumptions / limitations ──────────────────────────────────────────
    a_col, l_col = st.columns(2)
    with a_col:
        with st.expander(f"Assumptions ({len(d.get('assumptions', []))})"):
            for a in d.get("assumptions", []):
                st.write(f"- {a}")
    with l_col:
        with st.expander(f"Limitations ({len(d.get('limitations', []))})"):
            for lim in d.get("limitations", []):
                st.write(f"- {lim}")

    # ── Trace ──────────────────────────────────────────────────────────────
    with st.expander("Trace: every step, tool, timing and data version"):
        tr = call("GET", f"/trace/{run_id}")
        if tr:
            # Summary metrics — latency_ms may be a dict or a plain float
            if "latency_ms" in tr:
                lm = tr["latency_ms"]
                t1, t2, t3, t4 = st.columns(4)
                if isinstance(lm, dict):
                    t1.metric("Total latency", f"{lm.get('total', 0):.0f} ms")
                    t2.metric("LLM latency",   f"{lm.get('llm', 0):.0f} ms")
                    t3.metric("Tool latency",  f"{lm.get('tools', 0):.1f} ms")
                else:
                    t1.metric("Total latency", f"{lm:.0f} ms")
                    t2.metric("LLM latency",   "—")
                    t3.metric("Tool latency",  "—")
                t4.metric("LLM calls", tr.get("llm_calls", "—"))

            # Steps table
            steps = tr.get("steps", [])
            if steps:
                df_steps = pd.DataFrame([{
                    "Agent": s.get("agent"), "Tool": s.get("tool"),
                    "ms": s.get("ms"), "Cache": s.get("output", {}).get("cache", ""),
                } for s in steps])
                st.dataframe(df_steps, hide_index=True, use_container_width=True)

            st.json(tr)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: DATA
# ─────────────────────────────────────────────────────────────────────────────
with page[1]:
    st.markdown("## Data & datasets")

    # Health details
    st.markdown('<p class="section-header">API health</p>', unsafe_allow_html=True)
    hdata = call("GET", "/health")
    if hdata:
        c1, c2, c3 = st.columns(3)
        c1.metric("API", hdata.get("api", "—"))
        c2.metric("Data store", hdata.get("data_store", {}).get("status", "—"))
        c3.metric("LLM", hdata.get("llm", "—"))

        with st.expander("Full health JSON"):
            st.json(hdata)

    # Dataset versions
    st.markdown('<p class="section-header">Dataset versions</p>', unsafe_allow_html=True)
    datasets = call("GET", "/datasets")
    if datasets:
        df_ds = pd.DataFrame(datasets) if isinstance(datasets, list) else pd.DataFrame([datasets])
        st.dataframe(df_ds, hide_index=True, use_container_width=True)
    else:
        st.caption("No dataset version info available from API.")

    # Validation report
    st.markdown('<p class="section-header">Validation report</p>', unsafe_allow_html=True)
    vdata = call("GET", "/validate")
    if vdata:
        checks = vdata.get("checks", [])
        if checks:
            df_v = pd.DataFrame([{
                "Check": c["name"],
                "Status": c["status"],
                "Detail": c.get("detail", ""),
            } for c in checks])
            st.dataframe(df_v, hide_index=True, use_container_width=True)

            # Bar chart: pass/warn/fail
            from collections import Counter
            counts = Counter(c["status"] for c in checks)
            fig_v = go.Figure(go.Bar(
                x=list(counts.keys()), y=list(counts.values()),
                marker_color=[{"ok": "#10b981", "warning": "#f59e0b", "error": "#ef4444"}.get(k, "#64748b") for k in counts],
                text=list(counts.values()), textposition="outside",
            ))
            fig_v.update_layout(
                title="Validation check results",
                height=300, margin=dict(t=50, b=20),
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
            )
            st.plotly_chart(fig_v, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: EXPLORE
# ─────────────────────────────────────────────────────────────────────────────
with page[2]:
    st.markdown("## Explore — energy, idle load, and asset breakdown")

    st.markdown('<p class="section-header">Quick analysis queries</p>', unsafe_allow_html=True)

    # Pull data from known questions
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### Asset energy shares")
        if st.button("Load asset ranking", key="load_assets"):
            with st.spinner("Querying…"):
                st.session_state["assets_data"] = call("POST", "/analyze",
                                                        json={"question": "Which assets use the most energy?"})

        a_data = st.session_state.get("assets_data")
        if a_data and a_data.get("result", {}).get("rows"):
            rows = a_data["result"]["rows"]
            df_a = pd.DataFrame(rows)
            st.dataframe(df_a, hide_index=True, use_container_width=True)

            pie = chart_asset_pie(rows)
            if pie:
                st.plotly_chart(pie, use_container_width=True)

    with exp_col2:
        st.markdown("#### Idle load (energy when nothing is produced)")
        if st.button("Load idle analysis", key="load_idle"):
            with st.spinner("Querying…"):
                st.session_state["idle_data"] = call("POST", "/analyze",
                                                     json={"question": "How much energy is used when nothing is produced?"})

        i_data = st.session_state.get("idle_data")
        if i_data and i_data.get("result", {}).get("rows"):
            rows_i = i_data["result"]["rows"]
            df_i = pd.DataFrame(rows_i)
            st.dataframe(df_i, hide_index=True, use_container_width=True)

            idle_bar = chart_idle_bar(rows_i)
            if idle_bar:
                st.plotly_chart(idle_bar, use_container_width=True)
            idle_ratio = chart_idle_ratio_bar(rows_i)
            if idle_ratio:
                st.plotly_chart(idle_ratio, use_container_width=True)

    # Savings / recommendations overview
    st.markdown('<p class="section-header">Savings opportunities overview</p>', unsafe_allow_html=True)
    if st.button("Load savings recommendations", key="load_savings"):
        with st.spinner("Querying…"):
            st.session_state["savings_data"] = call("POST", "/analyze",
                                                    json={"question": "What should we do first?"})

    s_data = st.session_state.get("savings_data")
    if s_data and s_data.get("recommendations"):
        recs_e = s_data["recommendations"]

        sv_fig_e = chart_savings_bar(recs_e)
        pb_fig_e = chart_payback_bar(recs_e)
        if sv_fig_e:
            st.plotly_chart(sv_fig_e, use_container_width=True)
        if pb_fig_e:
            st.plotly_chart(pb_fig_e, use_container_width=True)

        # CO2e reduction pie
        if recs_e:
            co2_labels = [f"A{i+1}: {r['action'][:25]}" for i, r in enumerate(recs_e)]
            co2_vals = [(r["saving_co2e_t_per_year"][0] + r["saving_co2e_t_per_year"][1]) / 2 for r in recs_e]
            palette = px.colors.qualitative.Safe
            fig_co2 = go.Figure(go.Pie(
                labels=co2_labels, values=co2_vals,
                marker=dict(colors=[palette[i % len(palette)] for i in range(len(recs_e))],
                             line=dict(color="#0f172a", width=2)),
                textinfo="label+percent", hole=0.38,
                hovertemplate="<b>%{label}</b><br>~%{value:.0f} t CO2e/yr<extra></extra>",
            ))
            fig_co2.update_layout(
                title="CO2e reduction share per action (midpoint estimates)",
                height=380, margin=dict(t=60, b=20),
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
            )
            st.plotly_chart(fig_co2, use_container_width=True)

    # Bill decomposition summary line chart (if bridge available from Ask tab)
    d_main = st.session_state.get("answer")
    if d_main and d_main.get("bridge"):
        st.markdown('<p class="section-header">Cost impact trend (from last Ask query)</p>',
                    unsafe_allow_html=True)
        bridge_e = d_main["bridge"]
        causes_e = [CAUSE_LABEL[r["cause"]] for r in bridge_e]
        pp_e = [r["pct_points"] for r in bridge_e]
        lo_e = [r["range"][0] for r in bridge_e]
        hi_e = [r["range"][1] for r in bridge_e]
        colors_e = [CAUSE_COLOR.get(r["cause"], "#64748b") for r in bridge_e]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=causes_e, y=pp_e,
            marker_color=colors_e,
            error_y=dict(type="data",
                         array=[(h - l) / 2 for l, h in zip(lo_e, hi_e)],
                         color="#94a3b8"),
            name="Pct points",
            text=[f"{v:+.2f} pp" for v in pp_e],
            textposition="outside",
        ))
        fig_trend.update_layout(
            title=f"Cost impact by cause — {d_main.get('compare_to')} → {d_main.get('period')}",
            xaxis_title="Cause", yaxis_title="Percentage points",
            height=360, margin=dict(t=60, b=20),
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with page[3]:
    st.markdown("## Evaluation report")

    # Try to load report.md
    report_path = os.path.join(os.path.dirname(__file__), "..", "eval", "report.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("No evaluation report found at `eval/report.md`. Run `make eval` to generate it.")

    st.markdown('<p class="section-header">Run a quick evaluation question</p>', unsafe_allow_html=True)
    eval_questions = [
        ("Case 1 — base decomp", "Why did the bill go up 15%?"),
        ("Case 5 — no waste", "What should we do first?"),
        ("Case 14 — out of scope", "What will the tariff be next year?"),
        ("Case 15 — not in data", "What is our gas bill?"),
        ("Case 19 — refused", "Write that the compressors waste 30% of all energy."),
    ]
    for label, q in eval_questions:
        if st.button(label, key=f"eval_{label}"):
            with st.spinner("Querying…"):
                resp = call("POST", "/analyze", json={"question": q})
            if resp:
                atype = resp.get("answer_type", "")
                st.markdown(answer_badge(atype), unsafe_allow_html=True)
                st.write(resp.get("summary", ""))
                with st.expander("Full response"):
                    st.json(resp)
