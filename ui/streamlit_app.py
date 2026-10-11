"""Minimal UI: validation output, one bill-bridge chart, action table, question box, trace.
Talks to the API over HTTP only, so the UI shows exactly what a judge's curl would get."""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
LABEL = {"tariff": "Tariff (price)", "volume": "Volume", "product_mix": "Product mix", "weather": "Weather", "waste": "Unexplained / waste"}
EXAMPLES = ["Why did the bill go up 15%?", "Which assets use the most energy?", "How much energy is used when nothing is produced?",
            "What should we do first?", "Which action pays back fastest?", "What if the tariff had not changed?",
            "What will the tariff be next year?", "What changed after the correction?"]

st.set_page_config(page_title="Factory energy bill", layout="wide")
st.title("Factory energy: why the bill changed, and what to fix")


def call(method: str, path: str, **kw):
    try:
        r = requests.request(method, API + path, timeout=120, **kw)
        return r.json()
    except Exception as e:
        st.error(f"API not reachable at {API}: {e}")
        return None


def bridge_chart(bridge: list, total: float, period: str, ref: str):
    rows = sorted(bridge, key=lambda r: (r["cause"] == "waste", -r["pct_points"]))
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(rows) + ["total"],
        x=[LABEL[r["cause"]] for r in rows] + ["Total change"],
        y=[r["pct_points"] for r in rows] + [total],
        text=[f"{r['pct_points']:+.2f}<br>({r['range'][0]:.1f} to {r['range'][1]:.1f})" for r in rows] + [f"{total:+.2f}%"],
        textposition="outside", connector={"line": {"color": "#9aa0a6", "width": 1}},
        increasing={"marker": {"color": "#c2410c"}}, decreasing={"marker": {"color": "#15803d"}}, totals={"marker": {"color": "#334155"}}))
    fig.update_layout(title=f"Bill change {ref} to {period}: {total:+.2f}% (percentage points by cause, range in brackets)",
                      yaxis_title="Percentage points of the reference bill", showlegend=False, height=430, margin=dict(t=60, b=20))
    return fig


with st.sidebar:
    st.subheader("System")
    h = call("GET", "/health")
    if h:
        st.write(f"API **{h['api']}** · data **{h['data_store'].get('status')}**")
        st.caption(f"LLM: {h['llm']}")
        st.caption(f"Quarters loaded: {', '.join(h['data_store'].get('periods', [])) or 'none'}")
    st.subheader("Data validation")
    v = call("GET", "/validate")
    if v:
        st.write(f"Overall: **{v['status']}**")
        for c in v["checks"]:
            with st.expander(f"{c['status']} · {c['name']}"):
                st.write(c["detail"])
    st.subheader("Add or correct data")
    table = st.selectbox("Table", ["meter_15min", "tariff", "production", "weather"])
    up = st.file_uploader("CSV in the Annex B schema", type="csv")
    if up and st.button("Ingest"):
        res = call("POST", "/ingest", files={"file": (up.name, up.getvalue())}, data={"table": table})
        if res:
            st.write(f"{res['status']}: {res['rows_loaded']} loaded, {res['rows_rejected']} rejected")
            if res.get("rerun"):
                st.caption("Bill split before and after this file")
                st.dataframe(pd.DataFrame(res["rerun"]["bridge"]), hide_index=True)

question = st.text_input("Ask a question", value=st.session_state.get("q", EXAMPLES[0]))
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
        question = ex
        st.session_state["q"] = ex
        st.session_state["go"] = True

if st.button("Ask", type="primary") or st.session_state.pop("go", False):
    st.session_state["answer"] = call("POST", "/analyze", json={"question": question})

d = st.session_state.get("answer")
if d:
    st.caption(f"answer_type: {d['answer_type']} · run_id: {d['run_id']} · {d.get('compare_to') or ''} to {d.get('period') or ''}")
    (st.warning if d["answer_type"] in ("not_in_data", "clarification_needed", "refused") else st.info)(d["summary"])
    if d["bridge"]:
        total = round(sum(r["pct_points"] for r in d["bridge"]), 2)
        st.plotly_chart(bridge_chart(d["bridge"], total, d["period"], d["compare_to"]), width="stretch")
    if d["recommendations"]:
        st.subheader("Actions (suggestions; a manager approves)")
        fmt = lambda r: f"{r[0]:,.0f} to {r[1]:,.0f}" if r else "n/a"
        st.dataframe(pd.DataFrame([{
            "Rank": r["rank"], "Action": r["action"], "kWh / year": fmt(r["saving_kwh_per_year"]),
            "Money / year": fmt(r["saving_money_per_year"]),
            "t CO2e / year": f"{r['saving_co2e_t_per_year'][0]} to {r['saving_co2e_t_per_year'][1]}",
            "Payback (months)": f"{r['payback_months'][0]} to {r['payback_months'][1]}" if r["payback_months"] else "n/a",
            "Ease": r["ease"], "Confidence": r["confidence"], "Evidence": r["rationale"]} for r in d["recommendations"]]), hide_index=True)
    data = d.get("data") or {}
    if data.get("assets"):
        st.subheader("By sub-meter")
        st.dataframe(pd.DataFrame(data["assets"]), hide_index=True)
    if data.get("comparison"):
        st.subheader("Before and after the data update")
        st.dataframe(pd.DataFrame(data["comparison"]["rows"]), hide_index=True)
    left, right = st.columns(2)
    with left.expander("Assumptions"):
        for a in d["assumptions"]:
            st.write("- " + a)
    with right.expander("Limitations"):
        for a in d["limitations"]:
            st.write("- " + a)
    with st.expander("Trace: steps, tools and data behind this answer"):
        st.json(call("GET", f"/trace/{d['run_id']}"))
