"""Finance brief (R10). The LLM words the brief from the facts it is handed; a fixed
template does the same job in mock mode and whenever the LLM brief fails the number check."""
from __future__ import annotations

import json

LABEL = {"tariff": "tariff (price)", "volume": "production volume", "product_mix": "product mix",
         "weather": "weather", "waste": "unexplained energy"}

SYSTEM = """You write a short brief for a factory finance manager.
Rules you must follow:
1. Use ONLY the numbers in the FACTS JSON, copied exactly as written. Do not add, round, convert, combine or estimate numbers.
2. Never use a number that appears only in the question.
3. Explain price, production volume, product mix and weather BEFORE mentioning waste. Do not blame a machine unless FACTS names meter evidence.
4. Savings are always a low-to-high range. Actions are suggestions that need a manager's approval.
5. Plain sentences, at most 120 words, no bullet points, no numbered lists, no headings."""


def llm_brief(llm, question: str, facts: dict, bad_numbers: list | None = None) -> tuple[str, int]:
    user = f"QUESTION: {question}\n\nFACTS:\n{json.dumps(facts, default=str)}"
    if bad_numbers:
        user += (f"\n\nYour previous draft used numbers that are not in FACTS: {bad_numbers}. "
                 f"Rewrite it using only numbers from FACTS.")
    text, tokens = llm.complete(SYSTEM, user)
    return text.strip(), tokens


def _n(x) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}"


def _rng(r) -> str:
    return f"{_n(r[0])} to {_n(r[1])}"


def template_brief(intent: str, f: dict) -> str:
    """Deterministic brief built directly from the facts."""
    if intent in ("bill_bridge", "what_if", "data_change"):
        b, rows = f["bill"], {r["cause"]: r for r in f["bridge"]}
        verb = "rose" if b["change_pct"] >= 0 else "fell"
        if intent == "what_if":
            verb = "would have risen" if b["change_pct"] >= 0 else "would have fallen"
        others = sorted([c for c in rows if c != "waste"], key=lambda c: -abs(rows[c]["pct_points"]))
        parts = ", ".join(f"{LABEL[c]} {rows[c]['pct_points']} points" for c in others)
        w = rows["waste"]
        text = (f"The bill {verb} {abs(b['change_pct'])}% from {f['compare_to']} to {f['period']} "
                f"({_n(b['reference']['bill'])} to {_n(b['new']['bill'])}). Causes, in percentage points: {parts}. "
                f"Together these explain {f['explained_pct_points']} of the {b['change_pct']} points and are not machine faults. ")
        chk = f["waste_check"]
        if chk["status"] == "supported_by_meters":
            ev = f["evidence"][0]
            text += (f"The remaining {w['pct_points']} points (range {_rng(w['range'])}) are energy the baseline cannot explain, "
                     f"and the meters back this up: idle-hour load on {ev['meter']} is {ev['excess_kw']} kW above its "
                     f"reference since {ev['onset']}. The meter-backed part is controllable waste.")
        elif chk["status"] == "no_meter_evidence":
            text += (f"The remaining {w['pct_points']} points (range {_rng(w['range'])}) are unexplained. No sub-meter shows a "
                     f"matching idle-hour pattern, so we report it as unexplained and do not blame a machine.")
        elif chk["status"] == "meter_pattern_without_bill_leftover":
            text += (f"The baseline leaves {w['pct_points']} points unexplained, so waste did not add to the increase, "
                     f"although {', '.join(chk['meters'])} shows higher idle-hour load worth acting on.")
        else:
            text += (f"The remaining {w['pct_points']} points (range {_rng(w['range'])}) are within the baseline's uncertainty; "
                     f"no meter shows new waste.")
        if intent == "what_if":
            s = f["what_if"]
            text = (f"Scenario: {s['scenario']}. The {f['period']} bill would be {_n(s['bill_scenario'])} instead of "
                    f"{_n(s['bill_actual'])}, a change of {s['change_pct_scenario']}% against {f['compare_to']} instead of "
                    f"{s['change_pct_actual']}%. The tariff slice would be {s['tariff_points_scenario']} points instead of "
                    f"{s['tariff_points_actual']}; the other causes are unchanged because they are valued at the reference price. ") + text
        if intent == "data_change":
            c = f["comparison"]
            moved = "; ".join(f"{LABEL[r['cause']]} {r['before']} to {r['after']}" for r in c["rows"] if r["delta"] != 0)
            text = (f"After the data update the bill change is {c['total_after']}% (it was {c['total_before']}%). "
                    f"Slices that moved, in points: {moved or 'none'}. Loaded since the earlier result: {c['what_was_loaded']}. "
                    f"Sub-meter corrections change asset shares and waste evidence but not the bill, which uses the main meter. ") + text
        return text

    if intent == "bill_total":
        n = f["bill"]["new"]
        text = f"In {f['period']} the bill was {_n(n['bill'])} for {_n(n['kwh'])} kWh, an average price of {n['avg_price']} per kWh."
        if f.get("compare_to"):
            r = f["bill"]["reference"]
            text += (f" In {f['compare_to']} it was {_n(r['bill'])} for {_n(r['kwh'])} kWh at {r['avg_price']} per kWh. "
                     f"The bill changed by {f['bill']['change_pct']}% and energy by {f['bill']['kwh_change_pct']}%.")
        return text

    if intent == "top_users":
        a = f["assets"]
        top = "; ".join(f"{x['meter']} {_n(x['kwh'])} kWh ({x['share_of_main_pct']}%)" for x in a[:3])
        text = (f"In {f['period']} the main meter recorded {_n(f['main_kwh'])} kWh. The biggest users were: {top}. "
                f"Sub-meters do not add up to the main meter: {f['unmetered_pct_of_main']}% is unmetered "
                f"(tolerance {f['tolerance_pct']}%). Using the most energy is not the same as wasting it.")
        if f.get("focus"):
            x = f["focus"]
            text = f"{x['meter']} used {_n(x['kwh'])} kWh in {f['period']}, {x['share_of_main_pct']}% of the main meter (rank {x['rank']}). " + text
        return text

    if intent == "idle_waste":
        a = f["assets"]
        top = "; ".join(f"{x['meter']} {_n(x['idle_kwh'])} kWh" for x in a[:3])
        text = (f"In {f['period']} the factory used {_n(f['main_idle_kwh'])} kWh ({f['main_idle_share_pct']}% of the total) "
                f"while the production lines were stopped, over {_n(f['idle_hours'])} idle hours: {f['idle_window_on_working_days']} "
                f"on working days plus {f['non_working_days']} non-working days. Largest idle users: {top}. "
                f"Not all idle energy is waste. ")
        flagged = f.get("findings", [])
        if flagged:
            text += "Flagged: " + "; ".join(
                (f"{x['meter']} idle load is {x['excess_kw']} kW above its weather-adjusted reference since {x['onset']}"
                 if x["kind"] == "idle_growth" else
                 f"{x['meter']} stays at {x['idle_vs_working_pct']}% of its working load when idle") for x in flagged) + "."
        else:
            text += "No meter shows idle load above its weather-adjusted reference."
        if f.get("focus"):
            x = f["focus"]
            text = (f"{x['meter']} used {_n(x['idle_kwh'])} kWh in idle hours, averaging {x['avg_idle_kw']} kW idle "
                    f"against {x['avg_working_kw']} kW when working. ") + text
        return text

    if intent in ("savings", "ranking"):
        recs = f["recommendations"]
        if not recs:
            return (f"No controllable waste pattern was found on any meter in {f['period']}, so no action is recommended. "
                    f"The bill change is explained by price, production and weather.")
        lines = []
        for r in recs:
            pb = f"payback {_rng(r['payback_months'])} months" if r["payback_months"] else "payback not computable"
            lines.append(f"{r['rank']}) {r['action']}: saves {_rng(r['saving_kwh_per_year'])} kWh, "
                         f"{_rng(r['saving_money_per_year'])} in money and {_rng(r['saving_co2e_t_per_year'])} t CO2e per year; "
                         f"{pb}; ease {r['ease']}; confidence {r['confidence']}.")
        head = ("Ranked by payback, then ease. " if intent == "ranking" else "Recommended actions, fastest payback first. ")
        return (head + " ".join(lines) + " Costs and recoverable shares are stated assumptions; each action needs a manager's approval.")
    return ""
