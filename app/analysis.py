"""Runs the code tools for one labelled question and returns facts. No LLM in this file.

Every handler returns (answer_type, facts, steps). `facts` is the only thing the brief
writer sees, and the only source of numbers the number check accepts.
"""
from __future__ import annotations

import re

import pandas as pd

from .cleaning import Dataset, available_periods, parse_period
from .intents import find_meter
from .tools.assets import asset_breakdown, idle_load, period_mask
from .tools.billing import bill, is_peak, prices_as_of
from .tools.bridge import decompose
from .tools.savings import recommend
from .tools.waste import cross_check, find_waste


def base_assumptions(ds: Dataset) -> list[str]:
    c = ds.cfg
    return [
        "The main meter is the billed total; sub-meters are used for attribution only",
        f"Peak price applies {c['peak_hours']['start']:02d}:00-{c['peak_hours']['end']:02d}:00 local time; the tariff file names the periods but not their hours",
        f"Factory time zone {c['factory_timezone']} is used to match UTC meter readings to daily production and weather",
        f"Idle means the production lines are not running ({ds.log.get('idle_basis', 'n/a')})",
    ]


def resolve_periods(ds: Dataset, question: str, period: str | None, compare_to: str | None) -> dict:
    """Which quarters to compare. Never guesses: unknown text asks back, unloaded quarters say not_in_data."""
    have = available_periods(ds)
    if len(have) == 0:
        return {"exit": "not_in_data", "message": "No usable meter data is loaded yet. Load data with POST /ingest or scripts/load_data.py."}
    asked = {}
    for name, text in (("period", period), ("compare_to", compare_to)):
        if text:
            p = parse_period(text)
            if not p:
                return {"exit": "clarification_needed",
                        "message": f"I could not read '{text}' as a quarter. Use a label like {have[-1]}. Loaded quarters: {', '.join(have)}."}
            asked[name] = p
    in_q = re.findall(r"(?:20\d{2}\s*[-_/ ]?\s*Q\s*[1-4]|Q\s*[1-4]\s*[-_/ ]?\s*20\d{2})", question, re.I)
    named = [parse_period(x) for x in in_q]
    for bare in re.findall(r"(?<![\w-])Q([1-4])\b(?!\s*[-_/ ]?\s*20\d{2})", question, re.I):
        match = [h for h in have if h.endswith(f"Q{bare}")]
        named.append(match[0] if len(match) == 1 else f"Q{bare}")
    named = list(dict.fromkeys(named))
    if "period" not in asked and named:
        asked["period"] = max(named) if len(named) > 1 else named[0]
        if len(named) > 1 and "compare_to" not in asked:
            asked["compare_to"] = min(named)
    for p in asked.values():
        if p not in have:
            return {"exit": "not_in_data",
                    "message": f"{p} is not in the loaded data. Loaded quarters: {', '.join(have)}. I will not estimate a quarter that is not loaded."}
    new = asked.get("period", have[-1])
    if "compare_to" in asked:
        ref = asked["compare_to"]
    else:
        earlier = [h for h in have if h < new]
        ref = earlier[-1] if earlier else None
    if ref == new:
        return {"exit": "clarification_needed", "message": f"Both periods are {new}. Which quarter should it be compared with? Loaded: {', '.join(have)}."}
    note = [] if period and compare_to else [f"Periods used: {new} compared with {ref} (latest loaded quarter against the one before it, unless the request names them)"]
    return {"new": new, "ref": ref, "available": have, "note": note}


def _step(tool: str, output: dict) -> dict:
    return {"agent": "analyst", "tool": tool, "output": output}


def _bridge_bundle(ds, cfg, new, ref, price=None, scenario=None):
    """Bridge + waste evidence + cross-check + actions: the full chain behind any 'why' or 'what to do' answer."""
    b = decompose(ds, cfg, new, ref, price, scenario)
    if not b.get("ok"):
        return b, None, None, None, []
    w = find_waste(ds, cfg, new, ref)
    chk = cross_check(b, w)
    rec = recommend(ds, cfg, w, chk)
    steps = [
        _step("build_baseline", {"r2": b["baseline"]["r2"], "days_used": b["baseline"]["days_used"],
                                 "kwh_per_unit": b["baseline"]["kwh_per_unit"]}),
        _step("decompose_change", {**{r["cause"]: r["pct_points"] for r in b["bridge"]}, "total": b["bill"]["change_pct"]}),
        _step("find_idle_load", {"flagged": [{k: f.get(k) for k in ("meter", "kind", "excess_kw", "onset")} for f in w["findings"]],
                                 "cleared": [c["meter"] + ": " + c["reason"] for c in w["meters_checked"] if not c["flagged"]]}),
        _step("cross_check_waste", chk),
        _step("estimate_savings", {"actions": [{"action": r["action"], "kwh": r["saving_kwh_per_year"],
                                                "payback_months": r["payback_months"]} for r in rec["recommendations"]]}),
    ]
    return b, w, chk, rec, steps


def _bridge_facts(b, w, chk) -> dict:
    waste_pp = next(r["pct_points"] for r in b["bridge"] if r["cause"] == "waste")
    return {"period": b["period"], "compare_to": b["compare_to"], "bill": b["bill"], "bridge": b["bridge"],
            "explained_pct_points": round(b["total_pct_points"] - waste_pp, 2), "baseline": b["baseline"],
            "inputs": b["inputs"], "waste_check": chk,
            "evidence": [f for f in w["findings"] if f["kind"] == "idle_growth"],
            "meters_cleared": [{"meter": c["meter"], "reason": c["reason"]} for c in w["meters_checked"] if not c["flagged"]]}


def _waste_limits(chk: dict) -> list[str]:
    out = ["The waste slice is what the baseline cannot explain, so it also contains baseline error; see its range"]
    if chk["status"] == "supported_by_meters":
        out.append("Meter evidence covers idle hours only; whether the excess also runs during production is assumed in the high end of the savings range")
    return out


def run(intent: str, params: dict, ds: Dataset, cfg: dict, store, new: str, ref: str | None, question: str) -> dict:
    """Returns {answer_type, facts, steps, bridge, recommendations, assumptions, limitations} or an early exit."""
    out = {"bridge": [], "recommendations": [], "assumptions": base_assumptions(ds), "limitations": [], "steps": []}
    meter = find_meter(question, ds.submeters)
    if intent in ("top_users", "idle_waste") and not meter and re.search(r"\b(this|that|the) (asset|machine|meter|equipment)\b", question, re.I):
        return {"exit": "clarification_needed", "message": f"Which asset do you mean? Meters in the data: {', '.join(ds.submeters)}."}

    need_ref = intent not in ("top_users", "idle_waste", "bill_total")
    if ref is None and need_ref:
        return {"exit": "not_in_data", "message": f"Only {new} is loaded, so there is no earlier quarter to compare with."}

    if intent == "top_users":
        a = asset_breakdown(ds, new)
        facts = {**a, "focus": next((x for x in a["assets"] if x["meter"] == meter), None)}
        out.update(answer_type="calculated", facts=facts, steps=[_step("asset_breakdown", {
            "top": [(x["meter"], x["share_of_main_pct"]) for x in a["assets"][:3]], "unmetered_pct": a["unmetered_pct_of_main"]})])
        if not a["within_tolerance"]:
            out["limitations"].append(f"Sub-meters differ from the main meter by {a['unmetered_pct_of_main']}%, above the {a['tolerance_pct']}% tolerance")
        return out

    if intent == "idle_waste":
        i = idle_load(ds, new)
        w = find_waste(ds, cfg, new, ref) if ref else {"findings": [], "meters_checked": []}
        facts = {**i, "findings": w["findings"], "focus": next((x for x in i["assets"] if x["meter"] == meter), None),
                 "meters_cleared": [{"meter": c["meter"], "reason": c["reason"]} for c in w["meters_checked"] if not c["flagged"]]}
        out.update(answer_type="calculated", facts=facts, steps=[
            _step("find_idle_load", {"main_idle_kwh": i["main_idle_kwh"], "idle_hours": i["idle_hours"],
                                     "top": [(x["meter"], x["idle_kwh"]) for x in i["assets"][:3]]}),
            _step("find_waste_patterns", {"flagged": [(f["meter"], f["kind"], f["excess_kw"]) for f in w["findings"]]})])
        out["limitations"].append("Idle energy includes loads that must stay on (for example a furnace holding temperature); only flagged meters are treated as waste")
        return out

    if intent == "bill_total":
        r = lambda b: {k: round(v, 4 if k == "avg_price" else 2) for k, v in b.items() if k != "period"}
        if ref is None:
            facts = {"period": new, "compare_to": None, "bill": {"new": r(bill(ds, new))}}
        else:
            b = decompose(ds, cfg, new, ref)
            if not b.get("ok"):
                return {"exit": "not_in_data", "message": b["reason"]}
            facts = {"period": new, "compare_to": ref, "bill": b["bill"]}
        out.update(answer_type="calculated", facts=facts,
                   steps=[_step("price_intervals", {"bill": facts["bill"]["new"]["bill"], "kwh": facts["bill"]["new"]["kwh"]})])
        return out

    # everything below needs the full chain
    price, scenario = None, None
    if intent == "what_if":
        if params.get("tariff_unchanged"):
            end_ref = ds.wide.index[period_mask(ds, ref)].max()
            old = prices_as_of(end_ref, ds.tariff)
            peak = is_peak(ds.local, cfg)
            vals = [old.get("flat", old.get("peak" if pk else "off_peak")) for pk in peak]
            price = pd.Series(vals, index=ds.wide.index, dtype=float).fillna(ds.price)
            scenario = f"tariff stays at the {ref} prices for all of {new}"
        elif "tariff_change_pct" in params:
            pct = params["tariff_change_pct"]
            price = ds.price * (1 + pct / 100)
            scenario = f"every {new} price is {abs(pct):g}% {'lower' if pct < 0 else 'higher'} than actual"
        else:
            return {"exit": "clarification_needed",
                    "message": "Which what-if do you mean? Supported: a tariff that is N% lower or higher, or a tariff that had not changed. "
                               "For the effect of fixing waste, ask what we should do first."}

    b, w, chk, rec, steps = _bridge_bundle(ds, cfg, new, ref)
    if not b.get("ok"):
        return {"exit": "not_in_data", "message": b["reason"]}
    out["limitations"] += b["limitations"] + _waste_limits(chk)
    cleaned = {m: s for m, s in ds.log["meters"].items() if s["profile_filled"] or s["interpolated"]}
    if "main" in cleaned:
        out["limitations"].append(f"Main-meter gaps were filled ({cleaned['main']['interpolated']} intervals interpolated, "
                                  f"{cleaned['main']['profile_filled']} from the typical daily profile)")
    d = ds.daily[ds.daily.quarter.isin([new, ref])]
    if d.weather_missing.any() or d.production_missing.any():
        out["limitations"].append(f"{int(d.weather_missing.sum())} day(s) have no weather row (nearest value used) and "
                                  f"{int(d.production_missing.sum())} day(s) have no production row (treated as no production)")
    out["assumptions"] += rec["assumptions"]
    out["recommendations"] = rec["recommendations"]
    facts = _bridge_facts(b, w, chk)

    if intent in ("savings", "ranking"):
        facts = {"period": new, "compare_to": ref, "recommendations": rec["recommendations"], "ranking_rule": rec["ranking_rule"],
                 "prices_used": rec["prices_used"], "waste_check": chk, "bill_change_pct": b["bill"]["change_pct"]}
        out.update(answer_type="recommendation", facts=facts, steps=steps, bridge=b["bridge"])
        return out

    if intent == "what_if":
        s = decompose(ds, cfg, new, ref, price, scenario)
        ws = find_waste(ds, cfg, new, ref)
        facts = _bridge_facts(s, ws, cross_check(s, ws))
        t = lambda x: next(r["pct_points"] for r in x["bridge"] if r["cause"] == "tariff")
        facts["what_if"] = {"scenario": scenario, "bill_actual": b["bill"]["new"]["bill"], "bill_scenario": s["bill"]["new"]["bill"],
                            "bill_difference": round(s["bill"]["new"]["bill"] - b["bill"]["new"]["bill"], 2),
                            "change_pct_actual": b["bill"]["change_pct"], "change_pct_scenario": s["bill"]["change_pct"],
                            "tariff_points_actual": t(b), "tariff_points_scenario": t(s)}
        steps.insert(2, _step("reprice_scenario", facts["what_if"]))
        out["assumptions"].append("What-if: only prices change; energy use is kept exactly as measured")
        out.update(answer_type="decomposition", facts=facts, steps=steps, bridge=s["bridge"])
        return out

    if intent == "data_change":
        prev = store.previous_bridge(new, ref, b)
        if not prev:
            return {"exit": "not_in_data", "message": "There is no earlier result to compare with: no data has been corrected or added "
                                                      f"since the first analysis of {new} against {ref}."}
        before = {r["cause"]: r["pct_points"] for r in prev["bridge"]}
        rows = [{"cause": r["cause"], "before": before.get(r["cause"]), "after": r["pct_points"],
                 "delta": round(r["pct_points"] - before.get(r["cause"], 0), 2)} for r in b["bridge"]]
        facts["comparison"] = {"rows": rows, "total_before": prev["bill"]["change_pct"], "total_after": b["bill"]["change_pct"],
                               "bill_new_before": prev["bill"]["new"]["bill"], "bill_new_after": b["bill"]["new"]["bill"],
                               "what_was_loaded": store.ingests_since(prev["data_version"])}
        steps.append(_step("compare_with_previous_run", facts["comparison"]))

    out.update(answer_type="decomposition", facts=facts, steps=steps, bridge=b["bridge"])
    out["_bridge_result"] = b
    return out
