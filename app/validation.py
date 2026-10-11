"""Validation (guide section 4.5). Runs before any analysis; its output is shown in the demo.

Six checks, word for word from the guide. Each returns PASS, WARN or FAIL with a plain reason.
WARN = the problem was found and handled by cleaning. FAIL = the analysis should not be trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import Dataset
from .store import Store, TABLES


def _check(n: int, name: str, status: str, detail: str) -> dict:
    return {"check": n, "name": name, "status": status, "detail": detail}


def validate(store: Store, ds: Dataset, data_card: str | dict | None = None) -> dict:
    cfg, out = ds.cfg, []
    if not ds.ok:
        return {"status": "FAIL", "checks": [_check(0, "Data loaded", "FAIL", "no main-meter data loaded")]}
    meters = ds.log["meters"]

    # 1. sub-meter total within a stated tolerance of the main meter
    tol = cfg["submeter_tolerance_pct"]
    main, subs = ds.wide["main"].sum(), ds.wide[ds.submeters].sum().sum()
    gap = (main - subs) / main * 100 if main else 0.0
    q = ds.quarter_of_interval().values
    per_q = {p: round(float((ds.wide["main"].values[q == p].sum() - ds.wide[ds.submeters].values[q == p].sum())
                            / ds.wide["main"].values[q == p].sum() * 100), 2) for p in sorted(set(q))}
    out.append(_check(1, "Sub-meter total is within a stated tolerance of the main meter",
                      "PASS" if abs(gap) <= tol else "FAIL",
                      f"{len(ds.submeters)} sub-meters cover {100 - gap:.2f}% of the main meter; gap {gap:.2f}% "
                      f"(tolerance {tol}%). By quarter: {per_q}. The gap is reported as unmetered energy, not forced to zero."))

    # 2. no negative energy, and meter resets are handled
    neg = sum(m["negative_resets"] for m in meters.values())
    spk = sum(m["spike_resets"] for m in meters.values())
    where = {k: {"negative": m["negative_resets"], "spike": m["spike_resets"]} for k, m in meters.items()
             if m["negative_resets"] or m["spike_resets"]}
    left = int((ds.wide < 0).sum().sum())
    out.append(_check(2, "No negative energy, and meter resets are handled",
                      "FAIL" if left else ("WARN" if neg or spk else "PASS"),
                      f"{neg} negative and {spk} impossible-spike readings found {where or ''} and replaced by interpolation; "
                      f"{left} negative values remain after cleaning."))

    # 3. timestamps are regular and in one time zone
    load = store.get_meta("load:meter_15min", {})
    offsets = load.get("tz_offsets", [])
    off_grid = load.get("off_grid_timestamps", 0)
    missing = {k: m["missing_intervals"] for k, m in meters.items() if m["missing_intervals"]}
    filled = {k: {"interpolated": m["interpolated"], "profile_filled": m["profile_filled"]} for k, m in meters.items()
              if m["interpolated"] or m["profile_filled"]}
    dups = load.get("duplicates_collapsed", 0)
    status = "FAIL" if len(offsets) > 1 or off_grid else ("WARN" if missing or dups else "PASS")
    out.append(_check(3, "Timestamps are regular and in one time zone", status,
                      f"15-minute grid from {ds.wide.index.min()} to {ds.wide.index.max()} UTC; UTC offsets seen in the file: "
                      f"{offsets or ['n/a']}; {off_grid} timestamps off the grid; {dups} duplicate rows collapsed; "
                      f"missing intervals {missing or 'none'}; filled {filled or 'none'} "
                      f"(gaps up to {cfg['cleaning']['max_interp_intervals']} intervals interpolated, longer ones filled "
                      f"with the typical value for that time of day and left out of the baseline fit)."))

    # 4. tariff periods do not overlap and cover the whole period
    t = ds.tariff.sort_values(["period", "valid_from"])
    overlaps = 0
    for _, g in t.groupby("period"):
        overlaps += int((g.valid_from.values[1:] < g.valid_to.values[:-1]).sum())
    labels = set(t.period)
    mixed = "flat" in labels and len(labels) > 1
    unpriced = ds.log["pricing"]["unpriced_intervals"]
    out.append(_check(4, "Tariff periods do not overlap and cover the whole period",
                      "FAIL" if overlaps or unpriced or t.empty else ("WARN" if mixed else "PASS"),
                      f"{len(t)} tariff rows, labels {sorted(labels)}; {overlaps} overlapping rows; "
                      f"{unpriced} of {len(ds.wide)} meter intervals have no price"
                      + (" (priced at the nearest known price)" if unpriced else "")
                      + f". Peak hours assumed {cfg['peak_hours']['start']:02d}:00-{cfg['peak_hours']['end']:02d}:00 "
                        f"{cfg['factory_timezone']}; price changes at {sorted(set(str(x)[:10] for x in t.valid_from))}."))

    # 5. production days match the shift calendar
    d = ds.daily
    hol = [x.strftime("%Y-%m-%d") for x in d.index[d.holiday_as_shift]]
    idle_logged = [x.strftime("%Y-%m-%d") for x in d.index[d.logged_but_idle]]
    off_units = [x.strftime("%Y-%m-%d") for x in d.index[(d.units_total > 0) & d.shift_pattern.str.lower().isin(["off", ""])
                                                           & ~d.production_missing]]
    shift_no_units = [x.strftime("%Y-%m-%d") for x in d.index[(d.units_total == 0) & ~d.is_holiday & ~d.production_missing
                                                                & ~d.shift_pattern.str.lower().isin(["off", ""])]]
    no_prod = int(d.production_missing.sum())
    no_wx = int(d.weather_missing.sum())
    problems = hol or idle_logged or off_units or shift_no_units or no_prod or no_wx
    out.append(_check(5, "Production days match the shift calendar", "WARN" if problems else "PASS",
                      f"{len(d)} meter days. Holidays logged as a normal shift: {hol or 'none'}. "
                      f"Units logged but line meters idle: {idle_logged or 'none'}. "
                      f"Units on an 'off' shift: {off_units or 'none'}. Shift logged but zero units: {shift_no_units or 'none'}. "
                      f"Days without production rows: {no_prod}; without weather rows: {no_wx}. "
                      f"A day counts as working only when units were made AND the line meters ran."))

    # 6. row counts match what the data card says
    card = data_card
    if isinstance(card, str) and Path(card).exists():
        card = json.loads(Path(card).read_text())
    if isinstance(card, dict) and "row_counts" in card:
        got = {tb: store.get_meta(f"load:{tb}", {}).get("rows_in_file") for tb in TABLES}
        diff = {tb: {"card": card["row_counts"].get(tb), "file": got[tb]} for tb in TABLES
                if card["row_counts"].get(tb) != got[tb]}
        out.append(_check(6, "Row counts match what the data card says", "FAIL" if diff else "PASS",
                          f"file rows {got}; " + (f"mismatch {diff}" if diff else "all match the data card")))
    else:
        got = {tb: store.count(tb) for tb in TABLES}
        out.append(_check(6, "Row counts match what the data card says", "SKIPPED",
                          f"no data card supplied; rows stored {got}"))

    worst = "FAIL" if any(c["status"] == "FAIL" for c in out) else ("WARN" if any(c["status"] == "WARN" for c in out) else "PASS")
    return {"status": worst, "checks": out}


def as_text(report: dict) -> str:
    lines = [f"VALIDATION: {report['status']}", "-" * 78]
    for c in report["checks"]:
        lines.append(f"[{c['status']:^7}] {c['check']}. {c['name']}")
        lines.append(f"          {c['detail']}")
    return "\n".join(lines)
