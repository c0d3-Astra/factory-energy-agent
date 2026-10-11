"""Bill decomposition (R3). Order from Annex A: tariff, volume, product mix, weather, then waste.

With B = bill, E = kWh, p0 = reference average price (B0 / E0):

    B1 - B0 = (B1 - E1 x p0)          tariff slice: new energy at new prices minus new energy at the old price
            + (E1 - E0) x p0          energy change at the old price, split into:
                volume   = calendar days + extra units at the reference mix
                mix      = this quarter's units at this quarter's mix minus the same units at the reference mix
                weather  = change in degree-days
                waste    = whatever the baseline cannot explain (growth in unexplained energy)

Each slice is divided by B0, so the five percentage points sum to the total change by construction.
"""
from __future__ import annotations

import numpy as np

from .baseline import bootstrap_betas, fit_baseline, predict_total, totals
from .billing import bill

CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]


def energy_slices(beta: dict, t0: dict, t1: dict, products: list[str], use_mix: bool, mix_first: bool = False) -> dict:
    """kWh explained by volume, mix and weather when inputs move from reference (t0) to new (t1)."""
    calendar = beta["const"] * (t1["const"] - t0["const"]) + beta["working"] * (t1["working"] - t0["working"])
    V0, V1 = t0["units_total"], t1["units_total"]
    if use_mix and V0 > 0 and V1 > 0:
        r = np.array([beta[f"units_{p}"] for p in products])
        s0 = np.array([t0[f"units_{p}"] for p in products]) / V0
        s1 = np.array([t1[f"units_{p}"] for p in products]) / V1
        if mix_first:
            mix, volume = V0 * float(r @ (s1 - s0)), (V1 - V0) * float(r @ s1)
        else:
            volume, mix = (V1 - V0) * float(r @ s0), V1 * float(r @ (s1 - s0))
    elif use_mix:    # one quarter made nothing: shares are undefined, so all of it is volume
        volume, mix = sum(beta[f"units_{p}"] * (t1[f"units_{p}"] - t0[f"units_{p}"]) for p in products), 0.0
    else:
        volume, mix = beta["units_total"] * (V1 - V0), 0.0
    weather = beta["cdd"] * (t1["cdd"] - t0["cdd"]) + beta["hdd"] * (t1["hdd"] - t0["hdd"])
    return {"volume": calendar + volume, "product_mix": mix, "weather": weather}


def _round_to_total(values: dict, total: float, nd: int = 2) -> dict:
    """Round each slice; put the rounding remainder (at most a few hundredths) on the largest slice,
    so the printed numbers add up exactly to the printed total."""
    out = {k: round(float(v), nd) for k, v in values.items()}
    gap = round(round(float(total), nd) - sum(out.values()), nd)
    if gap:
        big = max(out, key=lambda k: abs(out[k]))
        out[big] = round(out[big] + gap, nd)
    return out


def decompose(ds, cfg: dict, new: str, ref: str, price=None, scenario: str | None = None) -> dict:
    """Full bridge for new vs ref. `price` replaces the interval prices of the new quarter (what-if)."""
    b0, b1 = bill(ds, ref), bill(ds, new, price)
    B0, B1, E0, E1, p0 = b0["bill"], b1["bill"], b0["kwh"], b1["kwh"], b0["avg_price"]
    if B0 <= 0 or E0 <= 0:
        return {"ok": False, "reason": f"no billed energy in {ref}"}
    fit = fit_baseline(ds, cfg, ref)
    if not fit["ok"]:
        return fit

    use_mix = fit["use_mix"]
    t0, t1 = totals(ds, ref, ds.products), totals(ds, new, ds.products)
    pp = lambda kwh: kwh * p0 / B0 * 100            # kWh at the old price -> percentage points of the old bill

    def all_slices(beta: dict, mix_first: bool = False) -> dict:
        s = energy_slices(beta, t0, t1, ds.products, use_mix, mix_first)
        s["waste"] = (E1 - E0) - sum(s.values())
        return s

    point = all_slices(fit["beta"])
    kwh = {"tariff": 0.0, **point}
    money = {"tariff": B1 - E1 * p0, **{k: v * p0 for k, v in point.items()}}
    pct = {k: v / B0 * 100 for k, v in money.items()}
    total_pct = (B1 - B0) / B0 * 100

    # ranges: resampled baseline fits x both orders of volume and mix
    runs = [all_slices(b, mf) for b in bootstrap_betas(fit, cfg["baseline"]["bootstrap_samples"], cfg["baseline"]["seed"])
            for mf in (False, True)] + [all_slices(fit["beta"], True)]
    rng = {}
    for k in point:
        vals = np.array([pp(r[k]) for r in runs])
        rng[k] = [min(float(np.percentile(vals, 5)), pct[k]), max(float(np.percentile(vals, 95)), pct[k])]
    # tariff range: price change valued on the new quarter's energy (ours) or the reference quarter's
    alt = E0 * (b1["avg_price"] - p0) / B0 * 100
    rng["tariff"] = [min(pct["tariff"], alt), max(pct["tariff"], alt)]

    shown = _round_to_total(pct, total_pct)
    bridge = [{"cause": c, "pct_points": shown[c],
               "range": [round(float(min(rng[c][0], shown[c])), 2), round(float(max(rng[c][1], shown[c])), 2)],
               "kwh": int(round(kwh[c])), "money": round(float(money[c]), 2)} for c in CAUSES]

    rates = ({p: round(fit["beta"][f"units_{p}"], 3) for p in ds.products} if use_mix
             else {"all_products": round(fit["beta"]["units_total"], 3)})
    limitations = list(fit["warnings"])
    if not use_mix:
        limitations.append("Baseline run without product mix: a mix shift cannot be separated and lands in waste")
    if t1["cdd"] > 0 and t0["cdd"] > 0 and t1["cdd"] / t1["const"] > 1.5 * t0["cdd"] / t0["const"]:
        limitations.append(f"{new} was much hotter than {ref}: the weather rate is stretched beyond the days it was learned on")
    new_only = [p for p in ds.products if t0[f"units_{p}"] == 0 and t1[f"units_{p}"] > 0]
    if new_only:
        limitations.append(f"Product(s) {new_only} were not made in {ref}; their energy cannot be learned and falls into waste")

    return {
        "ok": True, "period": new, "compare_to": ref, "scenario": scenario,
        "bill": {"reference": {k: round(v, 4 if k == "avg_price" else 2) for k, v in b0.items() if k != "period"},
                 "new": {k: round(v, 4 if k == "avg_price" else 2) for k, v in b1.items() if k != "period"},
                 "change": round(B1 - B0, 2), "change_pct": round(total_pct, 2),
                 "kwh_change": round(E1 - E0), "kwh_change_pct": round((E1 - E0) / E0 * 100, 2),
                 "avg_price_change_pct": round((b1["avg_price"] - p0) / p0 * 100, 2)},
        "bridge": bridge, "total_pct_points": round(sum(r["pct_points"] for r in bridge), 2),
        "baseline": {"model": "linear regression on daily main-meter kWh", "fitted_on": ref,
                     "r2": round(fit["r2"], 3), "days_used": fit["n_days"], "days_excluded": fit["days_excluded"],
                     "kwh_per_unit": rates, "base_kwh_per_day": round(fit["beta"]["const"]),
                     "working_day_kwh": round(fit["beta"]["working"]),
                     "kwh_per_cooling_degree_day": round(fit["beta"]["cdd"], 1),
                     "kwh_per_heating_degree_day": round(fit["beta"]["hdd"], 1),
                     "expected_kwh_new": round(predict_total(fit["beta"], t1)),
                     "expected_kwh_reference": round(predict_total(fit["beta"], t0))},
        "inputs": {"units_reference": round(t0["units_total"]), "units_new": round(t1["units_total"]),
                   "working_days_reference": int(t0["working"]), "working_days_new": int(t1["working"]),
                   "cdd_reference": round(t0["cdd"], 1), "cdd_new": round(t1["cdd"], 1)},
        "limitations": limitations,
    }
