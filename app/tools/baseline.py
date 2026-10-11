"""Baseline (R2): expected daily energy from production volume, product mix and weather.

    daily kWh = base + working-day overhead + sum(rate_p x units_p) + k_cool x CDD + k_heat x HDD

Fitted on the REFERENCE quarter only. If the new quarter were included, the model would
learn its waste as normal and hide it. Linear on purpose: each term is one cause, so the
effects separate cleanly and add up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def feature_names(products: list[str], use_mix: bool) -> list[str]:
    units = [f"units_{p}" for p in products] if use_mix else ["units_total"]
    return ["const", "working"] + units + ["cdd", "hdd"]


def features(daily: pd.DataFrame, products: list[str], use_mix: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=daily.index)
    X["const"] = 1.0
    X["working"] = daily.working.astype(float)
    for c in feature_names(products, use_mix)[2:]:
        X[c] = daily[c].astype(float)
    return X


def fit_days(ds, ref: str) -> pd.DataFrame:
    """Reference days clean enough to learn from."""
    d = ds.daily[ds.daily.quarter == ref]
    ok = (d.full_day & (d.gap_hours <= ds.cfg["cleaning"]["max_gap_hours_for_fit"])
          & ~d.logged_but_idle & ~d.production_missing & ~d.weather_missing)
    return d[ok]


def _physical(names: list[str]) -> list[str]:
    """Rates that cannot be negative: energy per unit and energy per degree-day."""
    return [n for n in names if n.startswith("units_") or n in ("cdd", "hdd")]


def constrained_lstsq(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Least squares where physical rates may not go below zero. A rate that comes out negative
    is fixed at zero and the fit is repeated without it (most negative first)."""
    active, dropped = list(range(len(names))), []
    while True:
        coef = np.linalg.lstsq(X[:, active], y, rcond=None)[0]
        neg = [(c, i) for c, i in zip(coef, active) if names[i] in _physical(names) and c < 0]
        if not neg:
            break
        worst = min(neg)[1]
        active.remove(worst)
        dropped.append(names[worst])
    full = np.zeros(len(names))
    full[active] = coef
    return full, dropped


def fit_baseline(ds, cfg: dict, ref: str) -> dict:
    use_mix = cfg["baseline"]["use_product_mix"]
    days = fit_days(ds, ref)
    X, y = features(days, ds.products, use_mix), days.kwh_main.astype(float)
    names = list(X.columns)
    # a column that never varies in the reference quarter (e.g. no heating days) cannot be learned
    keep = ["const"] + [c for c in names[1:] if X[c].nunique() > 1]
    warnings = [f"'{c}' did not vary in {ref}, so its effect cannot be estimated and is set to 0"
                for c in names if c not in keep]
    if len(days) < len(keep) + 5:
        return {"ok": False, "reason": f"only {len(days)} usable days in {ref}; too few to fit a baseline"}

    _, negative = constrained_lstsq(X[keep].values, y.values, keep)
    for c in negative:
        warnings.append(f"the data gave a negative rate for '{c}', which is physically impossible; it is set to 0, "
                        f"so this effect cannot be separated and any real effect lands in waste")
    keep = [c for c in keep if c not in negative]
    model = sm.OLS(y.values, X[keep].values).fit()
    beta = dict.fromkeys(names, 0.0)
    beta.update(dict(zip(keep, model.params)))
    ci = model.conf_int(alpha=0.05)
    for c in ("cdd", "hdd"):
        lo, hi = dict(zip(keep, ci)).get(c, (0, 0))
        if c in keep and lo <= 0:
            warnings.append(f"the weather rate ('{c}') is weakly determined from {ref}: 95% interval {lo:.0f} to {hi:.0f} kWh "
                            f"per degree-day includes zero, so the weather and waste slices have wide ranges")
    total = ds.daily[ds.daily.quarter == ref]
    return {
        "ok": True, "names": names, "keep": keep, "beta": beta, "use_mix": use_mix,
        "r2": float(model.rsquared), "n_days": int(len(days)), "days_excluded": int(len(total) - len(days)),
        "resid_std_kwh": float(np.sqrt(model.scale)),
        "conf_int": {k: [float(lo), float(hi)] for k, (lo, hi) in zip(keep, ci)},
        "X": X[keep].values, "y": y.values, "warnings": warnings,
    }


def bootstrap_betas(fit: dict, n: int, seed: int) -> list[dict]:
    """Refit on resampled reference days: how much the rates move if the sample were different."""
    rng = np.random.default_rng(seed)
    X, y, out = fit["X"], fit["y"], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        coef, _ = constrained_lstsq(X[i], y[i], fit["keep"])
        b = dict.fromkeys(fit["names"], 0.0)
        b.update(dict(zip(fit["keep"], coef)))
        out.append(b)
    return out


def totals(ds, period: str, products: list[str]) -> dict:
    """Quarter totals of every baseline input."""
    d = ds.daily[ds.daily.quarter == period]
    out = {"const": float(len(d)), "working": float(d.working.sum()), "cdd": float(d.cdd.sum()),
           "hdd": float(d.hdd.sum()), "units_total": float(d.units_total.sum())}
    for p in products:
        out[f"units_{p}"] = float(d[f"units_{p}"].sum())
    return out


def predict_total(beta: dict, tot: dict) -> float:
    return float(sum(beta[k] * tot[k] for k in beta))
