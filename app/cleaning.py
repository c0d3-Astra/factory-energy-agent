"""Cleaning (R1): put the meters on one regular 15-minute grid, handle resets, duplicates
and gaps, align with tariff, production and weather, and build the daily table.

Everything that was changed is counted in Dataset.log, and shown by validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .store import Store
from .tools.billing import interval_prices

OFF_SHIFTS = {"", "off", "none", "closed", "holiday", "0", "no shift", "nan"}


@dataclass
class Dataset:
    cfg: dict
    version: int
    wide: pd.DataFrame            # cleaned kWh per 15 min; index = UTC, columns = meters
    local: pd.DatetimeIndex       # the same instants in factory local time
    price: pd.Series              # price per kWh for each interval
    productive: pd.Series         # True while production lines are running
    daily: pd.DataFrame           # one row per local day
    products: list[str]
    submeters: list[str]
    tariff: pd.DataFrame
    production: pd.DataFrame
    weather: pd.DataFrame
    log: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return "main" in self.wide.columns and len(self.wide) > 0

    def quarter_of_interval(self) -> pd.Series:
        return pd.Series(quarter_label(self.local), index=self.wide.index)


def quarter_label(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.year.astype(str) + "-Q" + idx.quarter.astype(str)).values


def parse_period(text: str | None) -> str | None:
    """'2026-Q3', '2026 q3', 'Q3 2026' -> '2026-Q3'. None when it is not a quarter label."""
    if not text:
        return None
    m = re.search(r"(20\d{2})\s*[-_/ ]?\s*Q\s*([1-4])", text, re.I) or None
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    m = re.search(r"Q\s*([1-4])\s*[-_/ ]?\s*(20\d{2})", text, re.I)
    return f"{m.group(2)}-Q{m.group(1)}" if m else None


def clean_meter(s: pd.Series, cfg: dict) -> tuple[pd.Series, dict, pd.Series]:
    """One meter on the full grid -> (clean series, counts, mask of long-gap-filled intervals)."""
    c = cfg["cleaning"]
    missing = int(s.isna().sum())
    neg = s < 0
    q99 = s[s >= 0].quantile(0.99) if s.notna().any() else np.nan
    spike = (s > c["spike_factor"] * q99) if q99 and q99 > 0 else pd.Series(False, index=s.index)
    s = s.mask(neg | spike)

    nan = s.isna()
    run = (nan != nan.shift()).cumsum()
    run_len = nan.groupby(run).transform("sum")
    short = nan & (run_len <= c["max_interp_intervals"])
    s = s.where(~short, s.interpolate(limit_area="inside"))

    long = s.isna()
    if long.any():      # long gaps: typical value for that time of day on that kind of day
        key = [s.index.dayofweek >= 5, s.index.hour * 4 + s.index.minute // 15]
        s = s.where(~long, s.groupby(key).transform("median"))
        s = s.interpolate().bfill().ffill().fillna(0.0)
    stats = {"missing_intervals": missing, "negative_resets": int(neg.sum()), "spike_resets": int(spike.sum()),
             "interpolated": int(short.sum()), "profile_filled": int(long.sum())}
    return s, stats, long


def build_dataset(store: Store, cfg: dict) -> Dataset:
    meter, tariff = store.read("meter_15min"), store.read("tariff")
    production, weather = store.read("production"), store.read("weather")
    log: dict = {"meters": {}, "notes": []}

    if meter.empty:
        empty = pd.DataFrame()
        return Dataset(cfg, store.data_version(), empty, pd.DatetimeIndex([]), pd.Series(dtype=float),
                       pd.Series(dtype=bool), empty, [], [], tariff, production, weather, log)

    meter["ts"] = pd.to_datetime(meter.ts).dt.floor("15min")           # snap stray seconds to the grid
    raw = meter.groupby(["ts", "meter_id"]).kwh.last().unstack("meter_id")
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    raw = raw.reindex(grid)
    wide, long_fill = {}, {}
    for m in raw.columns:
        first, last = raw[m].first_valid_index(), raw[m].last_valid_index()
        s, stats, lf = clean_meter(raw[m].loc[first:last], cfg)
        wide[m], long_fill[m] = s.reindex(grid).fillna(0.0), lf.reindex(grid).fillna(False)
        log["meters"][m] = stats
    wide = pd.DataFrame(wide, index=grid)
    wide.index.name = "ts"

    tz = cfg["factory_timezone"]
    local = grid.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    price, price_stats = interval_prices(grid, local, tariff, cfg)
    log["pricing"] = price_stats

    submeters = [m for m in wide.columns if m != "main"]
    lines = [m for m in submeters if any(m.startswith(p) for p in cfg["production_meter_prefixes"])]
    day = local.normalize()

    # production and weather as daily tables
    prod = production.copy()
    prod["date"] = pd.to_datetime(prod.date)
    products = sorted(prod["product"].unique().tolist()) if len(prod) else []
    units = prod.pivot_table(index="date", columns="product", values="units", aggfunc="sum").fillna(0.0)
    units.columns = [f"units_{p}" for p in units.columns]
    cal = prod.groupby("date").agg(shift_pattern=("shift_pattern", "first"), is_holiday=("is_holiday", "max"))
    wx = weather.copy()
    wx["date"] = pd.to_datetime(wx.date)
    wx = wx.set_index("date")[["avg_temp_c", "cooling_degree_days", "heating_degree_days"]]

    days = pd.DatetimeIndex(sorted(set(day)))
    daily = pd.DataFrame(index=days)
    daily.index.name = "date"
    if "main" in wide:
        daily["kwh_main"] = wide["main"].groupby(day).sum().reindex(days).values
        daily["gap_hours"] = long_fill["main"].groupby(day).sum().reindex(days).values / 4
        daily["intervals"] = wide["main"].groupby(day).size().reindex(days).values
    daily = daily.join(units).join(cal).join(wx)
    daily["production_missing"] = daily.shift_pattern.isna()
    daily["weather_missing"] = daily.cooling_degree_days.isna()
    unit_cols = [f"units_{p}" for p in products]
    daily[unit_cols] = daily[unit_cols].fillna(0.0)
    daily["units_total"] = daily[unit_cols].sum(axis=1) if unit_cols else 0.0
    daily["shift_pattern"] = daily.shift_pattern.fillna("").astype(str)
    daily["is_holiday"] = daily.is_holiday.fillna(False).astype(bool)
    for c in ["cooling_degree_days", "heating_degree_days"]:
        daily[c] = daily[c].interpolate().bfill().ffill().fillna(0.0)
    daily = daily.rename(columns={"cooling_degree_days": "cdd", "heating_degree_days": "hdd"})

    # productive intervals: from the production-line meters only
    if lines:
        load = wide[lines].sum(axis=1)
        level = load[load > 0].quantile(0.95) if (load > 0).any() else 0.0
        productive = load > cfg["productive_threshold"] * level
        log["idle_basis"] = f"production-line meters {lines}"
        active = productive.groupby(day).sum().reindex(days).values >= 4          # at least 1 hour of running
        daily["lines_active"] = active
    else:
        shift = daily.shift_pattern.str.lower().reindex(day).values
        rng = [cfg["shift_hours"].get(s, [] if s in OFF_SHIFTS else cfg["default_shift_hours"]) for s in shift]
        start = np.array([r[0] if r else 0 for r in rng])
        end = np.array([r[1] if r else 0 for r in rng])
        worked = daily.units_total.reindex(day).values > 0
        productive = pd.Series(worked & (local.hour >= start) & (local.hour < end), index=grid)
        log["idle_basis"] = "shift_pattern hours from config (no production-line meters found)"
        daily["lines_active"] = daily.units_total > 0
    productive = pd.Series(np.asarray(productive), index=grid)

    daily["logged_but_idle"] = (daily.units_total > 0) & ~daily.lines_active
    daily["holiday_as_shift"] = daily.is_holiday & ~daily.shift_pattern.str.lower().isin(OFF_SHIFTS)
    daily["working"] = (daily.units_total > 0) & daily.lines_active
    daily["quarter"] = quarter_label(days)
    daily["full_day"] = daily.get("intervals", 0) >= 92

    return Dataset(cfg, store.data_version(), wide, local, price, productive, daily, products, submeters,
                   tariff, production, weather, log)


_cache: dict = {}


def get_dataset(store: Store, cfg: dict) -> Dataset:
    """Cleaned data, rebuilt only when new data has been loaded."""
    key = (store.path, store.data_version())
    if _cache.get("key") != key:
        _cache["key"], _cache["ds"] = key, build_dataset(store, cfg)
    return _cache["ds"]


def available_periods(ds: Dataset) -> list[str]:
    """Quarters with enough main-meter days to analyse, oldest first."""
    if not ds.ok:
        return []
    out = []
    for q, n in ds.daily.groupby("quarter").size().items():
        year, qn = int(q[:4]), int(q[-1])
        length = sum(pd.Period(f"{year}-{m:02d}", freq="M").days_in_month for m in range(3 * qn - 2, 3 * qn + 1))
        if n / length >= ds.cfg["min_quarter_coverage"]:
            out.append(q)
    return sorted(out)
