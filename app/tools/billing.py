"""Billing (R4): every 15-minute reading is priced at the tariff in force at that moment,
so a price change part-way through a quarter is handled without special cases."""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_peak(local: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    start, end = cfg["peak_hours"]["start"], cfg["peak_hours"]["end"]
    return np.asarray((local.hour >= start) & (local.hour < end))


def interval_prices(utc: pd.DatetimeIndex, local: pd.DatetimeIndex, tariff: pd.DataFrame, cfg: dict):
    """Price per kWh for each interval. Validity is matched in UTC, peak/off-peak by local hour.
    A 'flat' row applies to every hour. Returns (prices, stats)."""
    price = np.full(len(utc), np.nan)
    peak = is_peak(local, cfg)
    for row in tariff.sort_values("valid_from").itertuples(index=False):
        m = np.asarray((utc >= row.valid_from) & (utc < row.valid_to))
        if row.period == "peak":
            m &= peak
        elif row.period == "off_peak":
            m &= ~peak
        price[m] = row.price_per_kwh
    s = pd.Series(price, index=utc)
    unpriced = int(s.isna().sum())
    if unpriced:   # never drop energy from the bill: use the nearest known price and report it
        s = s.ffill().bfill()
    return s, {"unpriced_intervals": unpriced, "tariff_rows": int(len(tariff))}


def prices_as_of(when: pd.Timestamp, tariff: pd.DataFrame) -> dict:
    """Prices in force at one moment, by label (used for 'what if the tariff had not changed')."""
    live = tariff[(tariff.valid_from <= when) & (tariff.valid_to > when)]
    if live.empty:
        live = tariff[tariff.valid_from <= when].sort_values("valid_from").groupby("period").tail(1)
    return {r.period: float(r.price_per_kwh) for r in live.itertuples(index=False)}


def bill(ds, period: str, price: pd.Series | None = None) -> dict:
    """Bill and energy for one quarter from the main meter (the bill-authoritative total)."""
    mask = (ds.quarter_of_interval() == period).values
    kwh = ds.wide["main"].values[mask]
    p = (ds.price if price is None else price).values[mask]
    peak = is_peak(ds.local[mask], ds.cfg)
    energy, money = float(kwh.sum()), float((kwh * p).sum())
    return {"period": period, "kwh": energy, "bill": money,
            "avg_price": money / energy if energy else 0.0,
            "peak_kwh_share_pct": float(kwh[peak].sum() / energy * 100) if energy else 0.0,
            "days": int(mask.sum() / 96)}
