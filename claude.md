# Factory Energy Agent — Context for Claude

## What this project is

HCLTech Future Ready AI Engineer Hackathon, Track D (Oct 2026).

**Goal:** Explain *why* a factory's electricity bill rose 15% quarter-on-quarter by splitting the change into exact, additive causes — tariff, volume, product mix, weather, and waste — and then recommending actionable (and approved) savings. Only after tariff, volume, mix, and weather are accounted for can anything be called "waste" or blamed on a machine.

## Scoring (100 pts)

| Criterion | Pts |
|---|---|
| Correct cost decomposition| 25 |
| Waste/leak detection | 15 |
| Savings & CO2e quantification | 15 |
| Reasoning before blaming | 10 |
| LLM brief faithfulness | 10 |
| Evaluation rigour | 5 |
| Engineering quality | 10 |
| Architecture judgement | 10 |

**~75 pts are analytics correctness. The LLM is a thin layer on top.**

## Five core decisions

1. **Midpoint (Bennet/Shapley) decomposition** — parts add up exactly, order-independent.
2. **One linear model per sub-meter + unmetered bucket** — waste lands on a specific meter.
3. **Waste = closing residual, tested for significance** — blame gate requires p-value + n signals + post-change days.
4. **Bootstrap ranges from weekly block resampling** — honest uncertainty, not arbitrary ±%.
5. **LLM writes only from pre-formatted facts, checked number-by-number** — faithfulness is testable.

## Data inputs

| File | Content |
|---|---|
| `meter_15min.csv` | meter_id, ts (15-min UTC), kwh — main + 8 sub-meters, 2 quarters (~157k rows) |
| `tariff.csv` | valid_from, valid_to, period (peak/off_peak/flat), price_per_kwh |
| `production.csv` | date, product (family A/B/C), units, shift_pattern, is_holiday |
| `weather.csv` | date, avg_temp_c, cooling_degree_days, heating_degree_days |

Late data drop at 13:45: corrected tariff + one sub-meter calibration fix.

## Sub-meters

`compressors`, `hvac`, `furnace`, `lighting`, `line_1`, `line_2`, `line_3`, `other`  
`unmetered = main − Σ sub` (its own model, never blamed on a machine)

## Cost decomposition formula (midpoint split)

```
tariff     = Σ_π (p̄r−p̄b) × (Eb+Er)/2
volume     = shifts + calendar + unit volume effects (per family, midpoint)
mix        = Σ_k b_k × (share_r − share_b) × (V_b+V_r)/2
weather    = c×ΔCDD + h×ΔHDD
waste      = ΔE − volume − mix − weather   [closing term, per meter]
pp_j       = 100 × cost_j / C_base

assert |Σpp − total| < 0.01   # strict additivity
```

## Key analytics modules

| Module | Purpose |
|---|---|
| `app/ingest.py` + `validate.py` | Load, validate (10 checks V1-V10), chunk |
| `app/clean.py` | Dedupe, reset fix, gap fill, flag each row |
| `app/reconcile.py` | Holiday reclassification, unmetered bucket |
| `app/cost.py` | Interval-level price join, period totals |
| `app/baseline.py` | Per-meter OLS with HAC SEs |
| `app/decompose.py` | Exact midpoint decomposition |
| `app/uncertainty.py` | 1,000-replicate block bootstrap |
| `app/findings.py` | Changepoint leak detection, idle ratio |
| `app/savings.py` | kWh/money/CO2e/payback ranges from data |
| `app/gate.py` | Blame gate — reasoning before blaming |

## LLM usage (exactly 2 touchpoints)

1. **Router fallback** — parse ambiguous natural-language request → typed `Intent` enum + slots.
2. **Brief writer** — prose from a pre-formatted fact table only (no computation, no raw numbers).

Faithfulness check: extract every number from prose → match to facts table → mismatch → retry once → template fallback.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | Main Q&A — returns `EnergyAnswer` |
| `POST /ingest` | Upload corrected CSV |
| `GET /health` | API/store/LLM status |
| `GET /validate` | Latest validation report |
| `GET /trace/{run_id}` | Step-by-step trace with timings |
| `GET /datasets` | Version lineage |

## EnergyAnswer response structure

```json
{
  "run_id": "en-5d20ab",
  "answer_type": "decomposition | recommendation | calculated | not_in_data | clarification_needed | refused",
  "summary": "...",
  "period": "2026-Q3",
  "compare_to": "2026-Q2",
  "total_change_pct": 15.0,
  "bridge": [
    {"cause": "tariff",       "pct_points": 8.1,  "range": [7.9, 8.3], "money": 9720},
    {"cause": "product_mix",  "pct_points": 3.4,  "range": [2.8, 4.0], "money": 4080, "kwh": 32600},
    {"cause": "volume",       "pct_points": 1.2,  "range": [0.8, 1.6], "money": 1440, "kwh": 11500},
    {"cause": "weather",      "pct_points": 0.9,  "range": [0.5, 1.3], "money": 1080, "kwh": 8600},
    {"cause": "waste",        "pct_points": 1.4,  "range": [0.9, 1.9], "money": 1680, "kwh": 16100,
     "components": {"compressors": 1.2, "unmetered": 0.2}}
  ],
  "recommendations": [
    {
      "action": "Survey and repair compressed-air leaks",
      "rationale": "Night load 38% of day (was 14% in Q2)",
      "confidence": "medium",
      "estimated_impact": "Saves 120–160 MWh and 48–64 t CO2e/yr",
      "saving_kwh_per_year": [120000, 160000],
      "saving_money_per_year": [14400, 19200],
      "saving_co2e_t_per_year": [48, 64],
      "payback_months": [3, 6],
      "ease": "easy",
      "category": "cause_of_increase"
    }
  ],
  "assumptions": ["Emission factor 0.4 kg CO2e/kWh", "Bill = energy charges only"],
  "limitations": ["Weather effect estimated from one hot quarter"],
  "dataset_version": "ds-7f21c4",
  "validation_status": "warnings"
}
```

## Streamlit UI pages (design.md §11)

| Page | Key content |
|---|---|
| **Ask** | Question box, period selectors, answer card, bridge waterfall chart, recommendations table |
| **Data** | Health status, dataset versions, CSV upload, validation report, ingest log |
| **Explore** | Energy by meter (stacked area), idle-load heatmap (hour-of-week × meter), leak evidence chart |
| **Evaluation** | eval/report.md rendered, config comparison table |

## Charts required (upgrade task)

- **Waterfall / bridge chart** — pct points by cause, with error bar whiskers for ranges (existing, keep)
- **Pie chart** — cost share by cause (tariff / mix / volume / weather / waste)
- **Pie chart** — energy share by sub-meter (kWh distribution)
- **Bar chart** — savings comparison: kWh, money, CO2e per recommended action
- **Bar chart** — idle kWh per asset (production vs non-production hours)
- **Bar chart** — waste by meter (pct points)
- **Line chart** — energy trend over time (base vs reporting period, main meter and key sub-meters)
- **Line chart** — daily non-production kW with changepoint marker (leak evidence)
- **Bar chart grouped** — peak vs off-peak split, base vs reporting period

## Key domain rules for the UI

- Ranges **do not add up** — never sum ranges in charts; show point estimates with error bars
- Waste colour: **amber/orange** (controllable)
- Non-waste increases: **red**
- Decreases: **green**
- Totals/neutral: **dark slate**
- A manager using this needs to read the key finding in under 2 minutes
- All recommendations require approval — always label them as "suggestions"
- Emission factor: 0.4 kg CO2e per kWh (always stated as assumption)

## Tech stack

Python 3.12, FastAPI, LangGraph, DuckDB, SQLite, pandas, statsmodels, plotly, Streamlit, Docker.

## Environment variables

`MOCK_LLM`, `LLM_PROVIDER` (ollama/openai/anthropic/gemini), `LLM_MODEL`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_HOST`, `API_URL`
