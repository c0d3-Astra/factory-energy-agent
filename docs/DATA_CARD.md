# Data card: synthetic-factory v1.0 (Annex D)

| Field | Value |
|---|---|
| **Name and version** | synthetic-factory, v1.0, generated 11 Oct 2026 |
| **How it was made** | `python scripts/generate_data.py --seed 42`. A seeded numpy simulation; no LLM and no prompts were used to create the data. The generator was written with an AI coding assistant (see the AI-usage disclosure in the README). |
| **What it contains** | `meter_15min.csv` 158,105 rows (main + 8 sub-meters: compressors, furnace, hvac, lighting, utilities, line_1, line_2, line_3); `tariff.csv` 4 rows; `production.csv` 549 rows (3 product families x 183 days); `weather.csv` 183 rows. Date range 2026-04-01 to 2026-09-30 (2026-Q2 and 2026-Q3). Row counts are also in `data/sample/data_card.json`, which validation check 6 reads. |
| **Late-drop pack** | `data/late_drop/tariff.csv`: corrected tariff (price change moved from 1 Aug to 20 Jul, peak price 0.1456 corrected to 0.1440). `data/late_drop/meter_15min.csv`: hvac sub-meter for Q3 with a calibration fix (readings x 0.93), 8,832 rows. |
| **Privacy** | Fully synthetic. No real company or personal data. |

## Traps and edge cases planted on purpose

| Guide section 4.4 edge case | Where it is in the data |
|---|---|
| Sub-meters do not add up to the main meter | Main = sum of sub-meters + about 3% unmetered load, every interval |
| Missing intervals | main: 45 minutes on 2026-04-14 and 5 hours on 2026-07-21; hvac: 6 hours on 2026-06-03 |
| Meter resets | furnace 2026-05-20 10:00 (negative reading); line_2 2026-09-02 13:15 (register dump of 250,000) |
| Duplicates | 40 duplicated lighting rows |
| Tariff change in the middle of a quarter | Peak and off-peak prices rise 12% on 2026-08-01 |
| Mix shift that raises energy per unit without waste | Q3 makes more of product C (14 kWh per unit) and less of A (4 kWh per unit) |
| Hot period that raises cooling | 2026-08-03 to 2026-08-16 is 4 C hotter; all of Q3 is warmer than Q2 |
| Public holiday logged as a normal shift | 2026-08-26: `is_holiday` true, `shift_pattern` "2-shift", zero units, lines idle |
| Hidden waste pattern | Compressed-air leak: +35 kW on `compressors`, all hours, from 2026-08-10 |
| Standing waste (not part of the increase) | Lighting stays at 50 kW of its 80 kW when lines are stopped, in both quarters |

## Checks run

`python scripts/load_data.py data/sample --reset` runs the six validation checks. Result on this dataset: overall WARN
(checks 1, 4, 6 PASS; checks 2, 3, 5 WARN because the planted resets, gaps, duplicates and the mislogged holiday are found and handled).

## Known limits

- Energy is generated from a model close to the one the baseline assumes (linear in units and degree-days), so results on this data
  are optimistic compared with a real factory.
- One factory, two quarters, three product families, no heating season (heating degree-days are zero).
- Prices are plain numbers with no currency; action costs in `config.yaml` are assumptions in the same unit.
- Day-to-day noise is 2.5% and independent between days; real factories have drifts, start-up effects and maintenance stops.
- The answer key (`eval/truth.json`) is written by the generator and read only by `scripts/run_eval.py`.
