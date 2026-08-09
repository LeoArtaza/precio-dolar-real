# Policy-Driven Currency Bands

## Status

Approved by the user for implementation planning.

## Context

The chart currently hardcodes one monthly crawl rule for the entire band period. That is incorrect for Argentina's current regime:

- From the April 2025 launch through December 2025, the floor and ceiling moved by -1% and +1% per month.
- From January 2026, the BCRA changed the crawl to the latest monthly INDEC inflation rate with a two-month lag (T-2).

The band period must remain open-ended when no end date is configured and use the available inflation-expectation horizon. A configured end date caps the generated trajectory.

## Goals

1. Model currency-band update rules explicitly without encoding the whole regime in chart code.
2. Support three update policies:
   - no update: floor and ceiling remain constant;
   - fixed percentage: independent floor and ceiling monthly rates;
   - lagged inflation: floor and ceiling move according to monthly inflation at a configured lag.
3. Represent the 2025-to-2026 regime transition as two date-bounded policy segments.
4. Keep nominal band generation separate from the real-dollar adjustment and Plotly rendering.
5. Extend the open-ended band period to the last available expectations date without inventing a fixed inflation target.
6. Preserve future expectations without publishing future prices.

## Non-goals

- No generic policy engine, plugin registry, or class hierarchy.
- No change to the dollar-price data or to the real-dollar adjustment formula.
- No replacement of the existing Streamlit charting layer.
- No automatic browser/server launch.

## Proposed Design

### `BandasCambiarias`

Create a small domain object, preferably in a dedicated module, responsible only for generating nominal band limits over a date range.

Inputs:

- `fecha_inicio`;
- `piso_inicial`;
- `techo_inicial`;
- optional `fecha_fin`;
- ordered policy segments.

Output:

- a dated table of band segments containing start date, end date, floor, ceiling, policy name, and applied monthly inflation rate when relevant.

The class does not know about Plotly, Streamlit session state, or real-dollar adjustment.

### Update policies

Use a small policy interface/callable rather than a hierarchy of classes. Each policy receives the month being generated and the monthly inflation series, then returns the floor and ceiling multiplicative factors.

- `sin_actualizacion`: `(1.0, 1.0)`.
- `porcentaje_fijo(piso=-0.01, techo=0.01)`: `(1 + piso, 1 + techo)`.
- `inflacion_con_rezago(rezago=2)`: for band month `M`, read monthly inflation from `M - 2 months` and return `(1 - inflation, 1 + inflation)`.

The policy should fail clearly when a required inflation month is unavailable rather than silently creating a missing band segment. For an open-ended future period, the monthly inflation input may use REM expectations for months not yet covered by realized INDEC data; the source choice belongs to the data-preparation layer, not the band policy.

### Regime segments

The initial configuration will contain:

1. The existing band start and initial floor/ceiling.
2. A fixed-percentage segment through 2025-12-31.
3. An inflation-with-two-month-lag segment from 2026-01-01 onward.

The final segment ends at the configured `fecha_fin`, or at the last available expectations date when no end is configured. The final monthly segment is clipped exactly to that date.

### Data flow

1. `update_db.py` builds the realized and expected monthly inflation series.
2. It creates a price-only daily dataset for Postgres and a publication dataset that also includes future expectation rows with null prices.
3. `app.py` separates price rows from the full expectation horizon.
4. `BandasCambiarias` generates nominal limits using the correct policy segment for each month.
5. `app.py` applies the existing real-dollar adjustment to the generated limits and renders them.

## Edge Cases

- A no-update policy keeps both limits constant but still generates the band.
- A missing T-2 inflation value is an explicit data error, not a reason to fall back to ±1%.
- A final date before the start date produces no segments and a clear validation error.
- A final date inside a month clips the last segment without adding an extra month.
- Future expectation rows must have null prices so they cannot affect KPIs, price lines, or date sliders.

## Verification

Add focused tests for:

1. no-update policy keeps floor and ceiling unchanged;
2. fixed policy applies independent -1%/+1% factors;
3. lagged-inflation policy uses exactly T-2;
4. the 2025 fixed segment transitions to the 2026 lagged segment;
5. open-ended periods use the expectations horizon;
6. a partial final month is clipped exactly;
7. future expectation rows are continuous from the day after the last price and contain no prices;
8. existing app compilation and updater execution remain successful.

## Source

- BCRA, [Régimen de bandas cambiarias](https://web2.bcra.gob.ar/PublicacionesEstadisticas/bandas-cambiarias-piso-techo.asp)
- BCRA, [Objetivos y Planes 2026](https://www.bcra.gob.ar/noticias/objetivos-y-planes-2026/)
