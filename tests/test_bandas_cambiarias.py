import pandas as pd
import pytest

from bandas_cambiarias import (
    BandasCambiarias,
    InflacionConRezago,
    PorcentajeFijo,
    SinActualizacion,
    TramoPolitica,
    inflacion_mensual_desde_factor_diario,
)


def monthly_rates(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)))


def test_no_update_policy_keeps_both_bounds_constant():
    policy = SinActualizacion()

    assert policy.factores(pd.Timestamp("2026-01-01"), pd.Series(dtype=float)) == (1.0, 1.0)


def test_fixed_policy_returns_independent_floor_and_ceiling_factors():
    policy = PorcentajeFijo(piso_mensual=-0.01, techo_mensual=0.01)

    assert policy.factores(pd.Timestamp("2026-01-01"), pd.Series({"ignored": 0.5})) == (
        0.99,
        1.01,
    )


def test_lagged_inflation_policy_uses_exactly_two_months_before():
    policy = InflacionConRezago(rezago_meses=2)
    inflation = monthly_rates(
        {
            "2025-11-01": 0.03,
            "2025-12-01": 0.04,
            "2026-01-01": 0.05,
        }
    )

    assert policy.factores(pd.Timestamp("2026-01-15"), inflation) == (0.97, 1.03)


def test_lagged_inflation_policy_fails_when_required_month_is_missing():
    policy = InflacionConRezago(rezago_meses=2)

    with pytest.raises(ValueError, match="monthly inflation"):
        policy.factores(pd.Timestamp("2026-01-15"), monthly_rates({"2025-12-01": 0.04}))


def test_band_trajectory_transitions_to_t2_policy_on_january_first():
    bands = BandasCambiarias(
        fecha_inicio=pd.Timestamp("2025-04-15"),
        piso_inicial=1000,
        techo_inicial=1400,
        tramos=(
            TramoPolitica(
                desde=pd.Timestamp("2025-04-15"),
                hasta=pd.Timestamp("2026-01-01"),
                politica=PorcentajeFijo(-0.01, 0.01),
            ),
            TramoPolitica(
                desde=pd.Timestamp("2026-01-01"),
                hasta=None,
                politica=InflacionConRezago(2),
            ),
        ),
    )
    inflation = monthly_rates(
        {
            "2025-11-01": 0.03,
            "2025-12-01": 0.04,
            "2026-01-01": 0.05,
        }
    )

    trajectory = bands.trayectoria(inflation, fecha_fin=pd.Timestamp("2026-02-01"))
    january = trajectory.loc[trajectory["desde"] == pd.Timestamp("2026-01-01")].iloc[0]
    december = trajectory.loc[trajectory["desde"] == pd.Timestamp("2025-12-01")].iloc[0]

    assert january["politica"] == "inflacion_t-2"
    assert january["inflacion_mensual"] == pytest.approx(0.03)
    assert january["piso_nominal"] == pytest.approx(december["piso_nominal"] * 0.97)
    assert january["techo_nominal"] == pytest.approx(december["techo_nominal"] * 1.03)


def test_band_trajectory_clips_the_final_segment_and_supports_open_ended_object_horizon():
    bands = BandasCambiarias(
        fecha_inicio=pd.Timestamp("2025-04-15"),
        piso_inicial=1000,
        techo_inicial=1400,
        fecha_fin=pd.Timestamp("2025-06-17"),
        tramos=(
            TramoPolitica(
                desde=pd.Timestamp("2025-04-15"),
                hasta=None,
                politica=PorcentajeFijo(-0.01, 0.01),
            ),
        ),
    )

    trajectory = bands.trayectoria(pd.Series(dtype=float))

    assert trajectory.iloc[0]["desde"] == pd.Timestamp("2025-04-15")
    assert trajectory.iloc[-1]["hasta"] == pd.Timestamp("2025-06-17")


def test_no_update_segment_keeps_nominal_bounds_constant():
    bands = BandasCambiarias(
        fecha_inicio=pd.Timestamp("2026-01-01"),
        piso_inicial=1000,
        techo_inicial=1400,
        tramos=(
            TramoPolitica(
                desde=pd.Timestamp("2026-01-01"),
                hasta=None,
                politica=SinActualizacion(),
            ),
        ),
    )

    trajectory = bands.trayectoria(pd.Series(dtype=float), fecha_fin=pd.Timestamp("2026-04-01"))

    assert trajectory["piso_nominal"].tolist() == [1000, 1000, 1000]
    assert trajectory["techo_nominal"].tolist() == [1400, 1400, 1400]


def test_daily_factor_conversion_recovers_monthly_rate():
    monthly_rate = 0.02
    daily_factor = (1 + monthly_rate) ** (1 / 30.5)
    factors = pd.Series(
        daily_factor,
        index=pd.date_range("2025-01-01", periods=31, freq="D"),
    )

    recovered = inflacion_mensual_desde_factor_diario(factors)

    assert recovered.index.tolist() == [pd.Timestamp("2025-01-31")]
    assert recovered.iloc[0] == pytest.approx(monthly_rate)


def test_equal_start_and_end_returns_empty_schema():
    bands = BandasCambiarias(
        fecha_inicio=pd.Timestamp("2026-01-01"),
        piso_inicial=1000,
        techo_inicial=1400,
        tramos=(TramoPolitica(pd.Timestamp("2026-01-01"), None, SinActualizacion()),),
    )

    result = bands.trayectoria(pd.Series(dtype=float), fecha_fin=pd.Timestamp("2026-01-01"))

    assert result.empty
    assert list(result.columns) == [
        "desde",
        "hasta",
        "piso_nominal",
        "techo_nominal",
        "politica",
        "inflacion_mensual",
    ]
