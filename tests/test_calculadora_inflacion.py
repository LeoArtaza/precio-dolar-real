import pandas as pd
import pytest

from calculadora_inflacion import calcular_inflacion_periodo


def datos_de_prueba() -> pd.DataFrame:
    index = pd.date_range("2022-05-01", "2022-06-30", freq="D")
    arg = pd.Series(index=index, dtype=float)
    usd = pd.Series(index=index, dtype=float)
    arg.loc["2022-05-01":"2022-05-31"] = 1.10 ** (1 / 31)
    arg.loc["2022-06-01":"2022-06-30"] = 1.20 ** (1 / 30)
    usd.loc["2022-05-01":"2022-05-31"] = 1.02 ** (1 / 31)
    usd.loc["2022-06-01":"2022-06-30"] = 1.03 ** (1 / 30)

    return pd.DataFrame(
        {
            "inflacion_arg": arg,
            "inflacion_us": usd,
            "venta_informal": 100.0,
        },
        index=index,
    ).assign(
        venta_informal=lambda frame: frame.index.to_series().map(
            lambda date: 100.0 if date < pd.Timestamp("2022-06-30") else 130.0
        )
    )


def test_calculates_inclusive_period_and_real_usd_equivalent():
    result = calcular_inflacion_periodo(
        datos_de_prueba(),
        fecha_inicio="2022-05-15",
        fecha_fin="2022-06-02",
        monto_usd=500,
    )

    relative_factor = (1.10 * 1.20) / (1.02 * 1.03)
    real_blue_factor = 1.30 / relative_factor

    assert result.fecha_inicio == pd.Timestamp("2022-05-01")
    assert result.fecha_fin == pd.Timestamp("2022-06-30")
    assert result.factor_peso == pytest.approx(1.32)
    assert result.factor_dolar == pytest.approx(1.0506)
    assert result.factor_relativo == pytest.approx(relative_factor)
    assert result.monto_usd_equivalente == pytest.approx(500 / real_blue_factor)
    assert result.monto_usd_revertido == pytest.approx(500)
    assert result.inflacion_peso_pct == pytest.approx(32.0)
    assert result.inflacion_dolar_pct == pytest.approx(5.06)
    assert result.variacion_relativa_pct == pytest.approx((1 / real_blue_factor - 1) * 100)
    assert result.variacion_inversa_pct == pytest.approx((real_blue_factor - 1) * 100)


def test_calculates_nominal_and_real_blue_variation():
    result = calcular_inflacion_periodo(
        datos_de_prueba(),
        fecha_inicio="2022-05-01",
        fecha_fin="2022-06-30",
        monto_usd=500,
    )

    relative_factor = (1.10 * 1.20) / (1.02 * 1.03)

    assert result.precio_blue_inicial == pytest.approx(100)
    assert result.precio_blue_final == pytest.approx(130)
    assert result.variacion_blue_nominal_pct == pytest.approx(30)
    assert result.factor_blue_real == pytest.approx(1.30 / relative_factor)
    assert result.variacion_blue_real_pct == pytest.approx((1.30 / relative_factor - 1) * 100)


def test_rejects_non_positive_amount():
    with pytest.raises(ValueError, match="mayor que cero"):
        calcular_inflacion_periodo(
            datos_de_prueba(),
            fecha_inicio="2022-05-01",
            fecha_fin="2022-06-30",
            monto_usd=0,
        )


def test_rejects_missing_daily_inflation_factor():
    data = datos_de_prueba()
    data.loc[pd.Timestamp("2022-06-15"), "inflacion_arg"] = None

    with pytest.raises(ValueError, match="factores diarios"):
        calcular_inflacion_periodo(
            data,
            fecha_inicio="2022-05-01",
            fecha_fin="2022-06-30",
            monto_usd=500,
        )


def test_accepts_month_periods_from_streamlit_selectors():
    result = calcular_inflacion_periodo(
        datos_de_prueba(),
        fecha_inicio=pd.Period("2022-05", freq="M"),
        fecha_fin=pd.Period("2022-06", freq="M"),
        monto_usd=500,
    )

    assert result.fecha_inicio == pd.Timestamp("2022-05-01")
    assert result.fecha_fin == pd.Timestamp("2022-06-30")
