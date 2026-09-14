from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


Fecha = str | date | datetime | pd.Timestamp | pd.Period


def _timestamp(fecha: Fecha) -> pd.Timestamp:
    if isinstance(fecha, pd.Period):
        return fecha.to_timestamp()
    return pd.Timestamp(fecha)


def _inicio_de_mes(fecha: Fecha) -> pd.Timestamp:
    return _timestamp(fecha).to_period("M").start_time.normalize()


def _fin_de_mes(fecha: Fecha) -> pd.Timestamp:
    return _timestamp(fecha).to_period("M").end_time.normalize()


@dataclass(frozen=True)
class ResultadoInflacionPeriodo:
    """Inflation and real-blue-dollar metrics for an inclusive month period."""

    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    monto_usd_inicial: float
    factor_peso: float
    factor_dolar: float
    precio_blue_inicial: float
    precio_blue_final: float

    @property
    def inflacion_peso_pct(self) -> float:
        return (self.factor_peso - 1) * 100

    @property
    def inflacion_dolar_pct(self) -> float:
        return (self.factor_dolar - 1) * 100

    @property
    def factor_relativo(self) -> float:
        """Argentine price-level factor relative to US dollar inflation."""
        return self.factor_peso / self.factor_dolar

    @property
    def monto_usd_ajustado_por_inflacion_dolar(self) -> float:
        return self.monto_usd_inicial * self.factor_dolar

    @property
    def monto_usd_equivalente(self) -> float:
        """Equivalent current USD amount after the real blue-dollar change."""
        return self.monto_usd_inicial / self.factor_blue_real

    @property
    def monto_usd_revertido(self) -> float:
        return self.monto_usd_equivalente * self.factor_blue_real

    @property
    def variacion_relativa_pct(self) -> float:
        return (self.monto_usd_equivalente / self.monto_usd_inicial - 1) * 100

    @property
    def variacion_inversa_pct(self) -> float:
        return (self.factor_blue_real - 1) * 100

    @property
    def factor_blue_nominal(self) -> float:
        return self.precio_blue_final / self.precio_blue_inicial

    @property
    def variacion_blue_nominal_pct(self) -> float:
        return (self.factor_blue_nominal - 1) * 100

    @property
    def factor_blue_real(self) -> float:
        """Nominal blue-dollar movement net of Argentine/US inflation."""
        return self.factor_blue_nominal / self.factor_relativo

    @property
    def variacion_blue_real_pct(self) -> float:
        return (self.factor_blue_real - 1) * 100


def calcular_inflacion_periodo(
    data: pd.DataFrame,
    fecha_inicio: Fecha,
    fecha_fin: Fecha,
    monto_usd: float,
) -> ResultadoInflacionPeriodo:
    """Calculate inclusive-month inflation and real blue-dollar metrics.

    Dates are interpreted as months: the selected start date is normalized to
    the first day of its month and the end date to the last day of its month.
    Daily multiplicative factors are used so complete months are compounded
    without converting back to rounded percentages.
    """
    if monto_usd <= 0:
        raise ValueError("El monto debe ser mayor que cero.")

    fecha_inicio = _inicio_de_mes(fecha_inicio)
    fecha_fin = _fin_de_mes(fecha_fin)
    if fecha_inicio > fecha_fin:
        raise ValueError("La fecha inicial debe ser anterior a la fecha final.")

    required_columns = {"inflacion_arg", "inflacion_us", "venta_informal"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas: {missing}.")

    frame = data.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("No hay datos para calcular el período.")

    available_start = max(fecha_inicio, frame.index.min().normalize())
    expected_dates = pd.date_range(available_start, fecha_fin, freq="D")
    factors = frame.reindex(expected_dates)[["inflacion_arg", "inflacion_us"]]
    if factors.empty or factors.isna().any().any():
        raise ValueError("Faltan factores diarios de inflación en el período seleccionado.")

    factor_peso = float(factors["inflacion_arg"].prod())
    factor_dolar = float(factors["inflacion_us"].prod())
    if factor_peso <= 0 or factor_dolar <= 0:
        raise ValueError("Los factores de inflación deben ser positivos.")

    precios = frame.loc[fecha_inicio:fecha_fin, "venta_informal"].dropna()
    if precios.empty:
        raise ValueError("No hay precios del dólar blue en el período seleccionado.")

    return ResultadoInflacionPeriodo(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        monto_usd_inicial=float(monto_usd),
        factor_peso=factor_peso,
        factor_dolar=factor_dolar,
        precio_blue_inicial=float(precios.iloc[0]),
        precio_blue_final=float(precios.iloc[-1]),
    )
