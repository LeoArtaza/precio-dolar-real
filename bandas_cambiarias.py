"""Domain model for policy-driven currency bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = [
    "desde",
    "hasta",
    "piso_nominal",
    "techo_nominal",
    "politica",
    "inflacion_mensual",
]


class PoliticaActualizacion(Protocol):
    @property
    def nombre(self) -> str:
        ...

    def factores(
        self,
        mes: pd.Timestamp,
        inflacion_mensual: pd.Series,
    ) -> tuple[float, float]:
        ...


def _mes_fin(fecha: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(fecha).to_period("M").to_timestamp(how="end").normalize()


def _normalizar_inflacion_mensual(inflacion_mensual: pd.Series) -> pd.Series:
    if inflacion_mensual.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    valores = pd.to_numeric(inflacion_mensual, errors="raise").copy()
    indice = pd.to_datetime(valores.index, errors="raise")
    valores.index = pd.DatetimeIndex(indice).to_period("M").to_timestamp(how="end").normalize()
    return valores.groupby(level=0).last().sort_index()


@dataclass(frozen=True)
class SinActualizacion:
    @property
    def nombre(self) -> str:
        return "sin_actualizacion"

    def factores(
        self,
        mes: pd.Timestamp,
        inflacion_mensual: pd.Series,
    ) -> tuple[float, float]:
        return 1.0, 1.0


@dataclass(frozen=True)
class PorcentajeFijo:
    piso_mensual: float = -0.01
    techo_mensual: float = 0.01

    def __post_init__(self) -> None:
        piso_factor = 1 + self.piso_mensual
        techo_factor = 1 + self.techo_mensual
        if not np.isfinite(self.piso_mensual) or not np.isfinite(self.techo_mensual):
            raise ValueError("Fixed band rates must be finite")
        if piso_factor <= 0 or techo_factor <= 0:
            raise ValueError("Fixed band rates must produce positive factors")

    @property
    def nombre(self) -> str:
        return "porcentaje_fijo"

    def factores(
        self,
        mes: pd.Timestamp,
        inflacion_mensual: pd.Series,
    ) -> tuple[float, float]:
        return 1 + self.piso_mensual, 1 + self.techo_mensual


@dataclass(frozen=True)
class InflacionConRezago:
    rezago_meses: int = 2

    def __post_init__(self) -> None:
        if self.rezago_meses < 0:
            raise ValueError("Inflation lag must be non-negative")

    @property
    def nombre(self) -> str:
        return f"inflacion_t-{self.rezago_meses}"

    def factores(
        self,
        mes: pd.Timestamp,
        inflacion_mensual: pd.Series,
    ) -> tuple[float, float]:
        serie = _normalizar_inflacion_mensual(inflacion_mensual)
        mes_rezagado = _mes_fin(pd.Timestamp(mes) - pd.DateOffset(months=self.rezago_meses))
        tasa = serie.get(mes_rezagado, np.nan)
        if pd.isna(tasa):
            raise ValueError(
                f"Missing monthly inflation for lagged policy month {mes_rezagado.date()}"
            )

        piso_factor = 1 - float(tasa)
        techo_factor = 1 + float(tasa)
        if piso_factor <= 0 or techo_factor <= 0:
            raise ValueError("Monthly inflation must produce positive band factors")
        return piso_factor, techo_factor


@dataclass(frozen=True)
class TramoPolitica:
    desde: pd.Timestamp
    hasta: Optional[pd.Timestamp]
    politica: PoliticaActualizacion

    def __post_init__(self) -> None:
        desde = pd.Timestamp(self.desde).normalize()
        hasta = None if self.hasta is None else pd.Timestamp(self.hasta).normalize()
        if hasta is not None and hasta <= desde:
            raise ValueError("Policy segment end must be after its start")
        object.__setattr__(self, "desde", desde)
        object.__setattr__(self, "hasta", hasta)


@dataclass(frozen=True)
class BandasCambiarias:
    fecha_inicio: pd.Timestamp
    piso_inicial: float
    techo_inicial: float
    tramos: Sequence[TramoPolitica]
    fecha_fin: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        fecha_inicio = pd.Timestamp(self.fecha_inicio).normalize()
        fecha_fin = None if self.fecha_fin is None else pd.Timestamp(self.fecha_fin).normalize()
        if self.piso_inicial <= 0 or self.techo_inicial <= 0:
            raise ValueError("Initial band limits must be positive")
        if self.piso_inicial >= self.techo_inicial:
            raise ValueError("Initial floor must be below the initial ceiling")
        if fecha_fin is not None and fecha_fin < fecha_inicio:
            raise ValueError("Band end must not precede band start")

        tramos = tuple(self.tramos)
        if not tramos:
            raise ValueError("At least one policy segment is required")
        if any(previous.desde > current.desde for previous, current in zip(tramos, tramos[1:])):
            raise ValueError("Policy segments must be ordered by start date")

        object.__setattr__(self, "fecha_inicio", fecha_inicio)
        object.__setattr__(self, "fecha_fin", fecha_fin)
        object.__setattr__(self, "tramos", tramos)

    def _tramo_para(self, fecha: pd.Timestamp) -> TramoPolitica:
        for tramo in reversed(self.tramos):
            if fecha >= tramo.desde and (tramo.hasta is None or fecha < tramo.hasta):
                return tramo
        raise ValueError(f"No policy segment configured for {fecha.date()}")

    @staticmethod
    def _siguiente_mes(fecha: pd.Timestamp) -> pd.Timestamp:
        fin_mes = pd.Timestamp(fecha).to_period("M").to_timestamp(how="end").normalize()
        return fin_mes + pd.Timedelta(days=1)

    def trayectoria(
        self,
        inflacion_mensual: pd.Series,
        fecha_fin: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate nominal band segments ending at ``fecha_fin`` (exclusive)."""
        fecha_fin_efectiva = self.fecha_fin if fecha_fin is None else pd.Timestamp(fecha_fin).normalize()
        if fecha_fin_efectiva is None:
            raise ValueError("A band end date is required to generate a trajectory")
        if fecha_fin_efectiva < self.fecha_inicio:
            raise ValueError("Band end must not precede band start")
        if fecha_fin_efectiva == self.fecha_inicio:
            return pd.DataFrame(columns=TRAJECTORY_COLUMNS)

        inflacion_normalizada = _normalizar_inflacion_mensual(inflacion_mensual)
        filas = []
        desde = self.fecha_inicio
        piso = float(self.piso_inicial)
        techo = float(self.techo_inicial)
        es_segmento_inicial = True

        while desde < fecha_fin_efectiva:
            tramo = self._tramo_para(desde)
            hasta = min(self._siguiente_mes(desde), fecha_fin_efectiva)
            if tramo.hasta is not None:
                hasta = min(hasta, tramo.hasta)
            if hasta <= desde:
                raise ValueError(f"Policy segment does not advance from {desde.date()}")

            tasa = np.nan
            if not es_segmento_inicial:
                piso_factor, techo_factor = tramo.politica.factores(desde, inflacion_normalizada)
                piso *= piso_factor
                techo *= techo_factor
                if isinstance(tramo.politica, InflacionConRezago):
                    mes_rezagado = _mes_fin(
                        desde - pd.DateOffset(months=tramo.politica.rezago_meses)
                    )
                    tasa = float(inflacion_normalizada.loc[mes_rezagado])

            filas.append(
                {
                    "desde": desde,
                    "hasta": hasta,
                    "piso_nominal": piso,
                    "techo_nominal": techo,
                    "politica": tramo.politica.nombre,
                    "inflacion_mensual": tasa,
                }
            )
            desde = hasta
            es_segmento_inicial = False

        return pd.DataFrame(filas, columns=TRAJECTORY_COLUMNS)


def inflacion_mensual_desde_factor_diario(
    factores_diarios: pd.Series,
    dias_base: float = 30.5,
) -> pd.Series:
    """Recover monthly rates from the daily factors stored by the updater."""
    if dias_base <= 0:
        raise ValueError("The monthly conversion base must be positive")
    if factores_diarios.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    factores = pd.to_numeric(factores_diarios, errors="raise").dropna().copy()
    factores.index = pd.to_datetime(factores.index, errors="raise")
    factores = factores.sort_index()
    if (factores <= 0).any():
        raise ValueError("Daily inflation factors must be positive")

    factores_mensuales = factores.resample("ME").last().dropna()
    return factores_mensuales.pow(dias_base).sub(1).rename(factores_diarios.name)
