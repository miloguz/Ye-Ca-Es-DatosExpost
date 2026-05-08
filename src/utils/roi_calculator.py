"""
Cálculo de ROI para automatizaciones RPA.

Fórmula:
  Beneficio_Bruto = TiempoManual_Horas × ValorHora_COP × Num_Ejecuciones
  Costo_Robot     = DuracionRobot_Horas × ValorHora_COP × FACTOR_COSTO_ROBOT × Num_Ejecuciones
  Ahorro_Neto     = Beneficio_Bruto - Costo_Robot
  ROI_%           = (Ahorro_Neto / Costo_Robot) × 100

El FACTOR_COSTO_ROBOT (0.25) refleja que mantener un robot cuesta ~25%
de lo que costaría un operador humano ejecutando la misma tarea.
"""

import sqlite3

import numpy as np
import pandas as pd

from ..agent.database import DB_PATH

FACTOR_COSTO_ROBOT = 0.25


def _parse_duration_hours(series: pd.Series) -> pd.Series:
    """Convierte columna HH:MM:SS a horas decimales. NULL / '0' → NaN."""
    def _parse(val):
        if pd.isna(val) or str(val).strip() in ("0", ""):
            return np.nan
        parts = str(val).split(":")
        if len(parts) != 3:
            return np.nan
        try:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h + m / 60 + s / 3600
        except ValueError:
            return np.nan

    return series.apply(_parse)


def build_roi_dataset() -> pd.DataFrame:
    """
    Construye un DataFrame con el ROI calculado para cada bot.
    Fusiona RegistrosDPA_clean, TiemposManuales_clean y RolesAreas_clean.
    """
    with sqlite3.connect(DB_PATH) as conn:
        ejecuciones = pd.read_sql_query(
            """
            SELECT
                Automatizacion,
                COUNT(*) AS Num_Ejecuciones,
                AVG(
                    CASE
                        WHEN Duracion_Ejecucion IS NOT NULL AND Duracion_Ejecucion != '0'
                        THEN (
                            CAST(substr(Duracion_Ejecucion,1,2) AS REAL)*3600
                            + CAST(substr(Duracion_Ejecucion,4,2) AS REAL)*60
                            + CAST(substr(Duracion_Ejecucion,7,2) AS REAL)
                        ) / 3600.0
                    END
                ) AS DuracionPromedio_Horas,
                AVG(Numero_Transacciones) AS PromTransacciones,
                SUM(Exitosas) * 1.0 / NULLIF(SUM(Numero_Transacciones), 0) AS TasaExito,
                SUM(ErroresNegocio) * 1.0 / NULLIF(SUM(Numero_Transacciones), 0) AS TasaError,
                MIN(Fecha_ejecucion) AS PrimeraEjecucion,
                MAX(Fecha_ejecucion) AS UltimaEjecucion,
                COUNT(DISTINCT Area) AS NumAreas
            FROM RegistrosDPA_clean
            GROUP BY Automatizacion
            """,
            conn,
        )

        manuales = pd.read_sql_query(
            "SELECT NombreBot, TiempoManualHoras, Tecnologia, Estado, FechaSalidaProduccion FROM TiemposManuales_clean",
            conn,
        )

        roles = pd.read_sql_query(
            """
            SELECT NombreBot,
                   AVG(ValorHoraProyecto) AS ValorHoraPromedio,
                   COUNT(DISTINCT Rol_Impactado) AS NumRoles,
                   MAX(CASE WHEN Tipo_Impacto LIKE 'Ahorrado%' THEN 1 ELSE 0 END) AS EsAhorro
            FROM RolesAreas_clean
            GROUP BY NombreBot
            """,
            conn,
        )

    df = ejecuciones.merge(manuales, left_on="Automatizacion", right_on="NombreBot", how="left")
    df = df.merge(roles, on="NombreBot", how="left")

    df["PrimeraEjecucion"] = pd.to_datetime(df["PrimeraEjecucion"], errors="coerce")
    df["UltimaEjecucion"] = pd.to_datetime(df["UltimaEjecucion"], errors="coerce")
    df["DiasEnProduccion"] = (df["UltimaEjecucion"] - df["PrimeraEjecucion"]).dt.days.clip(lower=1)
    df["EjecucionesPorDia"] = df["Num_Ejecuciones"] / df["DiasEnProduccion"]

    df["Beneficio_Bruto_COP"] = (
        df["TiempoManualHoras"] * df["ValorHoraPromedio"] * df["Num_Ejecuciones"]
    )
    df["Costo_Robot_COP"] = (
        df["DuracionPromedio_Horas"].fillna(df["TiempoManualHoras"] * 0.1)
        * df["ValorHoraPromedio"]
        * FACTOR_COSTO_ROBOT
        * df["Num_Ejecuciones"]
    )
    df["Ahorro_Neto_COP"] = df["Beneficio_Bruto_COP"] - df["Costo_Robot_COP"]
    df["ROI_Porcentaje"] = (
        (df["Ahorro_Neto_COP"] / df["Costo_Robot_COP"].replace(0, np.nan)) * 100
    )
    df["TiempoAhorrado_Horas"] = (
        df["TiempoManualHoras"]
        - df["DuracionPromedio_Horas"].fillna(df["TiempoManualHoras"] * 0.1)
    ) * df["Num_Ejecuciones"]

    return df


def get_roi_summary(df: pd.DataFrame) -> dict:
    """Resumen ejecutivo del ROI total del portafolio."""
    df_valid = df.dropna(subset=["ROI_Porcentaje"])
    return {
        "total_bots": len(df),
        "bots_con_roi": len(df_valid),
        "ahorro_total_cop": df["Ahorro_Neto_COP"].sum(),
        "roi_promedio_pct": df_valid["ROI_Porcentaje"].mean(),
        "roi_mediano_pct": df_valid["ROI_Porcentaje"].median(),
        "mejor_bot": df_valid.loc[df_valid["ROI_Porcentaje"].idxmax(), "Automatizacion"]
        if not df_valid.empty
        else "N/A",
        "tiempo_ahorrado_horas": df["TiempoAhorrado_Horas"].sum(),
    }
