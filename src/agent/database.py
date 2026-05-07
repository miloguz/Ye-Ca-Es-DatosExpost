import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "database" / "Procesos_clean.db"

SCHEMA = """
Base de datos SQLite: Procesos_clean.db
Contiene datos de automatizaciones RPA de una organización de salud en Colombia.

=== TABLAS ===

1. RegistrosDPA_clean (618,875 filas)
   Registros históricos de ejecución de cada bot RPA.
   - Automatizacion     TEXT    Nombre del bot/proceso automatizado
   - Duracion_Ejecucion TEXT    Tiempo que tardó el robot: formato "HH:MM:SS" (90.76% NULL)
   - Numero_Transacciones INTEGER Cantidad de ítems procesados
   - Celula             TEXT    Célula organizacional (ej: "Servicios Organizacionales")
   - Fecha_ejecucion    TEXT    Fecha ISO "YYYY-MM-DD"
   - Hora_Ejecucion     TEXT    Hora "HH:MM:SS AM/PM"
   - Exitosas           REAL    Transacciones completadas exitosamente
   - ErroresNegocio     REAL    Errores de lógica de negocio
   - Area               TEXT    Área organizacional (ej: "Afiliacion")
   - TipoEjecucion      TEXT    Tipo de ejecución
   - IdProceso          INTEGER Identificador del proceso

2. TiemposManuales_clean (95 filas)
   Tiempo que le tomaría a un humano ejecutar cada tarea manualmente.
   - NombreBot              TEXT    Nombre del bot
   - TiempoManualHoras      REAL    Horas que tarda un humano en hacer la misma tarea
   - Tipo                   TEXT    Tipo de proceso
   - Tecnologia             TEXT    Plataforma RPA (UiPath, IRPA, Power Automate)
   - FechaSalidaProduccion  TEXT    Fecha en que el bot salió a producción
   - Estado                 TEXT    Estado actual: "Activo" o "Inactivo"
   - Desarrollador          TEXT    Nombre del desarrollador

3. RolesAreas_clean (91 filas)
   Roles humanos y áreas impactadas por cada bot, con valor económico por hora.
   - Proyecto           TEXT    Nombre del proyecto
   - NombreBot          TEXT    Nombre del bot
   - AreaImpactada      TEXT    Área de negocio afectada
   - Rol_Impactado      TEXT    Perfil del empleado que realizaba la tarea
   - Tipo_Impacto       TEXT    "Ahorrado" (tiempo liberado) o "Salvado" (errores evitados)
   - ValorHoraProyecto  REAL    Costo del rol en COP por hora

=== NOTAS SQL ===
- Para calcular duración en horas desde "HH:MM:SS":
    (CAST(substr(Duracion_Ejecucion,1,2) AS REAL)*3600
    + CAST(substr(Duracion_Ejecucion,4,2) AS REAL)*60
    + CAST(substr(Duracion_Ejecucion,7,2) AS REAL)) / 3600.0
- Filtrar duraciones nulas con: Duracion_Ejecucion IS NOT NULL AND Duracion_Ejecucion != '0'
- ROI estimado por bot = TiempoManualHoras * ValorHoraProyecto * Num_Ejecuciones
"""


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def get_schema() -> str:
    return SCHEMA


def execute_query(sql: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn)


def get_db_stats() -> dict:
    """Retorna estadísticas básicas de la BD para mostrar en el sidebar."""
    with get_connection() as conn:
        stats = {}
        for table in ["RegistrosDPA_clean", "TiemposManuales_clean", "RolesAreas_clean"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count

        bots_activos = conn.execute(
            "SELECT COUNT(*) FROM TiemposManuales_clean WHERE Estado = 'Activo'"
        ).fetchone()[0]
        stats["bots_activos"] = bots_activos

        fecha_min, fecha_max = conn.execute(
            "SELECT MIN(Fecha_ejecucion), MAX(Fecha_ejecucion) FROM RegistrosDPA_clean"
        ).fetchone()
        stats["fecha_inicio"] = fecha_min
        stats["fecha_fin"] = fecha_max

    return stats
