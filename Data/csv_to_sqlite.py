"""
csv_to_sqlite.py
----------------
Script para la conversión de archivos CSV a una base de datos SQLite.

Archivos de entrada (dentro de la carpeta 'Data'):
    - RegistrosDPA.csv
    - TiemposManuales.csv
    - RolesAreas.csv

Salida:
    - Data/Procesos.db

Librerías requeridas: pandas, sqlite3 (incluida en la stdlib de Python)
"""

import os
import sqlite3
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Ruta base: carpeta "Data" relativa al directorio donde se ejecuta el script
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# Archivos CSV a procesar: {nombre_tabla: nombre_archivo}
ARCHIVOS_CSV = {
    "RegistrosDPA":    "RegistrosDPA.csv",
    "TiemposManuales": "TiemposManuales.csv",
    "RolesAreas":      "RolesAreas.csv",
}

# Nombre de la base de datos de salida
NOMBRE_DB = "Procesos.db"

# Separador usado en los archivos CSV
# Cambia a "," si los archivos usan coma como separador
SEPARADOR = ";"

# Codificación esperada de los archivos CSV
ENCODING = "utf-8-sig"   # utf-8-sig tolera el BOM que agrega Excel al guardar como CSV


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def verificar_archivos(base_dir: str, archivos: dict) -> tuple[list, list]:
    """
    Verifica cuáles archivos existen y cuáles no en el directorio indicado.

    Args:
        base_dir: Ruta al directorio donde se buscan los archivos.
        archivos:  Diccionario {nombre_tabla: nombre_archivo}.

    Returns:
        Tupla (encontrados, faltantes) con listas de nombres de archivo.
    """
    encontrados = []
    faltantes = []

    for tabla, archivo in archivos.items():
        ruta = os.path.join(base_dir, archivo)
        if os.path.isfile(ruta):
            encontrados.append(tabla)
        else:
            faltantes.append(archivo)

    return encontrados, faltantes


def leer_csv(ruta: str, separador: str, encoding: str) -> pd.DataFrame:
    """
    Lee un archivo CSV y devuelve un DataFrame de pandas.

    Intenta primero con el separador indicado; si el resultado tiene una sola
    columna, reintenta con coma como separador de fallback.

    Args:
        ruta:      Ruta completa al archivo CSV.
        separador: Separador de columnas esperado.
        encoding:  Codificación del archivo.

    Returns:
        DataFrame con los datos del CSV.
    """
    df = pd.read_csv(ruta, sep=separador, encoding=encoding, engine="python")

    # Heurística: si hay solo 1 columna, el separador puede ser incorrecto
    if df.shape[1] == 1:
        sep_alterno = "," if separador != "," else ";"
        print(f"  [AVISO] Solo se detectó 1 columna con sep='{separador}'. "
              f"Reintentando con sep='{sep_alterno}'...")
        df = pd.read_csv(ruta, sep=sep_alterno, encoding=encoding, engine="python")

    return df


def cargar_a_sqlite(df: pd.DataFrame, nombre_tabla: str, conexion: sqlite3.Connection) -> None:
    """
    Carga un DataFrame en una tabla SQLite.

    Si la tabla ya existe, es reemplazada completamente (if_exists='replace').

    Args:
        df:           DataFrame a guardar.
        nombre_tabla: Nombre de la tabla destino en la base de datos.
        conexion:     Conexión activa a la base de datos SQLite.
    """
    df.to_sql(nombre_tabla, con=conexion, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CONVERSIÓN DE CSV A SQLITE — Procesos.db")
    print("=" * 60)
    print(f"\nDirectorio de datos : {BASE_DIR}")
    print(f"Base de datos salida: {os.path.join(BASE_DIR, NOMBRE_DB)}\n")

    # 1. Verificar que la carpeta Data existe
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] La carpeta 'Data' no existe en: {BASE_DIR}")
        print("        Crea la carpeta y coloca los archivos CSV dentro de ella.")
        return

    # 2. Verificar existencia de archivos CSV
    encontrados, faltantes = verificar_archivos(BASE_DIR, ARCHIVOS_CSV)

    if faltantes:
        print("[AVISO] Los siguientes archivos NO se encontraron y serán omitidos:")
        for f in faltantes:
            print(f"        - {f}")
        print()

    if not encontrados:
        print("[ERROR] No se encontró ningún archivo CSV. "
              "Verifica que estén en la carpeta 'Data'.")
        return

    # 3. Conectar a la base de datos SQLite (se crea si no existe)
    ruta_db = os.path.join(BASE_DIR, NOMBRE_DB)
    conexion = sqlite3.connect(ruta_db)

    tablas_creadas = []

    try:
        # 4. Iterar sobre los archivos encontrados y cargarlos
        for nombre_tabla in encontrados:
            nombre_archivo = ARCHIVOS_CSV[nombre_tabla]
            ruta_csv = os.path.join(BASE_DIR, nombre_archivo)

            print(f"Procesando: {nombre_archivo}")

            try:
                # Leer el CSV
                df = leer_csv(ruta_csv, SEPARADOR, ENCODING)

                # Información básica del archivo leído
                print(f"  - Filas encontradas  : {len(df):,}")
                print(f"  - Columnas detectadas: {list(df.columns)}")

                # Cargar a SQLite
                cargar_a_sqlite(df, nombre_tabla, conexion)

                tablas_creadas.append(nombre_tabla)
                print(f"  [OK] Tabla '{nombre_tabla}' creada exitosamente en Procesos.db\n")

            except Exception as e:
                print(f"  [ERROR] No se pudo procesar '{nombre_archivo}': {e}\n")

        # 5. Confirmar los cambios en la base de datos
        conexion.commit()

    finally:
        conexion.close()

    # 6. Resumen final
    print("-" * 60)
    print(f"Proceso finalizado. Tablas creadas: {len(tablas_creadas)}/{len(ARCHIVOS_CSV)}")
    for t in tablas_creadas:
        print(f"  ✓ {t}")
    if tablas_creadas:
        print(f"\nBase de datos guardada en:\n  {ruta_db}")
    print("=" * 60)


if __name__ == "__main__":
    main()
