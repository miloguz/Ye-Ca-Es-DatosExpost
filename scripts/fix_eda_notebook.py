"""
Script para corregir el notebook 02_Analisis_EDA_Procesos_RPA.ipynb:
1. Reemplaza textos en celdas Markdown (títulos, descripciones, conclusiones por sección).
2. Convierte la celda de CÓDIGO que imprime conclusiones finales en una celda MARKDOWN.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(
    r"c:\Users\esneydergm\OneDrive - Caja de Compensacion Familiar de Antioquia COMFAMA"
    r"\Escritorio\Proyecto1Especializacion\notebooks\02_Analisis_EDA_Procesos_RPA.ipynb"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. REEMPLAZOS EN CELDAS MARKDOWN EXISTENTES
# ─────────────────────────────────────────────────────────────────────────────
MARKDOWN_REPLACEMENTS = {
    "# Análisis Exploratorio de Datos (EDA) — Procesos RPA": [
        "# Análisis Exploratorio de Datos (EDA) — Procesos RPA\n",
        "\n",
        "Este notebook realiza un análisis exploratorio de la base de datos SQLite "
        "**`Procesos_clean.db`**, ubicada en `Data/database/Procesos_clean.db`.\n",
        "\n",
        "El objetivo es entender el comportamiento operativo de los procesos RPA, validar la "
        "calidad de los datos, convertir variables críticas a formatos analizables, detectar "
        "valores atípicos y generar conclusiones útiles para el proyecto.\n",
        "\n",
        "**Competencias cubiertas:**\n",
        "\n",
        "- Carga de datos desde SQLite.\n",
        "- Inspección automática de tablas disponibles.\n",
        "- Limpieza y normalización básica de columnas.\n",
        "- Conversión de duraciones a segundos y minutos.\n",
        "- Análisis estadístico descriptivo.\n",
        "- Tablas de frecuencia para variables categóricas.\n",
        "- Visualización de distribuciones y relaciones.\n",
        "- Detección de outliers con método IQR.\n",
        "- Interpretación y conclusiones por sección.",
    ],
    "## 2. Validación y Carga de la Base de Datos": [
        "## 2. Validación y Carga de la Base de Datos\n",
        "\n",
        "Se busca automáticamente el archivo `Procesos_clean.db` en rutas relativas al "
        "directorio de trabajo, garantizando compatibilidad independientemente del entorno "
        "de ejecución.",
    ],
    "## 3. Inspección de Tablas Disponibles": [
        "## 3. Inspección de Tablas Disponibles\n",
        "\n",
        "Se consultan las tablas disponibles en la base de datos antes de cargar los datos, "
        "permitiendo una selección dinámica y robusta de la fuente de información.",
    ],
    "## 4. Carga de Datos": [
        "## 4. Carga de Datos\n",
        "\n",
        "Se carga la tabla `RegistrosDPA_clean`, que contiene los registros operativos de "
        "ejecución de los procesos RPA. Si la tabla preferida no estuviera disponible, el "
        "proceso selecciona automáticamente la primera tabla encontrada en la base de datos.",
    ],
    "## 5. Exploración Inicial: Estructura y Calidad de Datos": [
        "## 5. Exploración Inicial: Estructura y Calidad de Datos\n",
        "\n",
        "Se revisan los tipos de datos, la presencia de valores nulos, filas duplicadas y "
        "los valores únicos por columna.\n",
        "\n",
        "**Conclusión:** El campo `Duracion_Ejecucion` presenta el **90.76% de valores "
        "nulos**, lo que limita el análisis de tiempos de ejecución a un subconjunto del "
        "9.24% de los registros. Los demás campos presentan completitud superior al 99%. "
        "No se encontraron filas duplicadas exactas.",
    ],
    "## 6. Funciones Auxiliares para Preparación de Variables": [
        "## 6. Funciones Auxiliares para Preparación de Variables\n",
        "\n",
        "Se definen funciones de utilidad para localizar columnas con variaciones en el "
        "nombre (ej. `Fecha_ejecucion`, `Fecha Ejecución`, `fecha`), convertir duraciones "
        "a segundos y detectar valores atípicos mediante el método IQR.",
    ],
    "## 7. Preparación de Variables para el Análisis": [
        "## 7. Preparación de Variables para el Análisis\n",
        "\n",
        "Se identifican automáticamente las columnas clave (automatización, duración, "
        "transacciones y fecha) y se crean variables derivadas: duración en "
        "segundos/minutos, transacciones numéricas y componentes temporales "
        "(año, mes, día de la semana, hora).\n",
        "\n",
        "**Conclusión:** Las variables de duración están disponibles solo para los registros "
        "con `Duracion_Ejecucion` no nula (~57,172 de 618,875 registros, equivalente al "
        "9.24%). Este hecho debe tenerse en cuenta al interpretar los análisis de tiempo "
        "de ejecución.",
    ],
    "## 8. Estadística Descriptiva": [
        "## 8. Estadística Descriptiva\n",
        "\n",
        "Resumen estadístico de las principales variables numéricas disponibles tras la "
        "preparación de datos.\n",
        "\n",
        "**Conclusión:** La mediana de transacciones es **1** (la media de 15.28 está "
        "inflada por valores extremos). La mediana de duración es ~10 minutos, mientras "
        "la media supera los 38 minutos, evidenciando una distribución muy asimétrica. "
        "El campo `ErroresNegocio` tiene mediana 0 pero promedio de 431.71, indicando "
        "que los errores son raros pero pueden ser masivos cuando ocurren.",
    ],
    "## 9. Frecuencias de Variables Categóricas": [
        "## 9. Frecuencias de Variables Categóricas\n",
        "\n",
        "Se muestran las categorías más frecuentes para las variables de texto con "
        "cardinalidad razonable (≤ 30 valores únicos).\n",
        "\n",
        "**Conclusión:** El **94.3%** de las ejecuciones pertenece a \"Servicios "
        "Organizacionales\". El área de **Afiliación** concentra el 90.6% de los "
        "registros. El **96.2%** de las ejecuciones son de tipo Transaccional (T). "
        "La actividad se distribuye toda la semana, con mayor volumen entre martes y "
        "viernes (~100K–112K ejecuciones/día) y menor actividad el domingo (~40K).",
    ],
    "## 10. Detección de Valores Atípicos (Outliers) — Método IQR": [
        "## 10. Detección de Valores Atípicos (Outliers) — Método IQR\n",
        "\n",
        "Se identifican valores atípicos en duración y número de transacciones usando el "
        "método del rango intercuartílico (IQR). Estos casos no son necesariamente errores: "
        "pueden representar ejecuciones con alto volumen, condiciones especiales de "
        "operación o procesos intensivos por naturaleza.\n",
        "\n",
        "**Conclusión:** El **1.09%** de los registros con duración supera el límite de "
        "~67 minutos (4,027.5 seg). El **4.95%** de los registros tiene más de 1 "
        "transacción. Los outliers en duración corresponden a procesos legítimamente "
        "complejos (`GestorRemitidos`, `Vacunacion tradicional`) que deben monitorearse "
        "de forma diferenciada.",
    ],
    "## 11. Visualizaciones Principales": [
        "## 11. Visualizaciones Principales\n",
        "\n",
        "Las gráficas a continuación permiten interpretar la distribución, dispersión y "
        "comportamiento por proceso de las variables clave.\n",
        "\n",
        "**Conclusión:** Los histogramas confirman la alta asimetría de las distribuciones: "
        "la mayoría de ejecuciones son cortas y unitarias, pero existe un subconjunto de "
        "procesos intensivos en tiempo y volumen. Esta bimodalidad sugiere la necesidad de "
        "segmentar el análisis según el tipo de proceso.",
    ],
    "## 13. Análisis Temporal": [
        "## 13. Análisis Temporal\n",
        "\n",
        "Se resume la evolución mensual de ejecuciones, duración promedio y transacciones "
        "promedio para identificar tendencias de crecimiento y estacionalidad.\n",
        "\n",
        "**Conclusión:** Se observa un **crecimiento exponencial** desde 2021 hasta 2024. "
        "En los primeros meses (2021), los procesos eran pocos pero muy intensivos. "
        "A medida que aumentaron los procesos (2022+), la duración promedio disminuyó y "
        "el volumen se estabilizó, evidenciando una maduración del programa RPA.",
    ],
    "## 14. Ranking de Procesos por Duración y Volumen": [
        "## 14. Ranking de Procesos por Duración y Volumen\n",
        "\n",
        "Se identifican los procesos con mayor duración promedio y mayor volumen operativo, "
        "con el fin de priorizar candidatos de optimización y detectar cuellos de botella.\n",
        "\n",
        "**Conclusión:** \"Correcciones vacunacion\" es el proceso con mayor duración "
        "promedio (~8.5 horas) y el mayor volumen por ejecución (~22,276 transacciones). "
        "\"GestorRemitidos\" combina alta frecuencia (1,707 ejecuciones) con larga duración "
        "(~7.5 horas promedio), siendo el proceso con mayor impacto operativo acumulado "
        "(1.68 millones de transacciones totales). Estos procesos son prioritarios para "
        "análisis de eficiencia.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. CELDA DE CÓDIGO → MARKDOWN (conclusiones finales con print)
#    Se busca por una firma única del código fuente.
# ─────────────────────────────────────────────────────────────────────────────
CODE_TO_MARKDOWN_SIGNATURES = [
    "conclusiones = []",   # firma de la celda de código a reemplazar
]

# Contenido Markdown que reemplazará esa celda de código
CONCLUSIONS_MARKDOWN = [
    "## 15. Conclusiones Generales del Análisis\n",
    "\n",
    "A continuación se presentan los hallazgos más relevantes del EDA sobre los "
    "procesos RPA registrados en `Procesos_clean.db`:\n",
    "\n",
    "---\n",
    "\n",
    "**1. Volumen total del dataset**\n",
    "\n",
    "Se cargaron **618,875 registros** desde la tabla `RegistrosDPA_clean`. "
    "Este volumen refleja la escala operativa del programa RPA en la organización.\n",
    "\n",
    "---\n",
    "\n",
    "**2. Duración de las ejecuciones**\n",
    "\n",
    "La duración promedio de las ejecuciones fue de **2,321.25 segundos** (~38.7 minutos), "
    "mientras que la mediana fue de **603.00 segundos** (~10 minutos). "
    "La diferencia significativa entre media y mediana indica la presencia de ejecuciones "
    "atípicamente largas que elevan el promedio. La mayoría de los procesos son cortos, "
    "pero un subconjunto intensivo distorsiona la distribución.\n",
    "\n",
    "---\n",
    "\n",
    "**3. Volumen de transacciones**\n",
    "\n",
    "El volumen promedio procesado fue de **15.28 transacciones por ejecución**. "
    "Sin embargo, la mediana es 1, lo que confirma que la mayoría de ejecuciones "
    "procesan un solo ítem. Esta variable es clave para explicar diferencias en "
    "tiempos de procesamiento entre ejecuciones.\n",
    "\n",
    "---\n",
    "\n",
    "**4. Valores atípicos (outliers) detectados**\n",
    "\n",
    "| Variable | Outliers | % del total |\n",
    "|---|---|---|\n",
    "| `Duracion_Segundos` | **6,761** | **1.09%** |\n",
    "| `Numero_Transacciones_Num` | **30,663** | **4.95%** |\n",
    "\n",
    "Estos casos no deben eliminarse automáticamente: pueden representar ejecuciones "
    "reales de alta complejidad o procesos con volúmenes excepcionalmente elevados. "
    "Se recomienda revisarlos individualmente antes de excluirlos del análisis.\n",
    "\n",
    "---\n",
    "\n",
    "**5. Correlación duración — transacciones**\n",
    "\n",
    "La correlación entre duración y número de transacciones fue de **0.21** "
    "(correlación débil positiva). Esto indica que procesar más transacciones no "
    "garantiza una ejecución más larga. La complejidad del proceso, el tipo de "
    "automatización y las condiciones del sistema explican mejor las variaciones "
    "en tiempo de ejecución.\n",
    "\n",
    "---\n",
    "\n",
    "**6. Recomendaciones**\n",
    "\n",
    "- Investigar el origen del **90.76% de nulos** en `Duracion_Ejecucion` para "
    "mejorar la cobertura del análisis de tiempos.\n",
    "- Priorizar la revisión de **Correcciones vacunacion** y **GestorRemitidos** "
    "como candidatos a optimización.\n",
    "- Implementar monitoreo diferenciado para procesos con outliers frecuentes "
    "en duración (`Duracion_Segundos`).\n",
    "- Considerar la segmentación del análisis entre procesos de tipo **T** "
    "(Transaccional) y **E** (Especial), dado su comportamiento diferenciado.",
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def replace_markdown_cells(cells: list, replacements: dict) -> int:
    """Reemplaza el contenido de celdas Markdown existentes."""
    count = 0
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        full_text = "".join(cell.get("source", []))
        for key, new_lines in replacements.items():
            if key in full_text:
                cell["source"] = new_lines
                count += 1
                break
    return count


def convert_code_to_markdown(cells: list, signatures: list, new_source: list) -> int:
    """
    Busca celdas de CÓDIGO que contengan alguna de las firmas dadas
    y las convierte en celdas MARKDOWN con new_source como contenido.
    """
    count = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        full_text = "".join(cell.get("source", []))
        if any(sig in full_text for sig in signatures):
            cell["cell_type"] = "markdown"
            cell["source"] = new_source
            # Las celdas markdown no tienen outputs ni execution_count
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
            count += 1
    return count


def main():
    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: No se encontró el notebook en:\n  {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])

    n_md = replace_markdown_cells(cells, MARKDOWN_REPLACEMENTS)
    n_code = convert_code_to_markdown(
        cells, CODE_TO_MARKDOWN_SIGNATURES, CONCLUSIONS_MARKDOWN
    )

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("✔ Notebook actualizado correctamente.")
    print(f"  Celdas Markdown reemplazadas : {n_md}")
    print(f"  Celdas de código → Markdown  : {n_code}")
    print(f"  Ruta: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
