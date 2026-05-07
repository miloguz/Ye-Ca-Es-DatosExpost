# Proyecto 1 — Especialización en Ciencia de Datos e IA
## Análisis de Procesos RPA con Agente SQL Local y Predicción de ROI

Pipeline completo de datos para el análisis del portafolio de automatizaciones RPA de una organización de salud en Colombia. Incluye preprocesamiento, EDA enfocado en ROI, modelo predictivo de retorno de inversión y una interfaz de chat SQL 100% local impulsada por un LLM corriendo en Ollama.

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Funcionalidades](#funcionalidades)
- [Datos](#datos)
- [Modelo de ROI](#modelo-de-roi)

---

## Descripción

El proyecto integra tres fuentes de datos sobre automatizaciones RPA (`RegistrosDPA`, `TiemposManuales`, `RolesAreas`) para responder tres preguntas clave:

1. **¿Cuánto valor generan los bots actuales?** — Cálculo de ROI real por automatización.
2. **¿Qué factores determinan el ROI?** — EDA correlacional y segmentación por cuadrantes.
3. **¿Cuánto ROI generará un nuevo bot antes de implementarlo?** — Modelo GBM predictivo.

Todo esto se expone a través de una interfaz de chat en lenguaje natural donde el usuario puede hacer preguntas sobre los datos sin conocer SQL.

---

## Arquitectura

```
Usuario (pregunta en español)
        │
        ▼
   Streamlit UI (app/chat.py)
        │
        ▼
   Agente SQL (src/agent/sql_agent.py)
        │
    ┌───┴────────────────────┐
    ▼                        ▼
Ollama LLM              SQLite DB
(qwen2.5-coder:7b)    (Procesos_clean.db)
    │                        │
    └───────┬────────────────┘
            ▼
    Respuesta en español + tabla de resultados
```

---

## Estructura del Proyecto

```
Proyecto1Especializacion/
├── app/
│   └── chat.py                  ← interfaz Streamlit (Chat SQL / ROI / Predicción)
├── src/
│   ├── agent/
│   │   ├── database.py          ← conexión SQLite y esquema para el LLM
│   │   └── sql_agent.py         ← agente SQL con Ollama (generar SQL + interpretar)
│   ├── models/
│   │   └── roi_predictor.py     ← pipeline GBM con log-transform del target
│   └── utils/
│       └── roi_calculator.py    ← cálculo de ROI desde las tres tablas
├── notebooks/
│   ├── 01_preprocesamiento.ipynb    ← limpieza y normalización de datos
│   ├── 02_eda_roi.ipynb             ← EDA completo enfocado en ROI
│   └── 03_modelo_roi.ipynb          ← entrenamiento y evaluación del modelo
├── scripts/
│   └── csv_to_sqlite.py         ← convierte CSVs fuente a Procesos.db
├── data/
│   ├── raw/                     ← archivos CSV originales (Git LFS)
│   └── database/
│       ├── Procesos.db          ← BD original (Git LFS)
│       └── Procesos_clean.db    ← BD limpia y preprocesada (Git LFS)
├── models/                      ← modelo entrenado roi_model.joblib (generado localmente)
├── reports/figures/             ← graficas generadas por los notebooks
├── .streamlit/config.toml       ← configuración headless de Streamlit
├── .claude/launch.json          ← configuración de servidores de desarrollo
├── pyproject.toml
└── run_app.bat                  ← acceso rapido en Windows
```

---

## Requisitos Previos

| Herramienta | Version minima | Notas |
|---|---|---|
| Python | **3.11** | Verificar con `python --version` |
| Ollama | **0.23+** | Descargar en https://ollama.com |
| RAM | **8 GB** | 16 GB recomendado |
| GPU (opcional) | VRAM >= 6 GB | Acelera el LLM significativamente |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto1Especializacion
```

### 2. Instalar dependencias Python

```bash
pip install streamlit ollama plotly xgboost joblib pandas scikit-learn seaborn matplotlib numpy
```

O con `uv` (recomendado):

```bash
pip install uv
uv sync
```

### 3. Instalar y configurar Ollama

```bash
# Descargar desde https://ollama.com/download
# Luego descargar el modelo:
ollama pull qwen2.5-coder:7b
```

### 4. Preparar la base de datos

Si es la primera vez, ejecutar los notebooks en orden:

```
notebooks/01_preprocesamiento.ipynb   ← genera Procesos_clean.db
notebooks/02_eda_roi.ipynb            ← EDA y genera data/roi_dataset.csv
notebooks/03_modelo_roi.ipynb         ← entrena y guarda models/roi_model.joblib
```

---

## Ejecución

```bash
# Opcion 1 — Script rapido (Windows)
run_app.bat

# Opcion 2 — Manual
ollama serve                        # terminal 1 (si no esta corriendo)
streamlit run app/chat.py           # terminal 2
```

La app queda disponible en **http://localhost:8501**

---

## Funcionalidades

### Pestana 1 — Chat SQL
Haz preguntas en español sobre los datos. El agente genera SQL, lo ejecuta y responde en lenguaje natural.

Ejemplos:
- *"¿Cuales son los 5 bots con mas ejecuciones?"*
- *"¿Que area tiene mayor tasa de error?"*
- *"¿Cuanto tiempo manual ahorro GestorRemitidos en total?"*
- *"¿Cual es el ROI promedio de los bots activos de UiPath?"*

### Pestana 2 — Analisis ROI
Visualizaciones interactivas del portafolio:
- Ranking de bots por ROI (%) y ahorro neto (COP)
- Scatter ROI vs volumen de ejecuciones por tecnologia
- Tabla detallada con metricas por automatizacion

### Pestana 3 — Prediccion ROI
Estima el ROI de un nuevo bot antes de implementarlo ingresando:
- Tiempo manual por ejecucion (horas)
- Ejecuciones esperadas
- Valor hora del rol (COP)
- Tecnologia RPA y duracion estimada del robot

---

## Datos

| Tabla | Filas | Descripcion |
|---|---|---|
| `RegistrosDPA_clean` | 618,875 | Registros historicos de ejecucion de bots |
| `TiemposManuales_clean` | 95 | Tiempo manual que reemplaza cada bot |
| `RolesAreas_clean` | 91 | Roles y areas impactadas con valor/hora |

Periodo cubierto: 2020 – 2024. Los archivos de base de datos estan almacenados en **Git LFS**.

---

## Modelo de ROI

**Formula de calculo:**

```
Beneficio_Bruto = TiempoManual x ValorHora x NumEjecuciones
Costo_Robot     = DuracionRobot x ValorHora x 0.25 x NumEjecuciones
Ahorro_Neto     = Beneficio_Bruto - Costo_Robot
ROI%            = (Ahorro_Neto / Costo_Robot) x 100
```

**Modelo predictivo:** XGBoostRegressor con aceleracion GPU (CUDA) y transformacion `log1p(ROI)` para manejar la distribucion sesgada del target. Detecta automaticamente GPU NVIDIA y hace fallback a CPU.

| Metrica | Valor |
|---|---|
| Algoritmo | XGBoost 2.x (`tree_method=hist`, `device=cuda`) |
| R2 (escala log, CV 5-fold) | ~0.41 |
| Top features | TiempoManualHoras, TasaExito, DuracionPromedio, ValorHora |
| GPU requerida | RTX 4060 (8 GB VRAM) recomendada, fallback a CPU automatico |
| Muestras de entrenamiento | ~33 bots con datos completos |

> El modelo mejora a medida que se completen los datos de `TiemposManuales` para mas bots.
