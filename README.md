# Proyecto 1 — Especialización en Ciencia de Datos e IA
## Sinfama · Análisis de Procesos RPA con Agente SQL Local y Calculadora de ROI

Pipeline completo de datos para el análisis del portafolio de automatizaciones RPA de una caja de compensación familiar (caso de estudio). Incluye preprocesamiento, EDA enfocado en ROI, calculadora determinística de ROI con costo operativo estandarizado y una interfaz de chat SQL 100% local impulsada por dos modelos de IA corriendo localmente: **Qwen 2.5 Coder** (LLM via Ollama) para text-to-SQL y **Whisper** (faster-whisper) para entrada por voz en español.

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Especificaciones de Hardware](#especificaciones-de-hardware)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Solución de Problemas](#solución-de-problemas)
- [Funcionalidades](#funcionalidades)
- [Datos](#datos)
- [Modelo de ROI](#modelo-de-roi)

---

## Descripción

El proyecto integra tres fuentes de datos sobre automatizaciones RPA (`RegistrosDPA`, `TiemposManuales`, `RolesAreas`) para responder tres preguntas clave:

1. **¿Cuánto valor generan los bots actuales?** — Cálculo de ROI real por automatización.
2. **¿Qué factores determinan el ROI?** — EDA correlacional y segmentación por cuadrantes.
3. **¿Cuánto ROI generará un nuevo bot antes de implementarlo?** — Calculadora basada en la fórmula del negocio (auditable, sin modelo de caja negra).

Todo esto se expone a través de una interfaz de chat en lenguaje natural donde el usuario puede hacer preguntas sobre los datos sin conocer SQL.

---

## Arquitectura

```
Usuario (voz 🎤 o texto en español)
        │
        ▼
   Streamlit UI (app/chat.py)
        │
   ┌────┴────────────┐
   ▼                 │  (texto directo)
Whisper-small        │
(faster-whisper,     │
 CPU, local)         │
   │                 │
   └────────┬────────┘
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

**Dos modelos de IA, 100% locales:**
- **Whisper-small** transcribe la voz del usuario a texto en español (~480 MB, corre en CPU).
- **Qwen 2.5 Coder 7B** convierte el texto en SQL, lo ejecuta y narra la respuesta (~5 GB, corre en GPU si está disponible).

---

## Estructura del Proyecto

```
Proyecto1Especializacion/
├── app/
│   └── chat.py                  ← interfaz Streamlit (Chat SQL / ROI / Predicción)
├── assets/
│   └── logo.svg                 ← logo Sinfama (paleta de marca)
├── src/
│   ├── agent/
│   │   ├── database.py          ← conexión SQLite y esquema para el LLM
│   │   └── sql_agent.py         ← agente SQL con Ollama (generar SQL + interpretar)
│   ├── models/
│   │   └── roi_predictor.py     ← pipeline XGBoost con log-transform del target
│   └── utils/
│       └── roi_calculator.py    ← cálculo de ROI desde las tres tablas
├── notebooks/
│   ├── 01_preprocesamiento.ipynb    ← limpieza y normalización de datos
│   ├── 02_eda_roi.ipynb             ← EDA completo enfocado en ROI
│   └── 03_modelo_roi.ipynb          ← entrenamiento y evaluación del modelo
├── scripts/
│   ├── csv_to_sqlite.py         ← convierte CSVs fuente a Procesos.db
│   ├── train_roi_model.py       ← pipeline reproducible (db → dataset → train → save)
│   └── download_whisper.py      ← pre-descarga el modelo Whisper-small (~480 MB)
├── data/
│   ├── raw/                     ← archivos CSV originales (Git LFS)
│   └── database/
│       ├── Procesos.db          ← BD original (Git LFS)
│       └── Procesos_clean.db    ← BD limpia y preprocesada (Git LFS)
├── models/                      ← roi_model.joblib (generado por train_roi_model.py)
├── reports/                     ← figuras (notebooks) y métricas JSON (pipeline)
├── .streamlit/config.toml       ← configuración headless de Streamlit
├── PITCH.md                     ← presentación académica del proyecto
├── pyproject.toml
└── run_app.bat                  ← acceso rápido en Windows
```

---

## Requisitos Previos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| uv | **0.4+** | Gestor de entornos y dependencias. Instalar con `pip install uv` o desde https://docs.astral.sh/uv/ |
| Python | **3.11** | uv puede instalarlo automáticamente (`uv python install 3.11`) |
| Ollama | **0.23+** | Descargar en https://ollama.com |

---

## Especificaciones de Hardware

El cuello de botella es el LLM local. La app usa dos modelos:

- **`qwen2.5-coder:7b`** en Ollama (cuantización Q4_K_M, ~4,7 GB disco, ~5,5 GB de memoria activa).
- **`whisper-small`** en faster-whisper (cuantización int8, ~480 MB disco, ~1 GB RAM al transcribir).

### Mínimos para correr la aplicación

| Recurso | Valor mínimo | Notas |
|---|---|---|
| RAM | **16 GB** | LLM + Whisper + Streamlit + Python + SO. Con 8 GB hay swap agresivo y la app se vuelve inusable. |
| CPU | 8 núcleos modernos (i7 / Ryzen 7) | En modo CPU se obtienen **~8–10 tokens/s** del LLM — cada respuesta del agente puede tardar 20–60 s. Whisper transcribe ~1 s por cada 5 s de audio. |
| Disco | **16 GB libres** | ~5 GB Qwen + ~480 MB Whisper + dependencias Python + bases de datos en LFS. Idealmente SSD. |
| GPU | Opcional (no requerida) | Si hay GPU NVIDIA con ≥ 6 GB VRAM, Ollama y XGBoost la usan automáticamente. Whisper queda en CPU para no competir por VRAM con el LLM. |
| Micrófono | Opcional | Solo necesario si quieres usar la entrada por voz. La app funciona perfectamente solo con texto. |
| Navegador | Chrome / Edge / Firefox actualizado | Requerido para el botón de micrófono (usa la API `MediaRecorder`). |

### Verificar tu máquina antes de instalar

```bash
# Ver RAM total (Windows PowerShell)
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory

# Ver GPU NVIDIA disponible y VRAM
nvidia-smi
```

Si `nvidia-smi` no está instalado o no detecta GPU, la app igual funciona en CPU — solo será más lenta para el chat y el entrenamiento de XGBoost hará fallback automático a CPU.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto1Especializacion
```

### 2. Instalar dependencias con uv

```bash
uv sync
```

Esto crea un entorno virtual aislado en `.venv/` e instala todas las dependencias declaradas en `pyproject.toml` / `uv.lock`.

### 3. Instalar y configurar Ollama

```bash
# Descargar desde https://ollama.com/download
# Luego descargar el modelo:
ollama pull qwen2.5-coder:7b
```

### 4. Descargar el modelo de transcripción de voz (~480 MB)

```bash
uv run python scripts/download_whisper.py
```

Solo es necesario una vez por máquina. El modelo Whisper-small queda cacheado en `~/.cache/huggingface/` y se reutiliza en todas las ejecuciones futuras. Habilita el botón de micrófono en la pestaña de Chat SQL para hacer preguntas por voz.

### 5. Preparar la base de datos y entrenar el modelo

**Opción A — pipeline reproducible (recomendado para re-entrenamiento):**

```bash
uv run python scripts/train_roi_model.py
```

Construye el dataset desde `Procesos_clean.db`, entrena XGBoost (GPU si hay CUDA, fallback CPU automático) y guarda el modelo + métricas en `reports/metrics_roi.json`.

**Opción B — notebooks paso a paso (recomendado la primera vez para entender el flujo):**

```bash
uv run jupyter lab
```

Luego abrir en orden:

```
notebooks/01_preprocesamiento.ipynb   ← genera Procesos_clean.db
notebooks/02_eda_roi.ipynb            ← EDA y genera data/roi_dataset.csv
notebooks/03_modelo_roi.ipynb         ← entrena y guarda models/roi_model.joblib
```

---

## Ejecución

> **Antes de ejecutar:** verifica que Ollama esté corriendo. Abre otra terminal y ejecuta `ollama list` — si responde con la lista de modelos, el servicio está activo. Si no, inícialo con `ollama serve` (en Windows, la app de Ollama suele dejarlo corriendo en segundo plano automáticamente).

```bash
uv run streamlit run app/chat.py
```

La app queda disponible en **http://localhost:8501**

### Permisos del navegador

La primera vez que aprietes el botón **🎤 Grabar pregunta** en la pestaña Chat SQL, el navegador te pedirá permiso para usar el micrófono. **Acepta** para habilitar la entrada por voz. Si lo rechazaste por accidente, puedes habilitarlo después desde el ícono del candado en la barra de direcciones.

> **Nota:** la API `MediaRecorder` solo funciona en `localhost` o sobre HTTPS. Si despliegas la app en otra máquina por IP en HTTP, el botón de micrófono fallará silenciosamente — usa siempre `localhost` o configura HTTPS.

---

## Solución de Problemas

### `uv sync` falla con error de Python

uv intenta usar la versión más reciente de Python que encuentre. Si tienes Python 3.13 o 3.14 instalado pero algunas dependencias (sklearn, pandas) aún no soportan esa versión, fija explícitamente Python 3.11:

```bash
uv sync --python 3.11
```

uv descargará Python 3.11 si no lo tienes.

### `ollama list` devuelve vacío o el chat tira `connection refused`

Ollama no está corriendo. Inicia el servicio:

```bash
ollama serve
```

En Windows, si instalaste Ollama con el instalador oficial, suele iniciarse automáticamente al arrancar el equipo. Verifica el ícono en la bandeja del sistema. Y comprueba que el modelo está descargado: `ollama list` debe mostrar `qwen2.5-coder:7b`.

### El botón 🎤 de micrófono no aparece o no graba

- **No aparece:** verifica que la última instalación corrió `uv sync` después de actualizar `pyproject.toml` (debe instalar `streamlit-mic-recorder`).
- **Aparece pero no graba:** revisa permisos del navegador. Click en el ícono del candado en la barra de direcciones → permitir micrófono. Recarga la página.
- **Graba pero no transcribe:** ejecuta `uv run python scripts/download_whisper.py` para asegurarte de que el modelo Whisper-small está descargado. Si la descarga falló por conexión, vuelve a correrlo.

### La primera grabación tarda mucho

Es normal — Streamlit carga el modelo Whisper en RAM la primera vez (~3 s adicionales). A partir de la segunda grabación queda cacheado y la transcripción es de ~1 s para audios de 5 s.

### `nvidia-smi` no se reconoce o XGBoost se queja de CUDA

No tienes drivers NVIDIA o no tienes GPU NVIDIA. **No es un problema** — la app detecta esto y hace fallback automático a CPU para Ollama y XGBoost. Solo serán más lentos.

---

## Funcionalidades

### Pestaña 1 — Chat SQL
Haz preguntas en español sobre los datos por **texto o por voz**. El agente genera SQL, lo ejecuta y responde en lenguaje natural.

**Por texto:** escribe la pregunta en el campo inferior y presiona Enter.

**Por voz:** presiona **🎤 Grabar pregunta**, di la pregunta, presiona **⏹️ Detener**. La app transcribe el audio localmente con Whisper y envía el texto al agente automáticamente.

Ejemplos:
- *"¿Cuáles son los 5 bots con más ejecuciones?"*
- *"¿Qué área tiene mayor tasa de error?"*
- *"¿Cuánto tiempo manual ahorró GestorRemitidos en total?"*
- *"¿Cuál es el ROI promedio de los bots activos de UiPath?"*

### Pestaña 2 — Análisis ROI
Visualizaciones interactivas del portafolio:
- Ranking de bots por ROI (%) y ahorro neto (COP)
- Scatter ROI vs volumen de ejecuciones por tecnología
- Tabla detallada con métricas por automatización

### Pestaña 3 — Calculadora de ROI
Calcula el ROI esperado de un nuevo proceso RPA aplicando la fórmula del negocio (no usa modelo predictivo, es matemática auditable). Solo requiere 4 entradas:
- Tiempo manual por ejecución (horas)
- Ejecuciones esperadas (total)
- Valor hora del rol humano (COP)
- Duración estimada del robot (horas)

El costo operativo del robot es **fijo en 7.300 COP/hora** (servidor Azure + licencia UiPath). La pestaña muestra ROI%, ahorro neto, beneficio bruto y costo total, junto con un expander que detalla la fórmula aplicada paso a paso.

---

## Datos

| Tabla | Filas | Descripción |
|---|---|---|
| `RegistrosDPA_clean` | 618,875 | Registros históricos de ejecución de bots |
| `TiemposManuales_clean` | 95 | Tiempo manual que reemplaza cada bot |
| `RolesAreas_clean` | 91 | Roles y áreas impactadas con valor/hora |

Período cubierto: 2020 – 2024. Los archivos de base de datos están almacenados en **Git LFS**.

---

## Modelo de ROI

**Fórmula de cálculo:**

```
Beneficio_Bruto = TiempoManual x ValorHora x NumEjecuciones
Costo_Robot     = DuracionRobot x 7300 x NumEjecuciones
Ahorro_Neto     = Beneficio_Bruto - Costo_Robot
ROI%            = (Ahorro_Neto / Costo_Robot) x 100
```

**Costo operativo del robot — 7.300 COP/hora (estándar):**

| Concepto | Valor mensual (COP) |
|---|---|
| Servidor Azure | 150.000 |
| Licencia UiPath robot + orquestador | 5.200.000 |
| **Total mensual** | **5.350.000** |
| Horas/mes (730 h ≈ 24 × 30,4) | ÷ 730 |
| **Costo por hora de robot** | **≈ 7.300 COP/h** |

Este valor reemplaza el factor antiguo (`ValorHora × 0,25`) y se aplica de forma uniforme a todas las soluciones, haciendo comparables los ROIs entre proyectos sin importar el rol que cada bot reemplace.

**Modelo predictivo:** XGBoostRegressor con aceleración GPU (CUDA) y transformación `log1p(ROI)` para manejar la distribución sesgada del target. Detecta automáticamente GPU NVIDIA y hace fallback a CPU.

| Métrica | Valor |
|---|---|
| Algoritmo | XGBoost 2.x (`tree_method=hist`, `device=cuda`) |
| R² (escala log, test split) | ~0.81 |
| R² CV 5-fold (media ± std) | 0.03 ± 0.27 (alta varianza por n pequeño) |
| MAE (escala original) | ~2.291% |
| Top features | TiempoManualHoras, TasaExito, DuracionPromedio, ValorHora |
| GPU requerida | RTX 4060 (8 GB VRAM) recomendada, fallback a CPU automático |
| Muestras de entrenamiento | ~30 bots con datos completos |

> El modelo mejora a medida que se completen los datos de `TiemposManuales` para más bots.
