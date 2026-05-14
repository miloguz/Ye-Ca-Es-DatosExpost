<p align="center">
  <img src="assets/logo.svg" width="220" alt="Sinfama" />
</p>

# Proyecto 1 — Especialización en Ciencia de Datos e IA

## Equipo

Proyecto desarrollado para la materia **proyecto 1**.

- Camilo Guzmán
- Yennifer Serna
- Esneyder Gómez

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
- [Síntesis de Voz (TTS)](#síntesis-de-voz-tts)
- [Datos](#datos)
- [Cálculo de ROI](#cálculo-de-roi)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Equipo](#equipo)

---

## Descripción

El proyecto integra tres fuentes de datos sobre automatizaciones RPA (`RegistrosDPA`, `TiemposManuales`, `RolesAreas`) para responder tres preguntas clave:

1. **¿Cuánto valor generan los bots actuales?** — Cálculo de ROI real por automatización.
2. **¿Qué factores determinan el ROI?** — EDA correlacional y segmentación por cuadrantes.
3. **¿Cuánto ROI generará un nuevo bot antes de implementarlo?** — Calculadora basada en la fórmula del negocio (auditable, sin modelo de caja negra).

### Modelos de IA utilizados

El proyecto integra **tres componentes de IA** para lograr una experiencia conversacional completa:

- **Qwen 2.5 Coder 7B** (vía Ollama) — LLM especializado en código que convierte preguntas en lenguaje natural a consultas SQL, las ejecuta sobre la base de datos SQLite y narra la respuesta en español. Corre **100% local**.
- **Whisper-small** (vía faster-whisper) — Modelo de transcripción de voz a texto en español, optimizado para CPU. Corre **100% local**.
- **Piper TTS** (`es_MX-ald-medium`) — Síntesis de voz neuronal **100% local** con voz en español latinoamericano. Se activa automáticamente si descargas el modelo ONNX vía `scripts/download_piper.py`.
- **Edge TTS** (`es-CO-SalomeNeural`) — Fallback en la nube con voz colombiana femenina si Piper no está disponible. Requiere internet (ver [Síntesis de Voz (TTS)](#síntesis-de-voz-tts)).

### Interfaz de chat con preguntas directas

Para que el usuario pueda hacer preguntas directas sobre los datos sin conocer SQL, se implementó un **agente SQL** (`src/agent/sql_agent.py`) que:

1. Recibe la pregunta del usuario en español (texto o voz).
2. Le entrega al LLM el esquema de la base de datos limpia (`Procesos_clean.db`) como contexto.
3. El LLM genera la consulta SQL apropiada, que se ejecuta sobre SQLite.
4. El resultado se devuelve en formato tabla **y** se acompaña de una explicación en lenguaje natural generada por el mismo LLM.

Adicionalmente, se integra **Whisper** para entradas por voz: el usuario presiona el botón 🎤, dicta su pregunta en español, y faster-whisper la transcribe localmente antes de enviarla al agente SQL — útil para hacer consultas rápidas sin escribir.

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
                │
                ▼
        ¿Modelo Piper local disponible?
          ┌─────┴─────┐
         sí           no
          │           │
          ▼           ▼
   Piper TTS     Edge TTS (nube)
   (ONNX local)  (Microsoft)
          │           │
          └─────┬─────┘
                ▼
      🔊 Reproducción de audio en el navegador
```

**Componentes de IA:**
- **Whisper-small** transcribe la voz del usuario a texto en español (~480 MB, corre en CPU, **local**).
- **Qwen 2.5 Coder 7B** convierte el texto en SQL, lo ejecuta y narra la respuesta (~4,7 GB, corre en GPU si está disponible, **local**).
- **Piper TTS** (`es_MX-ald-medium`) sintetiza la respuesta en CPU sin internet (~60 MB, **local**). Se activa si descargas el modelo ONNX.
- **Edge TTS** (`es-CO-SalomeNeural`) es el fallback en la nube cuando Piper no está descargado. Requiere internet.

---

## Estructura del Proyecto

```
Proyecto1Especializacion/
├── app/
│   └── chat.py                  ← interfaz Streamlit (Chat SQL / ROI / Calculadora)
├── assets/
│   └── logo.svg                 ← logo Sinfama (paleta de marca)
├── src/
│   ├── agent/
│   │   ├── database.py          ← conexión SQLite y esquema para el LLM
│   │   └── sql_agent.py         ← agente SQL con Ollama (generar SQL + interpretar)
│   ├── models/
│   │   └── roi_predictor.py     ← (experimental) pipeline XGBoost con log-transform
│   └── utils/
│       └── roi_calculator.py    ← cálculo de ROI desde las tres tablas
├── notebooks/
│   ├── 01_preprocesamiento.ipynb    ← limpieza y normalización de datos
│   ├── 02_eda_roi.ipynb             ← EDA completo enfocado en ROI
│   └── 03_modelo_roi.ipynb          ← (experimental) entrenamiento XGBoost
├── scripts/
│   ├── csv_to_sqlite.py         ← convierte CSVs fuente a Procesos.db
│   ├── train_roi_model.py       ← (experimental) pipeline XGBoost (db → train → save)
│   ├── download_whisper.py      ← pre-descarga el modelo Whisper-small (~480 MB)
│   └── download_piper.py        ← pre-descarga el modelo Piper TTS es_MX-ald-medium (~60 MB)
├── data/
│   ├── raw/                     ← archivos CSV originales (Git LFS)
│   └── database/
│       ├── Procesos.db          ← BD original (Git LFS)
│       └── Procesos_clean.db    ← BD limpia y preprocesada (Git LFS)
├── models/
│   ├── tts/                     ← modelo Piper TTS (generado por download_piper.py)
│   └── roi_model.joblib         ← (experimental) modelo XGBoost serializado
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
| Git | **2.30+** | Para clonar el repositorio. Descargar en https://git-scm.com |
| Git LFS | **3.0+** | Requerido para descargar las bases de datos del repositorio. Instalar con `winget install GitHub.GitLFS` / `brew install git-lfs` / `apt-get install git-lfs` |
| uv | **0.4+** | Gestor de entornos y dependencias. Instalar con `pip install uv` o desde https://docs.astral.sh/uv/ |
| Python | **3.11+** | Probado en 3.11 – 3.14. uv puede instalarlo automáticamente (`uv python install 3.11`) |
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
| Disco | **16 GB libres** | ~4,7 GB Qwen + ~480 MB Whisper + dependencias Python + bases de datos en LFS. Idealmente SSD. |
| GPU | Opcional (no requerida) | Si hay GPU NVIDIA con ≥ 6 GB VRAM, Ollama la usa automáticamente. Whisper queda en CPU para no competir por VRAM con el LLM. |
| Micrófono | Opcional | Solo necesario si quieres usar la entrada por voz. La app funciona perfectamente solo con texto. |
| Navegador | Chrome / Edge / Firefox actualizado | Requerido para el botón de micrófono (usa la API `MediaRecorder`). |

### Verificar tu máquina antes de instalar

```bash
# Ver RAM total (Windows PowerShell)
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory

# Ver GPU NVIDIA disponible y VRAM
nvidia-smi
```

Si `nvidia-smi` no está instalado o no detecta GPU, la app igual funciona en CPU — solo será más lenta para responder en el chat.

---

## Instalación

### 1. Instalar Git LFS

Las bases de datos (`Procesos.db`, `Procesos_clean.db`) y los CSVs crudos están almacenados en **Git LFS**, así que es necesario instalarlo **antes** de clonar el repositorio para que los archivos se descarguen correctamente.

```bash
# Windows (con winget)
winget install GitHub.GitLFS

# macOS (con Homebrew)
brew install git-lfs

# Linux (Debian/Ubuntu)
sudo apt-get install git-lfs
```

Luego, una sola vez por máquina, habilita LFS en tu usuario:

```bash
git lfs install
```

> Si ya clonaste el repositorio **antes** de instalar Git LFS, ejecuta `git lfs pull` dentro de la carpeta del proyecto para descargar los archivos faltantes.

### 2. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto1Especializacion
```

### 3. Instalar uv (si no lo tienes)

`uv` es el gestor de entornos y dependencias del proyecto. Si aún no lo tienes instalado:

```bash
pip install uv
```

> Alternativa oficial (independiente de Python): instaladores en [docs.astral.sh/uv](https://docs.astral.sh/uv/).

### 4. Instalar dependencias con uv

```bash
uv sync
```

Esto crea un entorno virtual aislado en `.venv/` e instala todas las dependencias declaradas en `pyproject.toml` / `uv.lock`.

### 5. Instalar y configurar Ollama

```bash
# Descargar desde https://ollama.com/download
# Luego descargar el modelo:
ollama pull qwen2.5-coder:7b
```

### 6. Descargar el modelo de transcripción de voz (~480 MB)

```bash
uv run python scripts/download_whisper.py
```

Solo es necesario una vez por máquina. El modelo Whisper-small queda cacheado en `~/.cache/huggingface/` y se reutiliza en todas las ejecuciones futuras. Sin este paso, el botón de micrófono aparece en la pestaña de Chat SQL pero la transcripción falla al usarlo.

### 6b. (Opcional pero recomendado) Descargar el modelo Piper TTS (~60 MB)

```bash
uv run python scripts/download_piper.py
```

Activa la **síntesis de voz local** para que la app pueda leer en voz alta las respuestas del agente sin internet. Si omites este paso, la app usa Edge TTS (Microsoft, en la nube) y necesita conexión.

### 7. Preparar la base de datos y entrenar el modelo experimental

> La BD limpia (`Procesos_clean.db`) y los datos crudos vienen incluidos en el repositorio vía **Git LFS**, por lo que la app puede correr inmediatamente después del paso 6 sin ejecutar este paso. Solo es necesario si quieres regenerar la BD desde los CSVs o experimentar con el modelo predictivo.

**Opción A — pipeline reproducible (Ejecución de notebooks):**

```bash
uv run python scripts/train_roi_model.py
```

Construye el dataset desde `Procesos_clean.db`, entrena XGBoost (GPU si hay CUDA, fallback CPU automático) y guarda el modelo + métricas en `reports/metrics_roi.json`.

**Opción B — notebooks paso a paso (recomendado la primera vez para entender el flujo):**

Antes de abrir los notebooks, **registra el kernel del proyecto en Jupyter** (una sola vez por máquina). Esto asegura que los notebooks usen el `.venv` con todas las dependencias instaladas, no un Python global donde pandas/xgboost/etc no existan:

```bash
uv run python -m ipykernel install --user --name sinfama-rpa --display-name "Python (Sinfama RPA · .venv)"
```

Luego levanta Jupyter Lab usando el entorno del proyecto:

```bash
uv run jupyter lab
```

Abre los notebooks en orden desde la interfaz de Jupyter:

```
notebooks/01_preprocesamiento.ipynb   ← genera Procesos_clean.db
notebooks/02_eda_roi.ipynb            ← EDA y genera data/roi_dataset.csv
notebooks/03_modelo_roi.ipynb         ← entrena y guarda models/roi_model.joblib
```

> **Importante:** la metadata de los notebooks ya apunta al kernel `sinfama-rpa`, por lo que Jupyter lo seleccionará automáticamente al abrirlos. Si ves un error tipo `ModuleNotFoundError: No module named 'pandas'`, casi seguro estás usando un kernel distinto — verifica en la esquina superior derecha del notebook que el kernel activo sea **"Python (Sinfama RPA · .venv)"**; si no, cámbialo desde el menú **Kernel → Change Kernel**.

> **Nota:** el modelo XGBoost (`03_modelo_roi.ipynb` y `scripts/train_roi_model.py`) es exploratorio y **no está integrado en la app**. La calculadora de ROI de la pestaña 3 usa exclusivamente la fórmula determinística.

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

El proyecto está probado en Python 3.11 – 3.14. Si una dependencia aún no soporta tu versión, fija explícitamente 3.11:

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

- **No aparece:** verifica que tu versión de Streamlit sea ≥ 1.31 (`uv pip show streamlit`). El componente de grabación es nativo (`st.audio_input`), no requiere paquetes adicionales.
- **Aparece pero no graba:** revisa permisos del navegador. Click en el ícono del candado en la barra de direcciones → permitir micrófono. Recarga la página.
- **Graba pero no transcribe:** ejecuta `uv run python scripts/download_whisper.py` para asegurarte de que el modelo Whisper-small está descargado. Si la descarga falló por conexión, vuelve a correrlo.

### La primera grabación tarda mucho

Es normal — Streamlit carga el modelo Whisper en RAM la primera vez (~3 s adicionales). A partir de la segunda grabación queda cacheado y la transcripción es de ~1 s para audios de 5 s.

### `nvidia-smi` no se reconoce

No tienes drivers NVIDIA o no tienes GPU NVIDIA. **No es un problema** — la app detecta esto y hace fallback automático a CPU para Ollama. Solo será más lento responder en el chat.

### El botón ▶ Escuchar respuesta no aparece

- Verifica que `edge-tts` esté instalado: `uv pip show edge-tts`. Si no, ejecuta `uv sync`.
- Revisa en la barra lateral de la app la sección **Voz TTS** — muestra si Edge TTS está disponible o el motivo de falla.

### Edge TTS falla o no reproduce audio

- **Sin internet:** Edge TTS requiere conexión a internet. Verifica la conectividad.
- **Firewall corporativo:** algunos firewalls bloquean el dominio `speech.platform.bing.com`. Si el entorno es corporativo sin salida a internet, Edge TTS no funcionará — considera integrar Piper (offline).
- **Error de biblioteca:** reinstala con `uv sync --upgrade`. En casos extremos: `uv pip install edge-tts --upgrade`.

### La descarga de Piper falla o es lenta

El script `download_piper.py` descarga desde HuggingFace (~60 MB). Si falla por conexión:

```bash
uv run python scripts/download_piper.py
```

Vuelve a correrlo — la descarga es idempotente (verifica si el archivo ya existe antes de descargar). Si el dominio HuggingFace está bloqueado, descarga manualmente los archivos `es_MX-ald-medium.onnx` y `es_MX-ald-medium.onnx.json` y colócalos en `models/tts/`.

---

## Funcionalidades

### Pestaña 1 — Chat SQL
Haz preguntas en español sobre los datos por **texto o por voz**. El agente genera SQL, lo ejecuta y responde en lenguaje natural.

**Por texto:** escribe la pregunta en el campo inferior y presiona Enter.

**Por voz:** presiona **🎤 Grabar pregunta**, di la pregunta, presiona **⏹️ Detener**. La app transcribe el audio localmente con Whisper y envía el texto al agente automáticamente.

**Escuchar la respuesta:** una vez que el agente responde, aparece el botón **▶ Escuchar respuesta**. Al pulsarlo, Edge TTS sintetiza la respuesta con voz colombiana (`es-CO-SalomeNeural`) y la reproduce en el navegador. El botón alterna entre ▶ y ❚❚ para pausar.

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

## Síntesis de Voz (TTS)

La app convierte las respuestas del agente SQL a audio con dos implementaciones posibles:

### TTS activo — Edge TTS (nube, por defecto)

La implementación actual usa **Microsoft Edge TTS** vía la librería `edge-tts` (incluida en `uv sync`). La voz por defecto es `es-CO-SalomeNeural` (colombiana femenina). Alternativa masculina: `es-CO-GonzaloNeural`.

| Aspecto | Detalle |
|---|---|
| Conectividad | **Requiere internet** (API de Microsoft) |
| Latencia | 1–3 s por respuesta |
| Calidad | Alta (voz neuronal de Microsoft) |
| Costo | Gratuito (uso razonable) |
| Configuración | Ninguna — se activa automáticamente al instalar `edge-tts` |

El estado de Edge TTS se muestra en la barra lateral de la app. Si `edge-tts` no está instalado, el botón de escuchar no aparece y la app funciona normalmente sin audio.

### TTS alternativo — Piper (local, sin internet)

El paquete `piper-tts` viene en las dependencias del proyecto (`uv sync`). Para activarlo, descarga el modelo de voz:

```bash
# Descargar el modelo Piper a models/tts/  (~62 MB)
uv run python scripts/download_piper.py
```

Una vez descargado, **la app usa Piper automáticamente** (prioridad sobre Edge TTS). El archivo se guarda en:
```
models/tts/
├── es_MX-claude-high.onnx
└── es_MX-claude-high.onnx.json
```

`es_MX-claude-high` es una voz **femenina latinoamericana en calidad alta** (22 kHz). Es la voz Piper más cercana al acento colombiano/paisa (Piper no tiene voz colombiana oficial; la mexicana es la alternativa más natural). La sintetización ocurre **localmente en CPU** con `onnxruntime`, sin internet.

| Aspecto | Detalle |
|---|---|
| Tipo | Voz neuronal femenina, español de México |
| Conectividad | **Sin internet** (modelo ONNX local) |
| Sample rate | 22.050 Hz (calidad high) |
| Latencia | 0.5–2 s en CPU moderna |
| Tamaño en disco | ~62 MB |
| Activación | Automática si existe `models/tts/es_MX-claude-high.onnx` |

### Resolución TTS (cascada)

En cada respuesta, la app intenta en este orden:
1. **Piper local** — si `piper-tts` está instalado y el modelo ONNX existe en `models/tts/`.
2. **Edge TTS (nube)** — fallback si Piper no está disponible o falla.
3. Sin audio — si ningún motor está disponible (la app sigue funcionando, solo sin voz).

El motor activo se muestra en la barra lateral de la app bajo "Voz TTS".

### Cambiar la voz de Edge TTS

Edita la constante en `app/chat.py`:

```python
TTS_VOICE = "es-CO-SalomeNeural"   # voz actual (colombiana femenina)
# Alternativas:
# TTS_VOICE = "es-CO-GonzaloNeural"  # colombiana masculina
# TTS_VOICE = "es-MX-DaliaNeural"    # mexicana femenina
# TTS_VOICE = "es-ES-ElviraNeural"   # española
```

---

## Datos

| Tabla | Filas | Descripción |
|---|---|---|
| `RegistrosDPA_clean` | 618,875 | Registros históricos de ejecución de bots |
| `TiemposManuales_clean` | 95 | Tiempo manual que reemplaza cada bot |
| `RolesAreas_clean` | 91 | Roles y áreas impactadas con valor/hora |

Período cubierto: 2020 – 2024. Los archivos de base de datos están almacenados en **Git LFS**.

---

## Cálculo de ROI

### En producción — fórmula determinística

La app calcula el ROI con una fórmula **auditable, sin caja negra**:

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

### Modelo predictivo (experimental, solo en notebook)

> **Estado:** experimental, **no integrado en la app**. Vive en [`notebooks/03_modelo_roi.ipynb`](notebooks/03_modelo_roi.ipynb) como ejercicio de exploración. La pestaña 3 de la app usa exclusivamente la fórmula determinística descrita arriba.

XGBoostRegressor con transformación `log1p(ROI)` para manejar la distribución sesgada del target.

| Métrica | Valor |
|---|---|
| Algoritmo | XGBoost 2.x (`tree_method=hist`) |
| R² (escala log, test split) | ~0,81 |
| R² CV 5-fold (media ± std) | 0,03 ± 0,27 (alta varianza por n pequeño) |
| MAE (escala original) | ~2.291 puntos porcentuales |
| Top features | TiempoManualHoras, TasaExito, DuracionPromedio, ValorHora |
| Muestras de entrenamiento | ~30 bots con datos completos |

> El modelo mejora a medida que se completen los datos de `TiemposManuales` para más bots.

---

## Limitaciones Conocidas

- **Muestra reducida para el modelo predictivo:** solo ~30 bots tienen datos completos en `TiemposManuales`, lo que produce alta varianza en validación cruzada. Por eso el modelo XGBoost queda en estado experimental y la app usa la calculadora determinística.
- **Hardware local requerido:** la app no funciona sin Ollama corriendo y sin el modelo Qwen descargado (~4,7 GB). No hay despliegue en la nube — todo se ejecuta en la máquina del usuario.
- **API `MediaRecorder` solo en `localhost` o HTTPS:** si despliegas la app en otra máquina por IP en HTTP, el botón de micrófono fallará silenciosamente.
- **Latencia del LLM en CPU:** sin GPU, una respuesta del agente SQL puede tardar 20–60 s. Con GPU NVIDIA (≥ 6 GB VRAM) baja a unos pocos segundos.
- **Costo operativo fijo del robot (7.300 COP/h):** es un promedio basado en la infraestructura actual (Azure + UiPath). Cambios reales en licenciamiento o infraestructura requieren actualizar el valor en la fórmula.
- **Edge TTS requiere internet:** si el modelo Piper no está descargado (ver paso `download_piper.py`), la app cae a Edge TTS, que requiere conexión a internet. Si no hay internet y no hay modelo Piper, el botón de audio simplemente no se muestra (la app sigue funcionando sin voz).


