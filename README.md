# 📊 Proyecto 1 — Especialización en Ciencia de Datos e Inteligencia Artificial

Proyecto de análisis de datos que incluye la conversión de archivos planos (CSV) a una base de datos SQLite, como primer paso del pipeline de procesamiento de información.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Configuración del Entorno](#-configuración-del-entorno)
- [Ejecución](#-ejecución)
- [Herramientas Recomendadas](#-herramientas-recomendadas)

---

## 📖 Descripción

Este proyecto procesa tres fuentes de datos en formato CSV (`RegistrosDPA`, `TiemposManuales`, `RolesAreas`) y las consolida en una única base de datos SQLite (`Procesos.db`), lista para ser consultada y analizada en etapas posteriores del proyecto.

---

## 📁 Estructura del Proyecto

```
Proyecto1Especializacion/
├── Data/
│   ├── RegistrosDPA.csv
│   ├── TiemposManuales.csv
│   ├── RolesAreas.csv
│   ├── csv_to_sqlite.py     ← script utilizado para convertir los archivos CSV a SQLite
│   └── Procesos.db          ← Información a procesar
├── .venv/                   ← entorno virtual
├── main.py
├── pyproject.toml
├── .python-version          ← fija la versión Python a 3.14
└── README.md
```

---

## ✅ Requisitos Previos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Python | **3.14.2** | Verificar con `python --version` |
| uv | última estable | Gestor de paquetes del proyecto |
| pandas | ≥ 3.0.2 | Declarado en `pyproject.toml` |

---

## ⚙️ Configuración del Entorno

El proyecto utiliza **Python 3.14.2** (definido en `.python-version`) y gestiona sus dependencias con **`uv`**.

### Opción A — Con `uv` (recomendado)

```powershell
# 1. Instalar uv si aún no lo tienes
pip install uv

# 2. Crear el entorno virtual e instalar dependencias en un solo paso
uv sync
```

> `uv sync` lee `pyproject.toml` y `uv.lock`, crea `.venv/` automáticamente e instala todas las dependencias fijadas.

---
## 🛠️ Herramientas Recomendadas

### 🔍 SQLite Viewer (Visual Studio Code)

Para inspeccionar el archivo `Data/Procesos.db` directamente desde VS Code, instala la extensión **SQLite Viewer**:

1. Abre VS Code.
2. Ve a la pestaña **Extensiones** (`Ctrl + Shift + X`).
3. Busca: `SQLite Viewer`.
4. Instala la extensión publicada por **Florian Klampfer**.
5. Haz clic derecho sobre `Procesos.db` en el explorador de archivos → **Open With... → SQLite Viewer**.