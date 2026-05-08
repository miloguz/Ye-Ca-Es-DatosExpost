# Sinfama RPA Analytics
## Pitch del proyecto · Especialización en Ciencia de Datos e IA

---

## 1. Problema y motivación

Las cajas de compensación familiar gestionan portafolios crecientes de automatizaciones RPA (Robotic Process Automation) cuyo retorno de inversión rara vez se mide de forma rigurosa. La pregunta que motiva este trabajo es operativa pero típica de la práctica analítica:

> **¿Cuánto valor monetario está generando cada bot del portafolio, qué factores explican ese valor, y cuál es el ROI esperado de un bot nuevo antes de implementarlo?**

El proyecto integra tres fuentes operativas (ejecuciones, tiempos manuales, valor-hora por rol) para producir métricas de ROI por automatización, un modelo predictivo de ROI ex-ante y una interfaz conversacional que permite a usuarios no técnicos consultar la base de datos en lenguaje natural.

---

## 2. Datos

| Tabla | Filas | Naturaleza |
|---|---|---|
| `RegistrosDPA_clean` | 618.875 | Series temporales de ejecución de bots (2020–2024) |
| `TiemposManuales_clean` | 95 | Tiempo humano que reemplaza cada bot |
| `RolesAreas_clean` | 91 | Valor-hora del rol impactado en COP |

Solo **~33 bots** tienen datos completos en las tres tablas (el cruce limita el `n` real de entrenamiento). Esto es honestamente la **principal limitación del proyecto** y un hallazgo en sí mismo: el negocio no tenía la trazabilidad de tiempos manuales necesaria para un análisis riguroso de ROI.

---

## 3. Pipeline de datos

```
CSVs crudos ──► csv_to_sqlite.py ──► Procesos.db
                                          │
                       01_preprocesamiento.ipynb
                                          ▼
                                   Procesos_clean.db
                                          │
            roi_calculator.build_roi_dataset()
                                          ▼
                                  DataFrame de ROI por bot
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                  app/chat.py                  scripts/train_roi_model.py
                  (Streamlit + Ollama)         (XGBoost + métricas JSON)
```

El comando `uv run python scripts/train_roi_model.py` reentrena todo end-to-end, garantizando reproducibilidad sin depender de Jupyter.

---

## 4. Modelo 1 · Cálculo determinístico de ROI

**Fórmula** (dominio del negocio, no aprendida de los datos):

$$
\text{ROI}_\% = \frac{T_{manual} - T_{robot} \cdot f_{robot}}{T_{robot} \cdot f_{robot}} \times 100
$$

donde $f_{robot} = 0{,}25$ representa que mantener el robot cuesta ≈25% del valor-hora humano (parámetro asumido del dominio).

### ¿Por qué no usar la salida del modelo ML para mostrar el ROI?

Decisión de diseño explícita: la app **muestra el ROI calculado por la fórmula**, no la predicción del XGBoost.

- La fórmula es determinística dadas las entradas del usuario y queda **internamente consistente** con el `Ahorro_Neto` que se muestra al lado.
- El modelo ML sirve como herramienta de inferencia y análisis de importancia de variables, no como fuente de verdad para un valor que el usuario va a defender ante un comité.
- Mezclar formula y modelo sin documentación habría producido contradicciones difíciles de explicar (una caja con ROI predicho de 50.000% pero ahorro neto coherente con 5.000%).

Esta decisión muestra una práctica importante en ciencia de datos aplicada: **no todo problema cuantitativo debe resolverse con un modelo**. Cuando existe una fórmula auditable y aceptada por el negocio, el modelo debe complementarla, no reemplazarla.

---

## 5. Modelo 2 · Predicción de ROI con XGBoost

### 5.1 Algoritmo elegido y justificación

**XGBoost (Extreme Gradient Boosting), regresión, `tree_method=hist`, `device=cuda` con fallback a CPU.**

| Criterio | XGBoost | Random Forest | Regresión Lineal | Red Neuronal |
|---|---|---|---|---|
| Datos pequeños (n≈30) | ✅ regularización L1/L2 | ⚠️ varianza alta | ✅ | ❌ overfit garantizado |
| Mezcla numérico + categórico | ✅ con OneHot | ✅ | ✅ | ✅ |
| Robusto a outliers extremos | ✅ con log-transform del target | ⚠️ | ❌ | ⚠️ |
| Interpretabilidad (feature importance, SHAP) | ✅ nativo | ✅ | ✅ coeficientes | ❌ |
| Aceleración GPU | ✅ CUDA out-of-the-box | ❌ | n/a | ✅ |

La elección **no fue arbitraria**: el target ROI tiene distribución muy sesgada (algunos bots con ROI > 100.000%) y un `n` chico. Boosting con regularización fuerte (`reg_alpha=0.1`, `max_depth=3`, `min_child_weight=3`) mitiga el sobre-ajuste, y la transformación `log1p(ROI)` convierte la distribución en algo aproximadamente simétrico antes de entrenar.

### 5.2 Pipeline de scikit-learn

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), 11 features numéricas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Tecnologia", "Estado"]),
    ])),
    ("model", XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        device="cuda" if disponible else "cpu",
    )),
])
```

Encapsular preprocesamiento + estimador en un `Pipeline` previene **fugas de datos** (`StandardScaler` se ajusta solo al fold de entrenamiento durante CV) y simplifica la serialización con `joblib`.

### 5.3 Métricas (último entrenamiento)

| Métrica | Valor | Interpretación |
|---|---|---|
| R² (test, log-scale) | 0,11 | Explicamos ~11% de la varianza fuera de muestra. Bajo, esperado con n=33. |
| R² CV 5-fold (mean ± std) | 0,26 ± 0,28 | Alta varianza entre folds: el modelo no es estable. |
| MAE (escala original) | 58.657% | Error absoluto enorme — refleja el sesgo del target. |
| Dispositivo | CUDA (RTX 4060) | Entrenamiento en ~3 segundos. |

**Lectura honesta de las métricas:** con 33 muestras efectivas no hay forma de entrenar un modelo predictivo confiable de un target con cola tan larga. El proyecto documenta esto explícitamente y deja la fórmula determinística como salida principal. **El valor académico está en haber identificado, medido y comunicado correctamente esta limitación**, no en haber inflado un número de R².

### 5.4 Variables más importantes (interpretabilidad)

El método `get_feature_importance()` devuelve el ranking nativo de XGBoost. Los predictores dominantes son consistentes con el dominio:

1. `TiempoManualHoras` — el numerador del ROI
2. `TasaExito` — robots que fallan no ahorran tiempo
3. `DuracionPromedio_Horas` — el denominador del ROI
4. `ValorHoraPromedio` — multiplicador de magnitud

El modelo **redescubre la fórmula** desde los datos, lo cual es a la vez tranquilizador (no aprendió ruido) y la razón por la que aporta poco valor predictivo adicional sobre la fórmula misma.

---

## 6. Modelo 3 · Agente SQL conversacional con LLM local

### 6.1 Arquitectura

```
Pregunta usuario  →  Streamlit  →  Ollama (qwen2.5-coder:7b)
                                          │
                                          ▼
                                Genera SQL en bloque ```sql ... ```
                                          │
                                          ▼
                                    SQLite (lectura)
                                          │
                                          ▼
                            Resultados → segundo prompt al LLM
                                          │
                                          ▼
                              Respuesta en español + tabla
```

### 6.2 Por qué LLM local (Ollama) y no API

| Criterio | API comercial | Ollama local |
|---|---|---|
| Privacidad | Datos salen del entorno | ✅ Datos nunca salen |
| Costo marginal | $/query | ✅ Cero |
| Latencia | 1–3 s | 2–8 s (depende GPU) |
| Reproducibilidad clase | ⚠️ requiere API key | ✅ funciona offline |

Para un caso académico con datos sensibles (información laboral por rol) y sin presupuesto de API, Ollama con un modelo de 7B parámetros es la elección natural. Específicamente **qwen2.5-coder:7b** porque está afinado para generación de código (incluyendo SQL) y cabe en 8 GB de VRAM.

### 6.3 Patrón de dos pasadas (generation + interpretation)

El agente hace **dos llamadas al LLM**:
1. **Generación**: dado el esquema de la BD y la pregunta, produce SQL.
2. **Interpretación**: dadas las filas resultado, produce respuesta en lenguaje natural.

Separar las dos pasadas resulta en respuestas más cortas, menos alucinadas, y permite enseñarle al modelo el esquema una sola vez sin re-prompting por cada interpretación.

### 6.4 Manejo de errores defensivo

Cuando el SQL generado falla (columna inexistente, sintaxis inválida) o Ollama está caído, el agente **no expone el stack trace**. Devuelve un mensaje de fallback genérico y registra el error en logs. Esto se decidió tras observar errores recurrentes con preguntas sobre la columna `Desarrollador` (datos parciales que el LLM intentaba usar en JOINs inválidos).

---

## 7. Tecnologías y por qué cada una

| Tecnología | Rol | Por qué |
|---|---|---|
| **Python 3.11** | Lenguaje base | Estándar de ciencia de datos; tipado moderno (`X \| None`) |
| **uv** | Gestor de dependencias | 10–100× más rápido que pip; lockfile reproducible |
| **SQLite** | Almacenamiento | Cero configuración; transaccional; suficiente para 600k filas |
| **pandas / numpy** | Manipulación de datos | Lingua franca de la disciplina |
| **scikit-learn** | Pipelines y métricas | Estándar académico para CV, métricas, ColumnTransformer |
| **XGBoost** | Modelo predictivo | GBM regularizado con GPU CUDA; ideal para tabular pequeño |
| **Ollama + qwen2.5-coder** | LLM local | Privacidad + cero costo + reproducibilidad |
| **Streamlit** | UI | Prototipado de apps de datos sin tocar HTML/JS |
| **Plotly** | Gráficos interactivos | Hover, zoom, log-scale sin código adicional |
| **joblib** | Serialización del modelo | Estándar sklearn |
| **Git LFS** | Versionado de DB y CSVs | SQLite y CSVs grandes no caben en blobs Git normales |
| **fpdf2** | Guía de instalación PDF | Reportes auto-generados sin LaTeX |

---

## 8. Buenas prácticas aplicadas

- **Separación de responsabilidades**: `agent/` (orquestación LLM), `models/` (ML), `utils/` (lógica de negocio).
- **Constantes centralizadas**: `DB_PATH` vive en `database.py`, los demás módulos lo importan vía `..agent.database`.
- **Type hints** en funciones públicas; firma `dict | None` en lugar de `Dict[str, Any]`.
- **Pipeline reentrenable** con un solo comando (`scripts/train_roi_model.py`), exportando métricas a JSON para tracking.
- **Logging estructurado** en lugar de `print` o `except: pass` silencioso.
- **`.gitignore` declarativo**: artefactos generados, caches, secretos, IDE.
- **Documentación viva**: README + PDF de instalación + este pitch.

---

## 9. Lecciones de proceso

1. **El modelo no es siempre la respuesta.** Cuando existe una fórmula del negocio, exhibirla y validarla es más útil que esconderla detrás de un regresor.
2. **`n=33` no es un dataset.** Documentar la limitación y proponer cómo levantarla (completar `TiemposManuales`) es un hallazgo de ciencia de datos, no un fracaso.
3. **GPU no garantiza un buen modelo.** Entrenar XGBoost en CUDA en 3 segundos no compensa la falta de muestras — la calidad del dato manda.
4. **Privacidad por diseño**: usar un LLM local cambia la conversación con el dueño del dato — no hay que negociar contratos de procesamiento.

---

## 10. Próximos pasos

- **Datos**: cerrar la cobertura de `TiemposManuales` para pasar de n=33 a n>80.
- **Modelado**: una vez con más datos, explorar Quantile Regression para predecir bandas (P10, P50, P90) en lugar de un punto.
- **Operacionalización**: scheduler nocturno que reentrene el modelo y publique métricas en un dashboard.
- **Trazabilidad**: integrar MLflow para versionar runs (R², MAE, hash del dataset) y permitir comparar reentrenamientos.

---

> **Demo en clase**: pestañas Chat SQL → Análisis ROI → Predicción ROI; mostrar `scripts/train_roi_model.py` corriendo y `reports/metrics_roi.json` actualizándose.
