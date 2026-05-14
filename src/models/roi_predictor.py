"""
Modelo de predicción de ROI para automatizaciones RPA.

Objetivo: dado un conjunto de características de un nuevo bot (tecnología,
tiempo manual estimado, volumen esperado de ejecuciones, valor hora del rol),
predecir el ROI en porcentaje y el ahorro neto en COP.

Algoritmo: XGBoostRegressor con aceleración GPU (RTX 4060 / CUDA).
Usa transformación log1p(ROI) para manejar la distribución sesgada del target.
Detecta automáticamente si hay GPU disponible y hace fallback a CPU.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

import xgboost as xgb

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "roi_model.joblib"

CATEGORICAL_FEATURES = ["Tecnologia", "Estado"]
NUMERIC_FEATURES = [
    "TiempoManualHoras",
    "Num_Ejecuciones",
    "DuracionPromedio_Horas",
    "PromTransacciones",
    "TasaExito",
    "TasaError",
    "ValorHoraPromedio",
    "EjecucionesPorDia",
    "DiasEnProduccion",
    "NumAreas",
    "NumRoles",
]
TARGET = "ROI_Porcentaje"


def _detect_device() -> str:
    """Detecta GPU NVIDIA mediante nvidia-smi sin inicializar contexto CUDA."""
    import subprocess
    candidates = [
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
         "--query-gpu=name", "--format=csv,noheader"],
    ]
    for cmd in candidates:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return "cuda"
        except Exception:
            continue
    return "cpu"


def _build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    preprocessor.set_output(transform="default")  # numpy output → compatible con XGBoost 3.x
    return preprocessor


def _build_pipeline(device: str = "cpu") -> Pipeline:
    estimator = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        device=device,
        tree_method="hist",   # requerido para GPU en XGBoost 2.x
        verbosity=0,
    )
    return Pipeline([("preprocessor", _build_preprocessor()), ("model", estimator)])


def _build_baseline_pipeline() -> Pipeline:
    """Baseline simple: Ridge sobre escala log(ROI).

    Sirve como sanity check para el XGBoost: si XGBoost no supera de forma
    consistente a una regresión lineal regularizada con n=33, no aporta valor
    sobre la fórmula del negocio. Lo importante NO es vencer al baseline en
    el split de test (eso es ruido con n pequeño) sino en CV.
    """
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y limpia las columnas necesarias para el modelo."""
    df = df.copy()
    df["TasaExito"] = df["TasaExito"].fillna(0.5)
    df["TasaError"] = df["TasaError"].fillna(0.0)
    df["DuracionPromedio_Horas"] = df["DuracionPromedio_Horas"].fillna(
        df["TiempoManualHoras"] * 0.1
    )
    df["ValorHoraPromedio"] = df["ValorHoraPromedio"].fillna(
        df["ValorHoraPromedio"].median()
    )
    df["Estado"] = df["Estado"].fillna("Desconocido")
    df["Tecnologia"] = df["Tecnologia"].fillna("Desconocida")
    df["NumRoles"] = df["NumRoles"].fillna(1)
    df["NumAreas"] = df["NumAreas"].fillna(1)
    return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def _bootstrap_test_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_log: pd.Series | np.ndarray,
    n_iter: int = 500,
    random_state: int = 42,
) -> dict:
    """Intervalos de confianza al 95% para R² (escala log) y MAE (escala original).

    Usa bootstrap de las predicciones del split de test. Necesario porque con
    n_test pequeño (≈6) el R² puntual es muy inestable. Reportar bandas evita
    que el lector interprete un R² de 0.81 como evidencia de generalización.
    """
    rng = np.random.default_rng(random_state)
    y_pred_log = pipeline.predict(X_test)
    y_test_log = np.asarray(y_test_log)
    y_test_raw = np.expm1(y_test_log)
    y_pred_raw = np.expm1(y_pred_log)

    r2_samples: list[float] = []
    mae_samples: list[float] = []
    n = len(y_test_log)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_test_log[idx])) < 2:
            continue  # R² indefinido si todos los y son iguales
        r2_samples.append(r2_score(y_test_log[idx], y_pred_log[idx]))
        mae_samples.append(mean_absolute_error(y_test_raw[idx], y_pred_raw[idx]))

    return {
        "r2_test_ci95_low": float(np.percentile(r2_samples, 2.5)) if r2_samples else float("nan"),
        "r2_test_ci95_high": float(np.percentile(r2_samples, 97.5)) if r2_samples else float("nan"),
        "mae_test_ci95_low": float(np.percentile(mae_samples, 2.5)) if mae_samples else float("nan"),
        "mae_test_ci95_high": float(np.percentile(mae_samples, 97.5)) if mae_samples else float("nan"),
        "bootstrap_iter": n_iter,
    }


def train(df: pd.DataFrame) -> dict:
    """
    Entrena el modelo con el DataFrame de ROI calculado.

    Usa XGBoost con GPU si hay CUDA disponible (RTX 4060 recomendado).
    El target usa transformación log1p(ROI) para manejar la distribución
    sesgada (ROI tiene outliers extremos de hasta 500,000%).

    Adicionalmente entrena un baseline de Ridge sobre el mismo split, y
    calcula intervalos de confianza al 95% por bootstrap para R² y MAE.
    Estas dos adiciones son cruciales con n pequeño: sin baseline no se
    puede afirmar que XGBoost aporta sobre una regresión lineal, y sin CI
    el R² puntual del test puede ser engañoso.

    Retorna métricas y guarda el pipeline en disco.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    device = _detect_device()

    df_clean = df.dropna(subset=[TARGET] + ["TiempoManualHoras", "ValorHoraPromedio"])
    df_pos = df_clean[df_clean[TARGET] > 0]  # log requiere valores positivos

    X = prepare_features(df_pos)
    y_raw = df_pos[TARGET]
    y_log = np.log1p(y_raw)

    n_cv = min(5, max(2, len(X) // 8))
    test_size = 0.2 if len(X) >= 30 else 0.15

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=test_size, random_state=42
    )
    y_test_raw = np.expm1(y_test_log)

    pipeline = _build_pipeline(device)
    pipeline.fit(X_train, y_train_log)

    y_pred_log = pipeline.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    cv_scores = cross_val_score(pipeline, X, y_log, cv=n_cv, scoring="r2")

    # Baseline Ridge — ¿XGBoost supera a la regresión lineal regularizada?
    baseline = _build_baseline_pipeline()
    baseline.fit(X_train, y_train_log)
    y_base_log = baseline.predict(X_test)
    y_base_raw = np.expm1(y_base_log)
    baseline_cv = cross_val_score(baseline, X, y_log, cv=n_cv, scoring="r2")

    # Bootstrap CI sobre métricas de test
    ci = _bootstrap_test_metrics(pipeline, X_test, y_test_log)

    metrics = {
        # XGBoost — métricas principales
        "r2": r2_score(y_test_log, y_pred_log),
        "mae_pct": mean_absolute_error(y_test_raw, y_pred_raw),
        "rmse_pct": np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        # Bootstrap CI 95% (lectura honesta con n pequeño)
        **ci,
        # Baseline Ridge — sanity check
        "baseline_r2": r2_score(y_test_log, y_base_log),
        "baseline_mae_pct": mean_absolute_error(y_test_raw, y_base_raw),
        "baseline_cv_r2_mean": baseline_cv.mean(),
        "baseline_cv_r2_std": baseline_cv.std(),
        # Metadatos
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(X),
        "device": device,
        "note": (
            "XGBoost entrenado en escala log(ROI). R2 en escala log, "
            "MAE/RMSE en escala original. Con n pequeño, CV R2 y CI95 mandan; "
            "el R2 puntual del test es referencial. Baseline Ridge se incluye "
            "como referencia: si XGBoost no lo supera en CV, no aporta valor."
        ),
    }

    joblib.dump(
        {
            "pipeline": pipeline,
            "baseline_pipeline": baseline,
            "log_transform": True,
            "device": device,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    return {**metrics, "pipeline": pipeline, "baseline_pipeline": baseline}


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Ejecuta el notebook 03_modelo_roi.ipynb primero."
        )
    return joblib.load(MODEL_PATH)


def get_feature_importance(top_n: int = 15, pipeline=None) -> pd.DataFrame:
    """Retorna importancia de variables del modelo entrenado."""
    if pipeline is None:
        artifact = load_model()
        pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = num_names + cat_names

    importance = model.feature_importances_
    df = pd.DataFrame({"feature": all_names, "importance": importance})
    return df.sort_values("importance", ascending=False).head(top_n)
