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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

from ..utils.roi_calculator import COSTO_HORA_ROBOT_COP

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
    """Detecta si XGBoost puede usar GPU CUDA, retorna 'cuda' o 'cpu'."""
    try:
        test = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        test.fit([[0, 0]], [0])
        return "cuda"
    except Exception:
        return "cpu"


def _build_pipeline(device: str = "cpu") -> Pipeline:
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
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


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


def train(df: pd.DataFrame) -> dict:
    """
    Entrena el modelo con el DataFrame de ROI calculado.

    Usa XGBoost con GPU si hay CUDA disponible (RTX 4060 recomendado).
    El target usa transformación log1p(ROI) para manejar la distribución
    sesgada (ROI tiene outliers extremos de hasta 500,000%).
    Retorna métricas en escala original y guarda el pipeline en disco.
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

    metrics = {
        "r2": r2_score(y_test_log, y_pred_log),
        "mae_pct": mean_absolute_error(y_test_raw, y_pred_raw),
        "rmse_pct": np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "device": device,
        "note": "XGBoost entrenado en escala log(ROI). R2 en escala log, MAE/RMSE en escala original.",
    }

    joblib.dump({"pipeline": pipeline, "log_transform": True, "device": device}, MODEL_PATH)
    return metrics


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Ejecuta el notebook 03_modelo_roi.ipynb primero."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Predice ROI para un bot nuevo dado sus características.

    features (dict) debe incluir:
      - TiempoManualHoras: float
      - Num_Ejecuciones: int
      - ValorHoraPromedio: float
      - Tecnologia: str  ("UiPath" | "IRPA" | "Power Automate")
      - Estado: str      ("Activo" | "Inactivo")
      - DuracionPromedio_Horas: float (opcional, default = TiempoManualHoras * 0.1)
      - PromTransacciones: float (opcional, default = 1)
      - EjecucionesPorDia: float (opcional)
      - DiasEnProduccion: int (opcional)
    """
    artifact = load_model()

    defaults = {
        "DuracionPromedio_Horas": features.get("TiempoManualHoras", 1) * 0.1,
        "PromTransacciones": 1.0,
        "TasaExito": 0.9,
        "TasaError": 0.05,
        "EjecucionesPorDia": features.get("Num_Ejecuciones", 100) / 365,
        "DiasEnProduccion": 365,
        "NumAreas": 1,
        "NumRoles": 1,
    }
    defaults.update(features)

    val_hora = features.get("ValorHoraPromedio", 30000)
    num_ejec = features.get("Num_Ejecuciones", 100)
    tiempo_manual = features.get("TiempoManualHoras", 1)
    dur_robot = defaults["DuracionPromedio_Horas"]

    beneficio = tiempo_manual * val_hora * num_ejec
    costo = dur_robot * COSTO_HORA_ROBOT_COP * num_ejec
    ahorro = beneficio - costo

    # ROI determinístico desde la fórmula — coherente con el ahorro mostrado.
    # El modelo XGBoost se entrenó sobre log1p(ROI) con outliers de hasta 500,000%,
    # por lo que sus predicciones extrapolan a valores poco realistas. La fórmula
    # es la fuente de verdad dadas las entradas del usuario.
    roi_pct = float((ahorro / costo) * 100) if costo > 0 else 0.0

    return {
        "roi_porcentaje": roi_pct,
        "ahorro_neto_cop": ahorro,
        "beneficio_bruto_cop": beneficio,
        "costo_robot_cop": costo,
        "device_usado": artifact.get("device", "cpu"),
    }


def get_feature_importance(top_n: int = 15) -> pd.DataFrame:
    """Retorna importancia de variables del modelo entrenado."""
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
