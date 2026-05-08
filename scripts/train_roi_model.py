"""
Pipeline reproducible de entrenamiento del modelo de ROI.

Construye el dataset desde la base de datos limpia (`Procesos_clean.db`),
entrena el regresor XGBoost (con GPU si está disponible) y guarda el
artefacto en `models/roi_model.joblib`.

Uso:
    uv run python scripts/train_roi_model.py
    uv run python scripts/train_roi_model.py --output models/roi_v2.joblib

Equivalente a ejecutar el notebook `03_modelo_roi.ipynb` end-to-end,
pero en un único comando — pensado para CI o re-entrenamientos rápidos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.roi_predictor import MODEL_PATH, train
from src.utils.roi_calculator import DB_PATH, build_roi_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_roi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help=f"Ruta del modelo serializado (default: {MODEL_PATH.relative_to(ROOT)})",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "reports" / "metrics_roi.json",
        help="JSON con las métricas de evaluación (default: reports/metrics_roi.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not DB_PATH.exists():
        log.error("No se encontró la base de datos: %s", DB_PATH)
        log.error("Genera Procesos_clean.db con notebooks/01_preprocesamiento.ipynb primero.")
        return 1

    log.info("Construyendo dataset de ROI desde %s", DB_PATH.name)
    df = build_roi_dataset()
    log.info("Dataset: %d filas, %d columnas", len(df), df.shape[1])

    log.info("Entrenando modelo (XGBoost)...")
    metrics = train(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output != MODEL_PATH:
        args.output.write_bytes(MODEL_PATH.read_bytes())

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    log.info("Dispositivo: %s", metrics["device"].upper())
    log.info("R² (log-scale, test): %.3f", metrics["r2"])
    log.info("R² CV %d-fold: %.3f ± %.3f", 5, metrics["cv_r2_mean"], metrics["cv_r2_std"])
    log.info("MAE (escala original): %.0f%%", metrics["mae_pct"])
    log.info("Modelo guardado en: %s", args.output)
    log.info("Métricas guardadas en: %s", args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
