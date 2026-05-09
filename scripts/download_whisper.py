"""
Pre-descarga el modelo Whisper-small para transcripción de voz.

Solo es necesario ejecutarlo una vez por máquina. El modelo queda cacheado en
~/.cache/huggingface/hub/ (Linux/Mac) o %USERPROFILE%\\.cache\\huggingface\\hub
(Windows) y se reutiliza en todas las ejecuciones futuras.

Tamaño aproximado: ~480 MB (cuantización int8).
"""

import sys

from faster_whisper import WhisperModel

MODEL_NAME = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def main() -> int:
    print(f"Descargando whisper-{MODEL_NAME} ({COMPUTE_TYPE}, ~480 MB)...")
    print("Esto puede tardar 2-5 minutos segun tu conexion.\n")
    try:
        WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as exc:
        print(f"\nError al descargar el modelo: {exc}", file=sys.stderr)
        return 1
    print(f"\nListo. Modelo whisper-{MODEL_NAME} cacheado y listo para usar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
