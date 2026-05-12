"""
Descarga el modelo de voz Piper TTS (es_MX-ald-medium).

Es la voz en español latinoamericano más cercana al español colombiano/paisa
disponible en el catálogo offline de Piper. Tamaño: ~60 MB.

Uso:
    uv run python scripts/download_piper.py
"""

import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "tts"
VOICE = "es_MX-ald-medium"
BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    "/es/es_MX/ald/medium"
)
FILES = [f"{VOICE}.onnx", f"{VOICE}.onnx.json"]


def _download_file(url: str, dest: Path) -> None:
    def progress(count, block, total_size):
        downloaded = count * block
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  {pct:3d}%  {downloaded // 1024:,} KB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"✓ {filename} ya existe en {dest}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Descargando {filename} …")
        try:
            _download_file(url, dest)
            print(f"✓ {filename} guardado")
        except Exception as exc:
            print(f"✗ Error al descargar {filename}: {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            sys.exit(1)

    print(f"\n✅ Modelo de voz listo en {MODEL_DIR}")


if __name__ == "__main__":
    main()
