"""
Descarga el modelo de voz Piper TTS (es_MX-claude-high).

Voz femenina latinoamericana en calidad alta — la más cercana a un acento
colombiano/paisa disponible en el catálogo offline de Piper (no hay una
voz colombiana oficial; la mexicana es la alternativa más natural).

Tamaño: ~110 MB (calidad high, 22.05 kHz).

Uso:
    uv run python scripts/download_piper.py
"""

import json
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "tts"
VOICE = "es_MX-claude-high"
BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    "/es/es_MX/claude/high"
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
            print(f"[OK] {filename} ya existe en {dest}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Descargando {filename} ...")
        try:
            _download_file(url, dest)
            print(f"[OK] {filename} guardado")
        except Exception as exc:
            print(f"[ERR] Error al descargar {filename}: {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            sys.exit(1)

    # Normaliza el JSON de configuración: algunos modelos antiguos guardan
    # `phoneme_type: "PhonemeType.ESPEAK"` (repr de enum) en lugar de `espeak`.
    # piper-tts >= 1.2 valida estrictamente y falla en runtime sin esta fix.
    config_path = MODEL_DIR / f"{VOICE}.onnx.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        raw = cfg.get("phoneme_type")
        if isinstance(raw, str) and raw.startswith("PhonemeType."):
            normalized = raw.split(".", 1)[1].lower()
            cfg["phoneme_type"] = normalized
            config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Normalizado phoneme_type: '{raw}' -> '{normalized}'")
    except Exception as exc:
        print(f"[WARN] No se pudo normalizar la config: {exc}", file=sys.stderr)

    print(f"\n[DONE] Modelo de voz listo en {MODEL_DIR}")


if __name__ == "__main__":
    main()
