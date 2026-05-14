"""Punto de entrada para el comando `sinfama-app`.

Lanza `streamlit run app/chat.py` desde el entorno actual.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Lanza la app Streamlit usando el mismo intérprete Python en uso."""
    script = str(Path(__file__).parent / "chat.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", script]))


if __name__ == "__main__":
    main()
