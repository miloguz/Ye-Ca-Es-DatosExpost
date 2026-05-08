"""Genera el PDF de instalacion y ejecucion del proyecto."""
from fpdf import FPDF
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "GUIA_INSTALACION.pdf"
LOGO_SVG = ROOT / "assets" / "logo.svg"

PRIMARY      = (219, 0, 97)     # #db0061
PRIMARY_DARK = (138, 0, 81)     # #8a0051
NAVY         = (27, 31, 48)     # #1b1f30
GRAY_BG      = (246, 248, 255)
GRAY_TXT     = (80, 80, 100)
WHITE        = (255, 255, 255)
BLACK        = (30, 30, 30)
YELLOW       = (240, 234, 20)   # accent
CODE_TXT     = (255, 255, 255)  # texto blanco en bloques de codigo (alto contraste sobre navy)
CODE_COMMENT = (240, 234, 20)   # comentarios (#...) en amarillo


def draw_logo(pdf, x, y, size=20):
    """Renderiza el logo de Sinfama. Intenta SVG; si falla, usa primitivas."""
    try:
        pdf.image(str(LOGO_SVG), x=x, y=y, w=size, h=size)
        return
    except Exception:
        pass
    pdf.set_fill_color(*PRIMARY)
    try:
        pdf.rect(x, y, size, size, "F", round_corners=True, corner_radius=size * 0.22)
    except TypeError:
        pdf.rect(x, y, size, size, "F")
    pdf.set_fill_color(*YELLOW)
    d = size * 0.26
    pdf.ellipse(x + size * 0.72 - d / 2, y + size * 0.13 - d / 2, d, d, "F")
    pdf.set_font("Helvetica", "B", int(size * 0.85))
    pdf.set_text_color(*WHITE)
    pdf.set_xy(x, y + size * 0.08)
    pdf.cell(size, size * 0.9, "S", align="C")


class PDF(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 18, "F")
        draw_logo(self, x=8, y=3, size=12)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*WHITE)
        self.set_xy(24, 4)
        self.cell(0, 10, "Sinfama  |  Agente SQL - Procesos RPA")
        self.set_text_color(200, 200, 220)
        self.set_font("Helvetica", "", 8)
        self.set_xy(140, 4)
        self.cell(0, 10, "Guia de instalacion y ejecucion", align="R")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY_TXT)
        self.set_draw_color(*PRIMARY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(1)
        self.cell(0, 5, f"Pagina {self.page_no()}", align="C")

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(*PRIMARY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {text}", ln=True, fill=True)
        self.ln(2)
        self.set_text_color(*BLACK)

    def body(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        self.set_x(10 + indent)
        self.multi_cell(0, 5.5, text)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        self.set_x(14)
        self.cell(5, 5.5, "-")
        self.set_x(19)
        self.multi_cell(0, 5.5, text)

    def code_block(self, lines):
        self.set_font("Courier", "", 9)
        x = 10
        y = self.get_y()
        padding = 3
        total_h = len(lines) * 5 + padding * 2
        self.set_fill_color(*NAVY)
        self.rect(x, y, 190, total_h, "F")
        # Borde izquierdo magenta para acento de marca
        self.set_fill_color(*PRIMARY)
        self.rect(x, y, 1.5, total_h, "F")
        self.set_xy(x + 4, y + padding)
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                self.set_text_color(*CODE_COMMENT)
            else:
                self.set_text_color(*CODE_TXT)
            self.set_x(x + 4)
            self.cell(0, 5, line, ln=True)
        self.ln(3)
        self.set_text_color(*BLACK)

    def table_row(self, cols, widths, header=False):
        if header:
            self.set_fill_color(*NAVY)
            self.set_text_color(*WHITE)
            self.set_font("Helvetica", "B", 9)
        else:
            self.set_fill_color(*GRAY_BG)
            self.set_text_color(*BLACK)
            self.set_font("Helvetica", "", 9)
        self.set_x(10)
        for col, w in zip(cols, widths):
            self.cell(w, 6.5, col, border=0, fill=True, ln=False)
        self.ln()

    def note(self, text):
        self.set_fill_color(*GRAY_BG)
        self.set_text_color(*GRAY_TXT)
        self.set_font("Helvetica", "I", 9)
        self.set_x(10)
        self.multi_cell(0, 5, f"  Nota: {text}", fill=True)
        self.ln(1)
        self.set_text_color(*BLACK)


pdf = PDF()
pdf.set_margins(10, 22, 10)
pdf.set_auto_page_break(auto=True, margin=16)
pdf.add_page()

# ── Portada ───────────────────────────────────────────────────────────────────
pdf.set_fill_color(*NAVY)
pdf.rect(0, 18, 210, 75, "F")
# Logo grande centrado
draw_logo(pdf, x=(210 - 26) / 2, y=24, size=26)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(*WHITE)
pdf.set_xy(10, 54)
pdf.cell(0, 10, "Sinfama", align="C", ln=True)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(*PRIMARY)
pdf.cell(0, 6, "Caja de compensacion", align="C", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(*WHITE)
pdf.cell(0, 7, "Agente SQL - Procesos RPA", align="C", ln=True)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(200, 200, 220)
pdf.cell(0, 7, "Guia de Instalacion y Ejecucion", align="C", ln=True)
pdf.ln(38)

pdf.set_text_color(*BLACK)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(
    0, 6,
    "Este documento describe como clonar, configurar y ejecutar el proyecto "
    "de analisis de procesos RPA con agente SQL local (Ollama) y prediccion "
    "de ROI (XGBoost + GPU) en cualquier equipo con Windows, macOS o Linux.",
)

# ── Requisitos previos ────────────────────────────────────────────────────────
pdf.section_title("1. Requisitos previos")
pdf.table_row(["Herramienta", "Version minima", "Descarga"], [48, 34, 108], header=True)
reqs = [
    ("Python",       "3.11",      "https://python.org/downloads"),
    ("Git",          "cualquier", "https://git-scm.com"),
    ("Git LFS",      "cualquier", "https://git-lfs.github.com"),
    ("Ollama",       "0.23+",     "https://ollama.com/download"),
    ("CUDA Toolkit", "12.x (op)", "https://developer.nvidia.com/cuda-downloads"),
]
for row in reqs:
    pdf.table_row(list(row), [48, 34, 108])
pdf.ln(2)
pdf.note(
    "GPU NVIDIA es opcional. XGBoost detecta CUDA y hace fallback a CPU automaticamente."
)

# ── Paso 2 ────────────────────────────────────────────────────────────────────
pdf.section_title("2. Instalar Git LFS (obligatorio para las bases de datos)")
pdf.body(
    "El repositorio usa Git LFS para los archivos .db (bases de datos SQLite, ~300 MB).\n"
    "Sin esto los archivos quedaran vacios."
)
pdf.ln(1)
pdf.code_block([
    "# Windows (winget)",
    "winget install GitHub.GitLFS",
    "",
    "# macOS",
    "brew install git-lfs",
    "",
    "# Ubuntu / Debian",
    "sudo apt install git-lfs",
    "",
    "# Activar globalmente (una sola vez por maquina)",
    "git lfs install",
])

# ── Paso 3 ────────────────────────────────────────────────────────────────────
pdf.section_title("3. Clonar el repositorio")
pdf.code_block([
    "git clone https://github.com/miloguz/Proyecto1Especializacion.git",
    "cd Proyecto1Especializacion",
    "",
    "# Si los .db quedaron vacios:",
    "git lfs pull",
])

# ── Paso 4 ────────────────────────────────────────────────────────────────────
pdf.section_title("4. Instalar dependencias Python")
pdf.body("Opcion A - con uv (recomendado, resuelve el entorno automaticamente):")
pdf.code_block(["pip install uv", "uv sync"])
pdf.body("Opcion B - con pip estandar:")
pdf.code_block([
    "pip install streamlit ollama plotly xgboost joblib pandas \\",
    "            scikit-learn seaborn matplotlib numpy",
])

# ── Paso 5 ────────────────────────────────────────────────────────────────────
pdf.section_title("5. Instalar Ollama y descargar el modelo LLM")
pdf.body(
    "El agente SQL necesita Ollama corriendo localmente y el modelo "
    "qwen2.5-coder:7b (~4.7 GB)."
)
pdf.code_block([
    "# Terminal 1 - servidor Ollama",
    "ollama serve",
    "",
    "# Terminal 2 - descargar modelo (solo la primera vez)",
    "ollama pull qwen2.5-coder:7b",
])
pdf.note(
    "En Windows, Ollama se instala como servicio y arranca automaticamente. "
    "No es necesario 'ollama serve' de forma manual."
)

# ── Paso 6 ────────────────────────────────────────────────────────────────────
pdf.section_title("6. Entrenar el modelo de prediccion de ROI")
pdf.body(
    "El archivo models/roi_model.joblib NO esta en el repositorio (figura en .gitignore).\n"
    "Hay que generarlo una vez antes de usar la pestana 'Prediccion ROI':"
)
pdf.code_block([
    "python -c \"",
    "import sys; sys.path.insert(0, '.')",
    "from src.utils.roi_calculator import build_roi_dataset",
    "from src.models.roi_predictor import train",
    "metrics = train(build_roi_dataset())",
    "print('Dispositivo:', metrics['device'].upper())",
    "print('Modelo guardado en models/roi_model.joblib')",
    "\"",
])
pdf.note(
    "Con GPU NVIDIA y CUDA instalado, el entrenamiento usa la GPU automaticamente."
)

# ── Paso 7 ────────────────────────────────────────────────────────────────────
pdf.section_title("7. Ejecutar la aplicacion")
pdf.code_block([
    "# Opcion A - script rapido (solo Windows)",
    "run_app.bat",
    "",
    "# Opcion B - manual (Windows / macOS / Linux)",
    "streamlit run app/chat.py",
])
pdf.body("Abrir en el navegador:  http://localhost:8501")

# ── Solucion de problemas ─────────────────────────────────────────────────────
pdf.section_title("8. Solucion de problemas comunes")
pdf.table_row(["Sintoma", "Causa probable", "Solucion"], [65, 55, 68], header=True)
issues = [
    ("Bases de datos vacias (0 bytes)", "Git LFS no instalado",       "git lfs install && git lfs pull"),
    ("ModuleNotFoundError: streamlit",  "Deps no instaladas",          "pip install streamlit  o  uv sync"),
    ("'Ollama no esta corriendo'",      "Servidor Ollama apagado",     "ollama serve en terminal aparte"),
    ("Prediccion ROI falla al cargar",  "Modelo no entrenado",         "Ejecutar el paso 6"),
    ("XGBoost no detecta GPU",          "CUDA Toolkit no instalado",   "Instalar CUDA; fallback CPU automatico"),
]
for row in issues:
    pdf.table_row(list(row), [65, 55, 68])

# ── Estructura del proyecto ───────────────────────────────────────────────────
pdf.section_title("9. Estructura del proyecto")
pdf.set_font("Courier", "", 8.5)
estructura = [
    "Proyecto1Especializacion/",
    "  app/chat.py                    <- streamlit run app/chat.py",
    "  assets/logo.svg                <- logo Sinfama",
    "  src/",
    "    agent/                       <- agente SQL + conexion DB",
    "    models/roi_predictor.py      <- XGBoost GPU/CPU",
    "    utils/roi_calculator.py      <- calculo de ROI",
    "  data/database/",
    "    Procesos_clean.db            <- base de datos (Git LFS)",
    "  models/                        <- roi_model.joblib (generado local)",
    "  notebooks/                     <- EDA y entrenamiento paso a paso",
    "  run_app.bat                    <- acceso rapido Windows",
    "  pyproject.toml                 <- dependencias del proyecto",
]
x, y = 10, pdf.get_y()
h = len(estructura) * 5 + 6
pdf.set_fill_color(*NAVY)
pdf.rect(x, y, 190, h, "F")
pdf.set_fill_color(*PRIMARY)
pdf.rect(x, y, 1.5, h, "F")
pdf.set_text_color(*CODE_TXT)
pdf.set_xy(x + 4, y + 3)
for line in estructura:
    if "<-" in line:
        path_part, _, comment_part = line.partition("<-")
        pdf.set_x(x + 4)
        pdf.set_text_color(*CODE_TXT)
        pdf.cell(pdf.get_string_width(path_part), 5, path_part, ln=False)
        pdf.set_text_color(*CODE_COMMENT)
        pdf.cell(0, 5, "<-" + comment_part, ln=True)
    else:
        pdf.set_x(x + 4)
        pdf.set_text_color(*CODE_TXT)
        pdf.cell(0, 5, line, ln=True)
pdf.set_text_color(*BLACK)
pdf.ln(4)

# ── Formula ROI ───────────────────────────────────────────────────────────────
pdf.section_title("10. Formula de calculo de ROI")
pdf.code_block([
    "Beneficio_Bruto = TiempoManual x ValorHora x NumEjecuciones",
    "Costo_Robot     = DuracionRobot x ValorHora x 0.25 x NumEjecuciones",
    "Ahorro_Neto     = Beneficio_Bruto - Costo_Robot",
    "ROI%            = (Ahorro_Neto / Costo_Robot) x 100",
])
pdf.body(
    "El modelo predictivo (XGBoost) aplica transformacion log1p(ROI) antes de entrenar\n"
    "para manejar la distribucion sesgada del target (ROI puede superar el 500,000%)."
)

pdf.output(str(OUT))
print(f"PDF generado: {OUT}")
