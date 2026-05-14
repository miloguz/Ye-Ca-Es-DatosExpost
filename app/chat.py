"""
Interfaz Streamlit para el Agente SQL de Procesos RPA.
Corre localmente con Ollama — sin API keys ni conexión a internet.

Iniciar:
    streamlit run app/chat.py
"""

import asyncio
import base64
import io
import re
import sys
import wave
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.database import get_db_stats, validate_db_schema
from src.agent.sql_agent import DEFAULT_MODEL, ask, list_available_models
from src.utils.roi_calculator import COSTO_HORA_ROBOT_COP, build_roi_dataset, get_roi_summary


@st.cache_resource(show_spinner="Cargando modelo de transcripción...")
def get_whisper_model():
    """Carga el modelo Whisper-small (CPU, int8) y lo cachea entre reruns."""
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe bytes de audio a texto en español usando faster-whisper."""
    model = get_whisper_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes), language="es", beam_size=5
    )
    return " ".join(seg.text for seg in segments).strip()


TTS_VOICE_EDGE = "es-CO-SalomeNeural"  # colombiana femenina; alternativa: es-CO-GonzaloNeural
PIPER_VOICE = "es_MX-claude-high"  # voz Piper offline femenina latinoamericana (HQ)
PIPER_MODEL_DIR = Path(__file__).parent.parent / "models" / "tts"


def _piper_available() -> tuple[bool, Path | None]:
    """Comprueba si el paquete piper-tts y el modelo ONNX están disponibles."""
    try:
        import piper  # noqa: F401, PLC0415
    except ImportError:
        return False, None
    model_path = PIPER_MODEL_DIR / f"{PIPER_VOICE}.onnx"
    config_path = PIPER_MODEL_DIR / f"{PIPER_VOICE}.onnx.json"
    if not model_path.exists() or not config_path.exists():
        return False, None
    return True, model_path


def _edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _tts_available() -> tuple[bool, str]:
    """Retorna (disponible, mensaje) según el motor TTS detectado.

    Prioridad: Piper local (offline) > Edge TTS (nube).
    """
    piper_ok, _ = _piper_available()
    if piper_ok:
        return True, f"🔊 Piper local · {PIPER_VOICE}"
    if _edge_tts_available():
        return True, f"🔊 Edge TTS (nube) · {TTS_VOICE_EDGE}"
    return False, "🔇 Sin motor TTS disponible (ejecuta: uv sync)"


def _strip_md(text: str) -> str:
    """Elimina marcas Markdown antes de sintetizar voz."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@st.cache_resource(show_spinner=False)
def _load_piper_voice():
    """Carga el modelo Piper ONNX y lo cachea entre reruns."""
    from piper import PiperVoice  # noqa: PLC0415

    model_path = PIPER_MODEL_DIR / f"{PIPER_VOICE}.onnx"
    return PiperVoice.load(str(model_path))


def _synthesize_piper(text: str) -> bytes | None:
    """Genera bytes WAV con Piper TTS local (sin internet)."""
    try:
        voice = _load_piper_voice()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return buffer.getvalue()
    except Exception:
        return None


def _synthesize_edge_tts(text: str) -> bytes | None:
    """Genera bytes MP3 con Microsoft Edge TTS (requiere internet)."""
    try:
        import edge_tts  # noqa: PLC0415

        async def _synth() -> bytes:
            communicate = edge_tts.Communicate(text, TTS_VOICE_EDGE)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            return bytes(audio_data)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_synth())
        finally:
            loop.close()
    except Exception:
        return None


def synthesize_speech(text: str) -> bytes | None:
    """Sintetiza la respuesta. Prioriza Piper local; fallback a Edge TTS."""
    text_clean = _strip_md(text)
    piper_ok, _ = _piper_available()
    if piper_ok:
        audio = _synthesize_piper(text_clean)
        if audio:
            return audio
    return _synthesize_edge_tts(text_clean)


def audio_play_button(audio_bytes: bytes, key: str) -> None:
    """Botón play/pause con audio embebido (detecta WAV vs MP3 por header)."""
    mime = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio id="a_{key}" src="data:{mime};base64,{b64}" preload="auto"></audio>
    <button id="b_{key}"
      onclick="(function(){{
        var a=document.getElementById('a_{key}');
        var b=document.getElementById('b_{key}');
        if(a.paused){{
          a.play();
          b.innerHTML='&#9646;&#9646;&nbsp; Pausar';
        }}else{{
          a.pause();
          b.innerHTML='&#9654;&nbsp; Escuchar respuesta';
        }}
        a.onended=function(){{b.innerHTML='&#9654;&nbsp; Escuchar respuesta';}};
      }})()"
      style="background:#db0061;color:#fff;border:none;border-radius:999px;
             padding:0.35rem 1.1rem;font-family:'Roboto',sans-serif;font-weight:600;
             font-size:0.82rem;cursor:pointer;margin-top:6px;
             transition:background 0.2s;"
      onmouseover="this.style.background='#8a0051'"
      onmouseout="this.style.background='#db0061'"
    >&#9654;&nbsp; Escuchar respuesta</button>
    """
    st.components.v1.html(html, height=52)


# ── Logo ──────────────────────────────────────────────────────────────────────
LOGO_SVG = (Path(__file__).parent.parent / "assets" / "logo.svg").read_text(encoding="utf-8")

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sinfama · Agente RPA",
    page_icon=str(Path(__file__).parent.parent / "assets" / "logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Validación del esquema de la BD ───────────────────────────────────────────
_db_ok, _db_errors = validate_db_schema()
if not _db_ok:
    st.error("**Error en la base de datos**: la app no puede arrancar.")
    for _err in _db_errors:
        st.error(f"• {_err}")
    st.info(
        "Soluciones:\n"
        "1. `git lfs pull` para descargar las BDs versionadas.\n"
        "2. Ejecuta `notebooks/01_preprocesamiento.ipynb` para regenerar `Procesos_clean.db`.\n"
        "3. Verifica que la versión de `csv_to_sqlite.py` coincida con la esperada."
    )
    st.stop()

# ── Tema Comfama ──────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""<style>
  /* ── Variables de marca Comfama ── */
  :root {
    --cfm-primary:        #db0061;
    --cfm-primary-light:  #f15894;
    --cfm-primary-dark:   #8a0051;
    --cfm-primary-bg:     #fce3ed;
    --cfm-navy:           #1b1f30;
    --cfm-navy-mid:       #3c3f52;
    --cfm-navy-light:     #6f7287;
    --cfm-tertiary-bg:    #f6f8ff;
    --cfm-gray-light:     #f3f3f3;
    --cfm-border:         #e9ebff;
    --cfm-green-accent:   #c8f5c8;
    --cfm-yellow-accent:  #f0ea14;
    --cfm-white:          #ffffff;
  }

  /* ── Tipografía global ── */
  html, body, [class*="css"], .stMarkdown, .stText, button, input, select {
    font-family: 'Roboto', sans-serif !important;
  }

  /* ── Fondo general (secondaryBackgroundColor es navy en config.toml,
        se restaura a light en todos los contextos fuera del sidebar) ── */
  .stApp { background-color: var(--cfm-white); }
  /* Restaura fondo blanco en área principal (secondaryBg = navy por el sidebar) */
  [data-testid="stMainBlockContainer"],
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stVerticalBlock"],
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stMarkdownContainer"],
  .stApp > section:not([data-testid="stSidebar"]) .stCodeBlock,
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stForm"],
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stNumberInput"] > div,
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stTextInput"] > div {
    background-color: var(--cfm-white) !important;
  }
  /* El sidebar y sus hijos nunca deben recibir fondo blanco */
  html body .stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
  html body .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  html body .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  html body .stApp [data-testid="stSidebar"] .stMarkdown {
    background-color: transparent !important;
  }
  [data-testid="stExpander"] { background-color: var(--cfm-tertiary-bg) !important; }
  /* Métricas fuera del sidebar: fondo suave con acento rosa */
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stMetric"] {
    background-color: var(--cfm-tertiary-bg) !important;
    border-left: 4px solid var(--cfm-primary) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
  }

  /* ── Header superior ── */
  header[data-testid="stHeader"] {
    background-color: var(--cfm-white);
    border-bottom: 3px solid var(--cfm-primary);
  }

  /* ── Sidebar ── */
  html body .stApp section[data-testid="stSidebar"],
  html body .stApp section[data-testid="stSidebar"] > div {
    background-color: #1b1f30 !important;
    border-right: 3px solid #db0061;
  }
  html body .stApp section[data-testid="stSidebar"] *,
  html body .stApp [data-testid="stSidebarContent"] * {
    color: #ffffff !important;
  }
  html body .stApp [data-testid="stSidebar"] .stMarkdown h1,
  html body .stApp [data-testid="stSidebar"] .stMarkdown h2,
  html body .stApp [data-testid="stSidebar"] .stMarkdown h3 {
    color: #f15894 !important;
  }
  html body .stApp [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #ffffff !important;
  }
  html body .stApp [data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #f15894 !important;
    font-weight: 700 !important;
  }
  html body .stApp [data-testid="stSidebar"] hr {
    border-color: #3c3f52 !important;
  }
  /* Selectbox sidebar — contenedor base-web */
  html body .stApp [data-testid="stSidebar"] div[data-baseweb="select"] > div,
  html body .stApp [data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #3c3f52 !important;
    border-color: #6f7287 !important;
    color: #ffffff !important;
  }
  html body .stApp [data-testid="stSidebar"] div[data-baseweb="select"] span,
  html body .stApp [data-testid="stSidebar"] div[data-baseweb="select"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
  }
  html body .stApp [data-testid="stSidebar"] select,
  html body .stApp [data-testid="stSidebar"] input {
    background-color: #3c3f52 !important;
    color: #ffffff !important;
    border-color: #6f7287 !important;
  }

  /* ── Botones primarios ── */
  .stButton > button {
    background-color: var(--cfm-primary) !important;
    color: var(--cfm-white) !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Roboto', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background-color 0.2s ease !important;
  }
  .stButton > button:hover {
    background-color: var(--cfm-primary-dark) !important;
  }

  /* ── Pestañas ── */
  [data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid var(--cfm-border);
    gap: 0.5rem;
  }
  [data-testid="stTabs"] [role="tab"] {
    font-family: 'Roboto', sans-serif !important;
    font-weight: 500 !important;
    color: var(--cfm-navy-light) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.25rem !important;
    border: none !important;
    background: transparent !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--cfm-primary) !important;
    border-bottom: 3px solid var(--cfm-primary) !important;
    font-weight: 700 !important;
  }

  /* ── Chat ── */
  [data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
  }
  [data-testid="stChatMessage"][data-testid*="user"] {
    background-color: var(--cfm-primary-bg) !important;
  }
  [data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: var(--cfm-tertiary-bg) !important;
  }
  /* El chat input debe ser blanco (secondaryBg es navy por el sidebar) */
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInputContainer"] > div,
  [data-testid="stBottom"] > div,
  [data-testid="stBottom"] {
    background-color: #ffffff !important;
  }
  [data-testid="stChatInput"] > div {
    background-color: #ffffff !important;
    border: 2px solid var(--cfm-border) !important;
    border-radius: 12px !important;
  }
  [data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: var(--cfm-navy) !important;
    border: none !important;
    font-family: 'Roboto', sans-serif !important;
  }
  [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--cfm-primary) !important;
    box-shadow: 0 0 0 2px var(--cfm-primary-bg) !important;
  }

  /* ── Expanders (Ver SQL) ── */
  [data-testid="stExpander"] {
    border: 1px solid var(--cfm-border) !important;
    border-radius: 8px !important;
    background-color: var(--cfm-tertiary-bg) !important;
  }
  [data-testid="stExpander"] summary {
    color: var(--cfm-navy-mid) !important;
    font-weight: 500 !important;
  }

  /* ── Métricas ── */
  [data-testid="stMetric"] {
    background-color: var(--cfm-tertiary-bg);
    border-radius: 10px;
    padding: 0.75rem;
    border-left: 4px solid var(--cfm-primary);
  }
  [data-testid="stMetricValue"] {
    color: var(--cfm-primary) !important;
    font-weight: 700 !important;
  }

  /* ── Headings ── */
  h1, h2, h3 {
    font-family: 'Roboto', sans-serif !important;
    color: var(--cfm-navy) !important;
    font-weight: 700 !important;
  }
  h1 { border-bottom: 3px solid var(--cfm-primary); padding-bottom: 0.5rem; }

  /* ── Tablas ── */
  [data-testid="stDataFrame"] thead tr th {
    background-color: var(--cfm-navy) !important;
    color: var(--cfm-white) !important;
    font-weight: 600 !important;
  }
  [data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: var(--cfm-tertiary-bg) !important;
  }
  [data-testid="stDataFrame"] tbody tr:hover {
    background-color: var(--cfm-primary-bg) !important;
  }

  /* ── Info / Warning / Error ── */
  [data-testid="stInfo"] { border-left: 4px solid var(--cfm-primary) !important; }
  [data-testid="stSuccess"] { border-left: 4px solid #2ea84b !important; }

  /* ── Inputs y selectboxes en area principal (fondo blanco, texto navy) ── */
  /* base-input es el contenedor real con fondo — debe ser blanco fuera del sidebar */
  [data-testid="stMainBlockContainer"] [data-baseweb="base-input"],
  [data-testid="stMainBlockContainer"] [data-baseweb="base-input"] input,
  [data-testid="stMainBlockContainer"] div[data-baseweb="select"] > div,
  [data-testid="stMainBlockContainer"] div[data-baseweb="select"] > div input {
    background-color: #ffffff !important;
    color: var(--cfm-navy) !important;
  }
  /* Bordes */
  [data-testid="stMainBlockContainer"] [data-baseweb="base-input"] {
    border-color: var(--cfm-border) !important;
    border-radius: 8px !important;
  }
  [data-testid="stMainBlockContainer"] div[data-baseweb="select"] > div {
    border-color: var(--cfm-border) !important;
    border-radius: 8px !important;
  }
  /* Focus */
  [data-testid="stMainBlockContainer"] [data-baseweb="base-input"]:focus-within {
    border-color: var(--cfm-primary) !important;
    box-shadow: 0 0 0 2px var(--cfm-primary-bg) !important;
  }

  /* ── Botones stepper (+/-) de number_input ── */
  [data-testid="stNumberInputStepDown"],
  [data-testid="stNumberInputStepUp"],
  [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] button,
  [data-testid="stMainBlockContainer"] [data-baseweb="input-wrapper"] button {
    background-color: var(--cfm-tertiary-bg) !important;
    color: var(--cfm-navy) !important;
    border: none !important;
    border-left: 1px solid var(--cfm-border) !important;
    border-radius: 0 8px 8px 0 !important;
  }
  [data-testid="stNumberInputStepDown"]:hover,
  [data-testid="stNumberInputStepUp"]:hover,
  [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] button:hover {
    background-color: var(--cfm-primary-bg) !important;
    color: var(--cfm-primary) !important;
  }
  [data-testid="stNumberInputStepDown"] svg,
  [data-testid="stNumberInputStepUp"] svg,
  [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] button svg {
    fill: var(--cfm-navy) !important;
    stroke: var(--cfm-navy) !important;
  }
  /* Labels de inputs */
  .stApp > section:not([data-testid="stSidebar"]) [data-testid="stWidgetLabel"],
  .stApp > section:not([data-testid="stSidebar"]) .stNumberInput label,
  .stApp > section:not([data-testid="stSidebar"]) .stSelectbox label,
  .stApp > section:not([data-testid="stSidebar"]) .stSlider label {
    color: var(--cfm-primary) !important;
    font-weight: 500 !important;
  }
  /* Texto dentro del selectbox desplegado */
  [data-baseweb="popover"] ul li {
    background-color: #ffffff !important;
    color: var(--cfm-navy) !important;
  }
  [data-baseweb="popover"] ul li:hover {
    background-color: var(--cfm-primary-bg) !important;
  }
  .stSpinner > div { border-top-color: var(--cfm-primary) !important; }

  /* ── Divisor ── */
  hr { border-color: var(--cfm-border) !important; }

  /* ── Métricas dentro del sidebar — fondo oscuro ── */
  html body .stApp [data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: #2a2e42 !important;
    border-left: 3px solid #db0061 !important;
    border-radius: 8px !important;
    padding: 0.35rem 0.5rem !important;
  }

  /* ── Compactar padding interno del sidebar ── */
  [data-testid="stSidebarContent"] > div {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
  }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    gap: 0.25rem !important;
  }

  /* ── Audio input (st.audio_input) — fondo claro, controles visibles ── */
  [data-testid="stAudioInput"],
  [data-testid="stAudioInput"] > div,
  [data-testid="stAudioInput"] section,
  [data-testid="stAudioInput"] [data-testid="stAudioInputWaveformContainer"],
  [data-testid="stAudioInput"] [data-testid="stAudioInputWaveformTimeCode"] {
    background-color: var(--cfm-tertiary-bg) !important;
    border-radius: 8px !important;
  }
  [data-testid="stAudioInput"] {
    border: 1px solid var(--cfm-border) !important;
    padding: 0.25rem !important;
  }
  /* Botón de grabar/detener — circular, color de marca */
  [data-testid="stAudioInput"] button {
    background-color: var(--cfm-primary) !important;
    color: var(--cfm-white) !important;
    border: none !important;
  }
  [data-testid="stAudioInput"] button:hover {
    background-color: var(--cfm-primary-dark) !important;
  }
  [data-testid="stAudioInput"] button svg,
  [data-testid="stAudioInput"] button path {
    fill: var(--cfm-white) !important;
    stroke: var(--cfm-white) !important;
    color: var(--cfm-white) !important;
  }
  /* Texto del temporizador */
  [data-testid="stAudioInput"] [data-testid="stAudioInputWaveformTimeCode"],
  [data-testid="stAudioInput"] span,
  [data-testid="stAudioInput"] p {
    color: var(--cfm-navy) !important;
  }
  /* Forma de onda del audio grabado */
  [data-testid="stAudioInput"] canvas {
    background-color: var(--cfm-tertiary-bg) !important;
  }

</style>
""", unsafe_allow_html=True)

# ── Header con logo Sinfama ───────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; padding:0.5rem 0 1.5rem 0; border-bottom:3px solid #db0061; margin-bottom:1.5rem;">
  <span style="display:inline-flex; width:42px; height:42px; flex-shrink:0;">{LOGO_SVG}</span>
  <div>
    <span style="font-family:'Roboto',sans-serif; font-size:1.4rem; font-weight:700; color:#db0061;">Caja de compensación Sinfama</span>
    <span style="font-family:'Roboto',sans-serif; font-size:1rem; font-weight:400; color:#6f7287; margin-left:0.75rem;">· Agente SQL — Procesos RPA</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 0.25rem 0; text-align:center;">
      <div style="display:inline-block; width:48px; height:48px; margin-bottom:1rem;">{LOGO_SVG}</div><br>
      <span style="font-size:1.05rem; font-weight:700; color:#f15894; font-family:'Roboto',sans-serif; letter-spacing:-0.3px;">Caja de compensación<br>Sinfama</span><br>
      <span style="display:inline-block; margin-top:0.5rem; font-size:0.7rem; color:#b7bad1; font-family:'Roboto',sans-serif;">Agente SQL · Procesos RPA</span>
    </div>
    """, unsafe_allow_html=True)

    # Selector de modelo
    st.markdown('<p style="margin:0.4rem 0 0.1rem; font-size:0.75rem; color:#b7bad1; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Modelo</p>', unsafe_allow_html=True)
    available = list_available_models()
    if available:
        model_options = available
        default_idx = (
            model_options.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in model_options
            else 0
        )
        selected_model = st.selectbox("Modelo Ollama", model_options, index=default_idx, label_visibility="collapsed")
    else:
        st.warning(
            "Ollama no está corriendo.\n"
            "1. https://ollama.com\n"
            "2. `ollama pull qwen2.5-coder:7b`"
        )
        selected_model = DEFAULT_MODEL

    # Estadísticas de la BD
    st.markdown('<p style="margin:0.6rem 0 0.1rem; font-size:0.75rem; color:#b7bad1; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Base de datos</p>', unsafe_allow_html=True)
    try:
        stats = get_db_stats()
        col1, col2 = st.columns(2)
        col1.metric("Ejecuciones", f"{stats['RegistrosDPA_clean']:,}")
        col2.metric("Bots activos", stats["bots_activos"])
        st.caption(f"Período: {stats['fecha_inicio']} → {stats['fecha_fin']}")
    except Exception as e:
        st.error(f"Error cargando stats: {e}")

    # Resumen ROI
    st.markdown('<p style="margin:0.6rem 0 0.25rem; font-size:0.75rem; color:#b7bad1; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Resumen ROI</p>', unsafe_allow_html=True)
    if st.button("Calcular ROI", use_container_width=True):
        with st.spinner("Calculando..."):
            try:
                df_roi = build_roi_dataset()
                summary = get_roi_summary(df_roi)
                st.session_state["roi_summary"] = summary
                st.session_state["roi_df"] = df_roi
            except Exception as e:
                st.error(f"Error: {e}")

    if "roi_summary" in st.session_state:
        s = st.session_state["roi_summary"]
        ahorro_m = s["ahorro_total_cop"] / 1_000_000
        st.metric("Ahorro", f"${ahorro_m:,.0f}M")
        st.caption(f"Top: **{s['mejor_bot']}**")

    # Estado TTS
    st.markdown('<p style="margin:0.6rem 0 0.1rem; font-size:0.75rem; color:#b7bad1; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Voz TTS</p>', unsafe_allow_html=True)
    _tts_ok, _tts_msg = _tts_available()
    st.caption(_tts_msg)

    st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Área principal — pestañas ─────────────────────────────────────────────────
tab_chat, tab_roi, tab_calc, tab_model = st.tabs([
    "💬 Chat SQL",
    "📊 Análisis ROI",
    "🧮 Calculadora de ROI",
    "🤖 Modelo Predictivo (experimental)",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("### Haz preguntas sobre los datos de procesos RPA")
    st.caption(
        "Ejemplos: *¿Cuáles son los 5 bots con más ejecuciones?* · "
        "*¿Qué área tiene mayor tasa de error?* · "
        "*¿Cuánto tiempo manual ahorró GestorRemitidos?*"
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Historial de mensajes
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("Ver SQL ejecutado"):
                    st.code(msg["sql"], language="sql")
            if msg.get("data") and msg.get("columns"):
                df_display = pd.DataFrame(msg["data"], columns=msg["columns"])
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            if msg["role"] == "assistant" and msg.get("audio"):
                audio_play_button(msg["audio"], key=f"h{i}")

    # Entrada por voz (opcional) — usa componente nativo de Streamlit
    col_mic, col_hint = st.columns([2, 3])
    with col_mic:
        audio = st.audio_input("🎤 Graba tu pregunta", key="mic_chat")
    with col_hint:
        st.caption(
            "Habla en español. Al detener la grabación, se transcribe localmente "
            "con Whisper y se envía al agente SQL."
        )

    voice_prompt = None
    if audio is not None:
        audio_bytes = audio.getvalue()
        audio_hash = hash(audio_bytes)
        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state["last_audio_hash"] = audio_hash
            with st.spinner("Transcribiendo audio..."):
                try:
                    voice_prompt = transcribe_audio(audio_bytes)
                except Exception as exc:
                    st.error(f"Error al transcribir: {exc}")
                    voice_prompt = None
            if not voice_prompt:
                st.warning(
                    "No detecté palabras en el audio. Intenta hablar más claro o más cerca del micrófono."
                )

    # Entrada por texto
    text_prompt = st.chat_input("Pregunta algo sobre los datos...")

    prompt = voice_prompt or text_prompt

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Consultando con {selected_model}..."):
                result = ask(
                    question=prompt,
                    history=st.session_state["messages"],
                    model=selected_model,
                )

            st.markdown(result["response"])

            if result.get("sql"):
                with st.expander("Ver SQL ejecutado"):
                    st.code(result["sql"], language="sql")

            if result.get("data") and result.get("columns"):
                df_result = pd.DataFrame(result["data"], columns=result["columns"])
                st.dataframe(df_result, use_container_width=True, hide_index=True)

            # Síntesis de voz TTS
            audio_bytes = None
            if _tts_available()[0]:
                with st.spinner("Sintetizando audio..."):
                    audio_bytes = synthesize_speech(result["response"])
            if audio_bytes:
                audio_play_button(audio_bytes, key=f"n{len(st.session_state['messages'])}")

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["response"],
                "sql": result.get("sql"),
                "data": result.get("data"),
                "columns": result.get("columns"),
                "audio": audio_bytes,
            }
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · ANÁLISIS ROI
# ─────────────────────────────────────────────────────────────────────────────
with tab_roi:
    import plotly.express as px

    st.markdown("### Análisis de ROI por automatización")

    if "roi_df" not in st.session_state:
        st.info(
            "Haz clic en **Calcular ROI** en el sidebar para cargar el análisis."
        )
    else:
        df_roi = st.session_state["roi_df"]
        df_valid = df_roi.dropna(subset=["ROI_Porcentaje", "Ahorro_Neto_COP"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 bots por ROI (%)")
            top_roi = df_valid.nlargest(10, "ROI_Porcentaje")[
                ["Automatizacion", "ROI_Porcentaje", "Ahorro_Neto_COP", "Num_Ejecuciones"]
            ].copy()
            top_roi["Ahorro_M_COP"] = top_roi["Ahorro_Neto_COP"] / 1e6
            fig = px.bar(
                top_roi,
                x="ROI_Porcentaje",
                y="Automatizacion",
                orientation="h",
                color="ROI_Porcentaje",
                color_continuous_scale="Greens",
                labels={"ROI_Porcentaje": "ROI (%)", "Automatizacion": "Bot"},
            )
            fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Top 10 bots por Ahorro Neto (COP)")
            top_ahorro = df_valid.nlargest(10, "Ahorro_Neto_COP")[
                ["Automatizacion", "Ahorro_Neto_COP", "ROI_Porcentaje"]
            ].copy()
            top_ahorro["Ahorro_M_COP"] = top_ahorro["Ahorro_Neto_COP"] / 1e6
            fig2 = px.bar(
                top_ahorro,
                x="Ahorro_M_COP",
                y="Automatizacion",
                orientation="h",
                color="Ahorro_M_COP",
                color_continuous_scale="Blues",
                labels={"Ahorro_M_COP": "Ahorro (M COP)", "Automatizacion": "Bot"},
            )
            fig2.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter ROI vs Ejecuciones
        st.markdown("#### ROI vs Volumen de ejecuciones")
        df_scatter = df_valid.copy()
        df_scatter["Tamaño"] = df_scatter["Ahorro_Neto_COP"].clip(lower=0)
        fig3 = px.scatter(
            df_scatter,
            x="Num_Ejecuciones",
            y="ROI_Porcentaje",
            size="Tamaño",
            size_max=50,
            color="Tecnologia",
            hover_name="Automatizacion",
            labels={
                "Num_Ejecuciones": "Número de ejecuciones",
                "ROI_Porcentaje": "ROI (%)",
            },
            log_x=True,
        )
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

        # Tabla completa
        st.markdown("#### Detalle por bot")
        display_cols = [
            "Automatizacion", "Tecnologia", "Estado",
            "Num_Ejecuciones", "TiempoManualHoras",
            "DuracionPromedio_Horas", "ROI_Porcentaje",
            "Ahorro_Neto_COP", "Beneficio_Bruto_COP",
        ]
        cols_present = [c for c in display_cols if c in df_roi.columns]
        df_display = df_roi[cols_present].copy()
        df_display["ROI_Porcentaje"] = df_display["ROI_Porcentaje"].round(1)
        df_display["Ahorro_Neto_COP"] = df_display["Ahorro_Neto_COP"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · PREDICCIÓN ROI
# ─────────────────────────────────────────────────────────────────────────────
with tab_calc:
    st.markdown("### Calculadora de ROI para un nuevo bot")
    st.caption(
        f"Calcula el ROI esperado de un nuevo proceso RPA aplicando la fórmula del "
        f"negocio. El costo operativo del robot se estandariza en "
        f"**{COSTO_HORA_ROBOT_COP:,} COP/hora** (servidor Azure + licencia UiPath)."
    )

    col_f, col_r = st.columns([1, 1])

    with col_f:
        st.markdown("#### Parámetros del proceso")
        tiempo_manual = st.number_input(
            "Tiempo manual por ejecución (horas)", min_value=0.01, value=2.0, step=0.25
        )
        num_ejecuciones = st.number_input(
            "Ejecuciones esperadas (total)", min_value=1, value=500, step=50
        )
        valor_hora = st.number_input(
            "Valor hora del rol humano (COP)", min_value=5000, value=30000, step=1000
        )
        duracion_robot = st.number_input(
            "Duración estimada del robot (horas)", min_value=0.001, value=0.15, step=0.05
        )

    with col_r:
        st.markdown("#### Resultado calculado")
        if st.button("Calcular ROI", type="primary", use_container_width=True):
            beneficio = tiempo_manual * valor_hora * num_ejecuciones
            costo = duracion_robot * COSTO_HORA_ROBOT_COP * num_ejecuciones
            ahorro = beneficio - costo
            roi = (ahorro / costo * 100) if costo > 0 else 0.0

            color = "normal" if roi > 0 else "inverse"
            st.metric("ROI calculado", f"{roi:,.0f}%", delta=f"{roi:,.0f}%", delta_color=color)
            st.metric("Ahorro neto", f"${ahorro / 1e6:,.2f}M COP")
            st.metric("Beneficio bruto", f"${beneficio / 1e6:,.2f}M COP")
            st.metric("Costo total del robot", f"${costo / 1e6:,.2f}M COP")

            with st.expander("Ver fórmula aplicada"):
                st.code(
                    f"Beneficio = {tiempo_manual} h × {valor_hora:,} COP/h × {num_ejecuciones:,}\n"
                    f"          = {beneficio:,.0f} COP\n\n"
                    f"Costo     = {duracion_robot} h × {COSTO_HORA_ROBOT_COP:,} COP/h × {num_ejecuciones:,}\n"
                    f"          = {costo:,.0f} COP\n\n"
                    f"Ahorro    = Beneficio − Costo = {ahorro:,.0f} COP\n"
                    f"ROI%      = (Ahorro / Costo) × 100 = {roi:,.1f}%",
                    language="text",
                )

            if roi > 500:
                st.success("Excelente oportunidad de automatización.")
            elif roi > 100:
                st.info("Buena candidata para automatización.")
            elif roi > 0:
                st.warning("ROI positivo pero bajo. Considera optimizar el proceso.")
            else:
                st.error("ROI negativo. Este proceso puede no ser rentable para RPA.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · MODELO PREDICTIVO (EXPERIMENTAL)
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    import json
    import plotly.express as px

    st.markdown("### Modelo predictivo de ROI — XGBoost (experimental)")

    st.warning(
        "**⚠️ Esta pestaña es exploratoria, no operativa.** El modelo fue entrenado "
        "con solo ~30 bots con datos completos en las tres tablas, lo que produce "
        "alta varianza en validación cruzada. La pestaña 3 (Calculadora) usa la "
        "**fórmula determinística** del negocio, no este modelo. Aquí mostramos el "
        "modelo para cerrar el ciclo análisis → producto y para inspección de "
        "importancia de variables."
    )

    MODELS_DIR = Path(__file__).parent.parent / "models"
    REPORTS_DIR = Path(__file__).parent.parent / "reports"
    MODEL_FILE = MODELS_DIR / "roi_model.joblib"
    METRICS_FILE = REPORTS_DIR / "metrics_roi.json"

    if not MODEL_FILE.exists():
        st.info(
            "**El modelo aún no se ha entrenado.** Ejecuta uno de los siguientes "
            "para generarlo:\n\n"
            "```bash\n"
            "uv run python scripts/train_roi_model.py\n"
            "```\n\n"
            "O abre `notebooks/03_modelo_roi.ipynb` desde Jupyter."
        )
    else:
        try:
            import joblib  # noqa: PLC0415
            from src.models.roi_predictor import get_feature_importance  # noqa: PLC0415

            artifact = joblib.load(MODEL_FILE)
            pipeline = artifact["pipeline"]
            saved_metrics = artifact.get("metrics", {})
        except Exception as exc:
            st.error(f"No se pudo cargar el modelo: {exc}")
            st.stop()

        # Si hay reports/metrics_roi.json, mostrarlo (más reciente que el embebido)
        metrics: dict = saved_metrics
        if METRICS_FILE.exists():
            try:
                metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        st.markdown("#### Métricas del último entrenamiento")
        n_total = metrics.get("n_total", metrics.get("n_train", 0) + metrics.get("n_test", 0))
        if n_total and n_total < 50:
            st.error(
                f"**n_total = {n_total}** muestras. "
                "Con menos de 50 muestras y 13 features, cualquier modelo tabular "
                "estará dominado por el ruido. Lee las métricas con escepticismo: "
                "el CV R² es lo único que importa."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "R² (test, log)",
            f"{metrics.get('r2', 0):.3f}" if "r2" in metrics else "N/A",
            help="Sobre el split de test. Con n_test ≈ 6 es muy inestable.",
        )
        ci_low = metrics.get("r2_test_ci95_low")
        ci_high = metrics.get("r2_test_ci95_high")
        if ci_low is not None and ci_high is not None:
            m2.metric(
                "R² CI95% (bootstrap)",
                f"[{ci_low:.2f}, {ci_high:.2f}]",
                help="Intervalo de confianza al 95% via bootstrap del split de test.",
            )
        m3.metric(
            "R² CV-5 media ± std",
            f"{metrics.get('cv_r2_mean', 0):.3f} ± {metrics.get('cv_r2_std', 0):.3f}"
            if "cv_r2_mean" in metrics else "N/A",
            help="Esta es la métrica que importa. Cerca de cero ⇒ el modelo no generaliza.",
        )
        m4.metric(
            "MAE (% ROI)",
            f"{metrics.get('mae_pct', 0):,.0f}" if "mae_pct" in metrics else "N/A",
            help="Error absoluto medio en puntos porcentuales de ROI.",
        )

        # Baseline Ridge (sanity check)
        if "baseline_r2" in metrics:
            st.markdown("#### Baseline · Ridge sobre log(ROI)")
            st.caption(
                "Sanity check: ¿XGBoost supera de forma consistente a una regresión lineal regularizada? "
                "Si no la supera en CV (la métrica más confiable con n pequeño), XGBoost no aporta valor "
                "y la fórmula determinística sigue siendo la mejor opción operativa."
            )
            b1, b2, b3 = st.columns(3)
            b1.metric("Baseline R² (test, log)", f"{metrics['baseline_r2']:.3f}")
            b2.metric(
                "Baseline R² CV-5",
                f"{metrics['baseline_cv_r2_mean']:.3f} ± {metrics['baseline_cv_r2_std']:.3f}",
            )
            b3.metric("Baseline MAE", f"{metrics['baseline_mae_pct']:,.0f}")

            delta_cv = metrics["cv_r2_mean"] - metrics["baseline_cv_r2_mean"]
            if delta_cv > 0.05:
                st.info(
                    f"XGBoost supera al baseline en CV por **+{delta_cv:.3f}** R². "
                    "Aporta señal sobre la lineal, aunque con n bajo la diferencia puede no ser robusta."
                )
            else:
                st.warning(
                    f"XGBoost solo supera al baseline en CV por **{delta_cv:+.3f}** R². "
                    "Con esta diferencia, no hay evidencia de que el modelo no-lineal aporte sobre Ridge. "
                    "La fórmula determinística sigue siendo la opción operativa correcta."
                )

        # Importancia de variables
        st.markdown("#### Importancia de variables (XGBoost)")
        try:
            imp = get_feature_importance(top_n=15, pipeline=pipeline)
            fig_imp = px.bar(
                imp.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Magma",
                labels={"importance": "Importancia", "feature": "Variable"},
            )
            fig_imp.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Las variables que dominan (TiempoManualHoras, ValorHoraPromedio, "
                "DuracionPromedio_Horas, TasaExito) son exactamente las que componen "
                "la fórmula determinística del ROI. El modelo *redescubre la fórmula* "
                "desde los datos en lugar de aprender una relación nueva."
            )
        except Exception as exc:
            st.error(f"No se pudo calcular feature importance: {exc}")

        # Predicción puntual (con disclaimer)
        st.markdown("#### Predicción puntual (con advertencia)")
        st.caption(
            "Esta predicción se ofrece **solo con fines exploratorios**. Para decisiones "
            "operativas, usa la pestaña 'Calculadora de ROI' que aplica la fórmula auditable."
        )

        with st.form("model_predict"):
            c1, c2 = st.columns(2)
            with c1:
                p_tiempo = st.number_input("TiempoManualHoras", min_value=0.01, value=2.0, step=0.25)
                p_ejec = st.number_input("Num_Ejecuciones", min_value=1, value=500, step=50)
                p_dur = st.number_input("DuracionPromedio_Horas", min_value=0.001, value=0.2, step=0.05)
                p_trans = st.number_input("PromTransacciones", min_value=0.0, value=10.0, step=1.0)
                p_exito = st.slider("TasaExito", 0.0, 1.0, 0.95, 0.01)
                p_error = st.slider("TasaError", 0.0, 1.0, 0.02, 0.01)
            with c2:
                p_valor = st.number_input("ValorHoraPromedio (COP)", min_value=5000, value=30000, step=1000)
                p_ejdia = st.number_input("EjecucionesPorDia", min_value=0.01, value=2.0, step=0.5)
                p_dias = st.number_input("DiasEnProduccion", min_value=1, value=180, step=30)
                p_areas = st.number_input("NumAreas", min_value=1, value=1, step=1)
                p_roles = st.number_input("NumRoles", min_value=1, value=1, step=1)
                p_tec = st.selectbox("Tecnologia", ["UiPath", "IRPA", "Power Automate", "Desconocida"])
                p_est = st.selectbox("Estado", ["Activo", "Inactivo", "Desconocido"])

            predict_btn = st.form_submit_button("Predecir ROI con el modelo", type="primary")

        if predict_btn:
            row = pd.DataFrame([{
                "TiempoManualHoras": p_tiempo,
                "Num_Ejecuciones": p_ejec,
                "DuracionPromedio_Horas": p_dur,
                "PromTransacciones": p_trans,
                "TasaExito": p_exito,
                "TasaError": p_error,
                "ValorHoraPromedio": p_valor,
                "EjecucionesPorDia": p_ejdia,
                "DiasEnProduccion": p_dias,
                "NumAreas": p_areas,
                "NumRoles": p_roles,
                "Tecnologia": p_tec,
                "Estado": p_est,
            }])

            import numpy as np  # noqa: PLC0415
            try:
                pred_log = pipeline.predict(row)[0]
                pred_pct = float(np.expm1(pred_log))

                # Referencia de la fórmula
                beneficio = p_tiempo * p_valor * p_ejec
                costo = p_dur * COSTO_HORA_ROBOT_COP * p_ejec
                roi_formula = (beneficio - costo) / costo * 100 if costo > 0 else 0

                c1, c2 = st.columns(2)
                c1.metric("ROI predicho (XGBoost)", f"{pred_pct:,.0f}%")
                c2.metric(
                    "ROI fórmula (referencia)",
                    f"{roi_formula:,.0f}%",
                    delta=f"{pred_pct - roi_formula:+,.0f} pp",
                    help="Diferencia entre la predicción del modelo y la fórmula del negocio.",
                )
                st.info(
                    "Si la diferencia es grande, **confía en la fórmula** — el modelo "
                    "tiene CV R² cerca de cero y puede dar valores poco confiables. "
                    "La predicción del modelo se ofrece para comprender qué variables "
                    "lo mueven, no para sustituir el cálculo del negocio."
                )
            except Exception as exc:
                st.error(f"No se pudo predecir: {exc}")
