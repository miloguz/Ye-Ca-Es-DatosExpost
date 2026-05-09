"""
Interfaz Streamlit para el Agente SQL de Procesos RPA.
Corre localmente con Ollama — sin API keys ni conexión a internet.

Iniciar:
    streamlit run app/chat.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.database import get_db_stats
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

# ── Logo ──────────────────────────────────────────────────────────────────────
LOGO_SVG = (Path(__file__).parent.parent / "assets" / "logo.svg").read_text(encoding="utf-8")

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sinfama · Agente RPA",
    page_icon=str(Path(__file__).parent.parent / "assets" / "logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Área principal — pestañas ─────────────────────────────────────────────────
tab_chat, tab_roi, tab_calc = st.tabs(["💬 Chat SQL", "📊 Análisis ROI", "🧮 Calculadora de ROI"])

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
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("Ver SQL ejecutado"):
                    st.code(msg["sql"], language="sql")
            if msg.get("data") and msg.get("columns"):
                df_display = pd.DataFrame(msg["data"], columns=msg["columns"])
                st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Entrada por voz (opcional)
    col_mic, col_hint = st.columns([1, 4])
    with col_mic:
        audio = mic_recorder(
            start_prompt="🎤 Grabar pregunta",
            stop_prompt="⏹️ Detener",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="mic_chat",
        )
    with col_hint:
        st.caption(
            "Habla tu pregunta en español. Al detener, se transcribe localmente "
            "con Whisper y se envía al agente SQL."
        )

    voice_prompt = None
    if audio and audio.get("bytes"):
        if audio.get("id") != st.session_state.get("last_audio_id"):
            st.session_state["last_audio_id"] = audio.get("id")
            with st.spinner("Transcribiendo audio..."):
                try:
                    voice_prompt = transcribe_audio(audio["bytes"])
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

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["response"],
                "sql": result.get("sql"),
                "data": result.get("data"),
                "columns": result.get("columns"),
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
