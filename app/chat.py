"""
Interfaz Streamlit para el Agente SQL de Procesos RPA.
Corre localmente con Ollama — sin API keys ni conexión a internet.

Iniciar:
    streamlit run app/chat.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.database import get_db_stats
from src.agent.sql_agent import DEFAULT_MODEL, ask, list_available_models
from src.utils.roi_calculator import build_roi_dataset, get_roi_summary

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente RPA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agente SQL · RPA")
    st.caption("Powered by Ollama · 100% local")
    st.divider()

    # Selector de modelo
    st.subheader("Modelo")
    available = list_available_models()
    if available:
        model_options = available
        default_idx = (
            model_options.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in model_options
            else 0
        )
        selected_model = st.selectbox("Modelo Ollama", model_options, index=default_idx)
    else:
        st.warning(
            "Ollama no está corriendo o no hay modelos descargados.\n\n"
            "**Pasos:**\n"
            "1. Instala Ollama: https://ollama.com\n"
            "2. `ollama pull qwen2.5-coder:7b`\n"
            "3. `ollama serve`"
        )
        selected_model = DEFAULT_MODEL

    st.divider()

    # Estadísticas de la BD
    st.subheader("Base de datos")
    try:
        stats = get_db_stats()
        col1, col2 = st.columns(2)
        col1.metric("Ejecuciones", f"{stats['RegistrosDPA_clean']:,}")
        col2.metric("Bots activos", stats["bots_activos"])
        st.caption(f"Período: {stats['fecha_inicio']} → {stats['fecha_fin']}")
    except Exception as e:
        st.error(f"Error cargando stats: {e}")

    st.divider()

    # Resumen ROI
    st.subheader("Resumen ROI")
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
        st.metric("Ahorro total", f"${ahorro_m:,.1f}M COP")
        st.metric("ROI promedio", f"{s['roi_promedio_pct']:.0f}%")
        st.metric("Tiempo ahorrado", f"{s['tiempo_ahorrado_horas']:,.0f} h")
        st.caption(f"Mejor bot: **{s['mejor_bot']}**")

    st.divider()

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Área principal — pestañas ─────────────────────────────────────────────────
tab_chat, tab_roi, tab_predict = st.tabs(["💬 Chat SQL", "📊 Análisis ROI", "🔮 Predicción ROI"])

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

    # Input del usuario
    if prompt := st.chat_input("Pregunta algo sobre los datos..."):
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
with tab_predict:
    st.markdown("### Predicción de ROI para un nuevo bot")
    st.caption(
        "Ingresa las características de un nuevo proceso RPA para estimar su ROI "
        "antes de implementarlo. Requiere haber entrenado el modelo primero "
        "(notebook `03_modelo_roi.ipynb`)."
    )

    col_f, col_r = st.columns([1, 1])

    with col_f:
        st.markdown("#### Características del bot")
        tiempo_manual = st.number_input(
            "Tiempo manual por ejecución (horas)", min_value=0.01, value=2.0, step=0.25
        )
        num_ejecuciones = st.number_input(
            "Ejecuciones esperadas (total)", min_value=1, value=500, step=50
        )
        valor_hora = st.number_input(
            "Valor hora del rol (COP)", min_value=5000, value=30000, step=1000
        )
        tecnologia = st.selectbox(
            "Tecnología RPA", ["UiPath", "Power Automate", "IRPA", "Desconocida"]
        )
        duracion_robot = st.number_input(
            "Duración estimada del robot (horas)", min_value=0.001, value=0.15, step=0.05
        )
        estado = st.selectbox("Estado esperado", ["Activo", "Inactivo"])

    with col_r:
        st.markdown("#### Resultado estimado")
        if st.button("Predecir ROI", type="primary", use_container_width=True):
            try:
                from src.models.roi_predictor import predict

                result = predict(
                    {
                        "TiempoManualHoras": tiempo_manual,
                        "Num_Ejecuciones": num_ejecuciones,
                        "ValorHoraPromedio": valor_hora,
                        "Tecnologia": tecnologia,
                        "Estado": estado,
                        "DuracionPromedio_Horas": duracion_robot,
                        "PromTransacciones": 5.0,
                        "TasaExito": 0.9,
                        "TasaError": 0.05,
                        "EjecucionesPorDia": num_ejecuciones / 365,
                        "DiasEnProduccion": 365,
                        "NumAreas": 1,
                        "NumRoles": 1,
                    }
                )
                roi = result["roi_porcentaje"]
                ahorro = result["ahorro_neto_cop"]
                beneficio = result["beneficio_bruto_cop"]
                costo = result["costo_robot_cop"]

                color = "normal" if roi > 0 else "inverse"
                st.metric("ROI predicho", f"{roi:.0f}%", delta=f"{roi:.0f}%", delta_color=color)
                st.metric("Ahorro neto estimado", f"${ahorro / 1e6:.2f}M COP")
                st.metric("Beneficio bruto", f"${beneficio / 1e6:.2f}M COP")
                st.metric("Costo estimado robot", f"${costo / 1e6:.2f}M COP")

                if roi > 500:
                    st.success("Excelente oportunidad de automatización.")
                elif roi > 100:
                    st.info("Buena candidata para automatización.")
                elif roi > 0:
                    st.warning("ROI positivo pero bajo. Considera optimizar el proceso.")
                else:
                    st.error("ROI negativo. Este proceso puede no ser rentable para RPA.")

            except FileNotFoundError:
                st.error(
                    "El modelo no está entrenado aún.\n\n"
                    "Ejecuta el notebook `notebooks/03_modelo_roi.ipynb` primero."
                )
            except Exception as e:
                st.error(f"Error al predecir: {e}")
