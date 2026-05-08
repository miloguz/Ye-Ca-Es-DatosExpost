"""
Agente SQL que usa Ollama (LLM local) para responder preguntas
en lenguaje natural sobre la base de datos de procesos RPA.

Flujo:
  1. El usuario hace una pregunta.
  2. El LLM genera SQL si la pregunta requiere datos.
  3. El agente ejecuta el SQL contra Procesos_clean.db.
  4. El LLM interpreta los resultados y responde en español.
"""

import logging
import re

import ollama
import pandas as pd

from .database import execute_query, get_schema

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"
MAX_HISTORY_MESSAGES = 8
MAX_RESULT_ROWS_IN_PROMPT = 30

DEFAULT_FALLBACK_MESSAGE = (
    "Lo siento, no fue posible responder esta duda. "
    "Intenta reformular la pregunta o consultar otra información."
)

_SYSTEM_PROMPT_TEMPLATE = """Eres un experto en análisis de datos de procesos RPA (Robotic Process Automation)
para una organización de salud en Colombia. Puedes consultar una base de datos SQLite
con el historial de ejecuciones de bots, tiempos manuales y ROI estimado.

{schema}

=== INSTRUCCIONES ===
- Si la pregunta requiere consultar datos, genera UNA sola query SQL válida para SQLite,
  envuelta exactamente en ```sql ... ```.
- Si la pregunta NO requiere SQL (saludos, conceptos generales), responde directamente.
- Nunca inventes datos; basa las respuestas SOLO en los resultados de la query.
- Responde SIEMPRE en español, de forma clara y concisa.
- Para ROI usa: Beneficio = TiempoManualHoras × ValorHoraProyecto × Num_Ejecuciones.
- Si el usuario pide un ranking, limita con LIMIT 10 salvo que indique otro número.
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(schema=get_schema())


def _extract_sql(text: str) -> str | None:
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(SELECT|WITH|INSERT|UPDATE)\b.*?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip("`").strip()
    return None


def _format_df_for_prompt(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "La consulta no retornó resultados."
    preview = df.head(MAX_RESULT_ROWS_IN_PROMPT)
    lines = [" | ".join(str(v) for v in row) for row in preview.itertuples(index=False)]
    header = " | ".join(df.columns)
    return header + "\n" + "\n".join(lines)


def ask(question: str, history: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """
    Procesa una pregunta del usuario y retorna un diccionario con:
      - response (str): respuesta en lenguaje natural
      - sql (str | None): SQL generado y ejecutado
      - data (list[dict] | None): filas resultado de la query
      - columns (list[str] | None): nombres de columnas
      - error (str | None): mensaje de error si ocurrió
    """
    messages = [{"role": "system", "content": _build_system_prompt()}]

    for msg in history[-(MAX_HISTORY_MESSAGES):]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    try:
        step1 = ollama.chat(model=model, messages=messages)
        raw = step1["message"]["content"]
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ("refused", "connect", "connection")):
            msg = (
                "No se puede conectar a Ollama. "
                "Asegúrate de tener Ollama instalado y corriendo:\n\n"
                "```\nollama serve\n```"
            )
        else:
            msg = DEFAULT_FALLBACK_MESSAGE
        return {"response": msg, "sql": None, "data": None, "columns": None, "error": err}

    sql = _extract_sql(raw)
    if not sql:
        return {"response": raw, "sql": None, "data": None, "columns": None, "error": None}

    try:
        df = execute_query(sql)
    except Exception as e:
        logger.warning("SQL execution failed: %s\nSQL: %s", e, sql)
        return {
            "response": DEFAULT_FALLBACK_MESSAGE,
            "sql": None,
            "data": None,
            "columns": None,
            "error": str(e),
        }

    result_text = _format_df_for_prompt(df)
    interp_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"Los resultados de la consulta son:\n\n{result_text}\n\n"
                "Interpreta estos resultados de forma clara y concisa en español. "
                "No repitas el SQL. Resalta los puntos más relevantes."
            ),
        },
    ]

    try:
        step2 = ollama.chat(model=model, messages=interp_messages)
        final_response = step2["message"]["content"]
    except Exception:
        final_response = f"Consulta ejecutada. Resultados con {len(df)} filas."

    return {
        "response": final_response,
        "sql": sql,
        "data": df.to_dict("records"),
        "columns": list(df.columns),
        "error": None,
    }


def list_available_models() -> list[str]:
    """Retorna los modelos Ollama disponibles localmente."""
    try:
        models = ollama.list()
        return [m["model"] for m in models.get("models", [])]
    except Exception as e:
        logger.info("No se pudo listar modelos de Ollama: %s", e)
        return []
