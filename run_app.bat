@echo off
echo.
echo =========================================
echo   Agente SQL - Procesos RPA
echo   Iniciando aplicacion Streamlit...
echo =========================================
echo.
echo Asegurate de tener Ollama corriendo:
echo   ollama serve  (en otra terminal)
echo   ollama pull qwen2.5-coder:7b
echo.
streamlit run app/chat.py
