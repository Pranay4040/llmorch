@echo off
rem Run llmorch from a checkout without activating the venv or fixing PATH.
rem
rem `pip install -e .` puts an `llmorch` executable in .venv\Scripts, which only
rem works bare if that folder is on PATH — so the first thing anyone meets is
rem "'llmorch' is not recognized", which is a worse introduction than the tool
rem deserves. From the repository root, `llmorch` finds this file instead.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
    "%HERE%.venv\Scripts\python.exe" -m llmorch %*
) else (
    python -m llmorch %*
)
exit /b %ERRORLEVEL%
