@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\run_pipeline.py
pause
