@echo off
cd /d %~dp0
call .venv\Scripts\activate
python src\step_04_test_recommendation.py
pause
