@echo off
echo Starting Addis Traffic Detection API...
cd /d "%~dp0"
call venv\Scripts\activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
