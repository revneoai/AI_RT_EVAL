@echo off
echo AI Evaluation Platform - Developer Setup
echo =====================================

:: Check if running for the first time
if not exist "venv" (
    echo First time setup...
    call check_structure.bat
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo Setup complete!
    echo.
)

:: Activate virtual environment
call venv\Scripts\activate

:: Add project root to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Start services
echo Starting services...
echo.
echo API server will start on http://localhost:8000
echo Dashboard will start on http://localhost:8501
echo.
echo Press Ctrl+C in respective windows to stop services
echo.

:: Start API server in new window
start cmd /k "title API Server && venv\Scripts\activate && set PYTHONPATH=%PYTHONPATH%;%CD% && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Start dashboard in new window
start cmd /k "title Dashboard && venv\Scripts\activate && set PYTHONPATH=%PYTHONPATH%;%CD% && streamlit run ai_collaborative_platform/interfaces/dashboard/app.py"

echo Services are starting...
echo.
echo Quick links:
echo - API docs: http://localhost:8000/docs
echo - Dashboard: http://localhost:8501 