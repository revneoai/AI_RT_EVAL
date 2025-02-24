@echo off
echo AI Evaluation Platform - Developer Setup
echo =====================================

:: Set absolute paths
set "VENV_PATH=%CD%\venv"
set "PYTHON_PATH=%VENV_PATH%\Scripts\python.exe"
set "PIP_PATH=%VENV_PATH%\Scripts\pip.exe"

:: Check if running for the first time
if not exist "venv" (
    echo First time setup...
    call check_structure.bat
    python -m venv venv
    call "%VENV_PATH%\Scripts\activate"
    "%PIP_PATH%" install --upgrade pip
    
    echo Installing core dependencies...
    "%PIP_PATH%" install -r requirements.txt
    
    echo Installing additional dependencies...
    :: Install dask components separately to avoid conflicts
    "%PIP_PATH%" install "dask[complete]>=2023.3.0" --no-deps
    "%PIP_PATH%" install distributed>=2023.3.0
    "%PIP_PATH%" install fsspec>=2023.3.0
    "%PIP_PATH%" install cloudpickle>=3.0.0
    
    echo Installing GitHub helper dependencies...
    "%PIP_PATH%" install inquirer>=3.1.3 colorama>=0.4.6
    
    echo Setup complete!
    echo.
) else (
    echo Virtual environment exists, checking for updates...
    call "%VENV_PATH%\Scripts\activate"
    
    :: Only update if requested
    set /p UPDATE_DEPS="Update dependencies? (y/N): "
    if /i "%UPDATE_DEPS%"=="y" (
        echo Updating dependencies...
        "%PIP_PATH%" install -r requirements.txt --no-deps
        "%PIP_PATH%" install "dask[complete]>=2023.3.0" --no-deps
        "%PIP_PATH%" install distributed>=2023.3.0 fsspec>=2023.3.0 cloudpickle>=3.0.0
        echo Dependencies updated!
    )
)

:: Verify Dask installation in the virtual environment
echo Verifying Dask installation...
"%PYTHON_PATH%" -c "import dask.dataframe as dd; print('Dask verified')" || (
    echo Dask not found, running fix_dask.py...
    "%PYTHON_PATH%" fix_dask.py
)

:: Add project root to PYTHONPATH
set "PYTHONPATH=%CD%"

:: Start services
echo Starting services...
echo.
echo API server will start on http://localhost:8000
echo Dashboard will start on http://localhost:8501
echo.
echo Press Ctrl+C in respective windows to stop services
echo.

:: Start API server in new window with absolute paths
start cmd /k "title API Server && call "%VENV_PATH%\Scripts\activate" && set PYTHONPATH=%PYTHONPATH% && "%PYTHON_PATH%" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Start dashboard in new window with absolute paths
start cmd /k "title Dashboard && call "%VENV_PATH%\Scripts\activate" && set PYTHONPATH=%PYTHONPATH% && "%PYTHON_PATH%" -m streamlit run ai_collaborative_platform/interfaces/dashboard/app.py"

echo Services are starting...
echo.
echo Quick links:
echo - API docs: http://localhost:8000/docs
echo - Dashboard: http://localhost:8501

:: Add dependency check
echo.
echo Checking critical dependencies...
"%PYTHON_PATH%" -c "import dask.dataframe as dd; import distributed; print('Dask installation: OK')" 2>nul || echo Warning: Dask not properly installed
"%PYTHON_PATH%" -c "import streamlit; print('Streamlit installation: OK')" 2>nul || echo Warning: Streamlit not properly installed 