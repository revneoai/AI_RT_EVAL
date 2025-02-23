@echo off
echo Starting AI Collaborative Platform...
echo.
echo Please follow these steps in order:
echo ================================
echo 1. Setup: Run 'setup.bat'
echo 2. API Server: Run 'run.bat'
echo 3. Dashboard: Run 'run_dashboard.bat' in a new terminal
echo.
echo Opening required terminals...

start cmd /k "echo Terminal 1 - Setup and API Server && echo Run: setup.bat, then run.bat"
start cmd /k "echo Terminal 2 - Dashboard && echo Run: run_dashboard.bat" 