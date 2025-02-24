@echo off
echo Setting up AI Collaborative Platform...

:: Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Run the project setup script
python setup_project.py

:: Create necessary directories
if not exist logs mkdir logs
if not exist data mkdir data

echo Setup complete! Run 'run.bat' to start the application 