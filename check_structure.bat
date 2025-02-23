@echo off
echo Checking project structure...

:: Create main project directory if it doesn't exist
if not exist "ai_collaborative_platform" (
    echo Creating project structure...
    mkdir ai_collaborative_platform
    mkdir ai_collaborative_platform\core
    mkdir ai_collaborative_platform\core\evaluation
    mkdir ai_collaborative_platform\interfaces
    mkdir ai_collaborative_platform\interfaces\api
    mkdir ai_collaborative_platform\interfaces\dashboard
)

:: Create __init__.py files
echo Creating __init__.py files...
echo. > ai_collaborative_platform\__init__.py
echo. > ai_collaborative_platform\core\__init__.py
echo. > ai_collaborative_platform\core\evaluation\__init__.py
echo. > ai_collaborative_platform\interfaces\__init__.py
echo. > ai_collaborative_platform\interfaces\api\__init__.py
echo. > ai_collaborative_platform\interfaces\dashboard\__init__.py

:: Move files to correct locations if they exist
if exist "backend\core\evaluation\drift_detector.py" (
    move "backend\core\evaluation\drift_detector.py" "ai_collaborative_platform\core\evaluation\"
)
if exist "backend\core\evaluation\anomaly_detector.py" (
    move "backend\core\evaluation\anomaly_detector.py" "ai_collaborative_platform\core\evaluation\"
)

echo Structure check complete! 