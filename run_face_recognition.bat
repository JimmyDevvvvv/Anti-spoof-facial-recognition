@echo off
echo ========================================
echo    Face Recognition System Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
pip show opencv-contrib-python >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
    echo Requirements installed!
    echo.
) else (
    echo Checking OpenCV version...
    python -c "import cv2; print('Current OpenCV version:', cv2.__version__)"
    python -c "import cv2; exit(0 if cv2.__version__.startswith('4.10') else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Fixing OpenCV version...
        pip uninstall opencv-python opencv-contrib-python -y
        pip install opencv-contrib-python==4.10.0.84
        echo OpenCV version fixed!
        echo.
    ) else (
        echo Dependencies are up to date!
        echo.
    )
)

REM Display menu
:menu
echo ========================================
echo           AVAILABLE OPTIONS
echo ========================================
echo 1. Run Face Recognition Test (Lenient)
echo 2. Run Face Recognition Test (Balanced)
echo 3. Run Face Recognition Test (Strict)
echo 4. Run Simple Recognition Test
echo 5. Run Panel Visibility Test
echo 6. Run Recognition Panel Test
echo 7. Check OpenCV Compatibility
echo 8. Check System Status
echo 9. Exit
echo ========================================
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto lenient
if "%choice%"=="2" goto balanced
if "%choice%"=="3" goto strict
if "%choice%"=="4" goto simple
if "%choice%"=="5" goto panels
if "%choice%"=="6" goto recognition_test
if "%choice%"=="7" goto compatibility
if "%choice%"=="8" goto status
if "%choice%"=="9" goto exit
echo Invalid choice! Please try again.
echo.
goto menu

:lenient
echo.
echo Starting Face Recognition Test (Lenient Mode)...
echo Press Q or ESC to quit, D to toggle debug, S to toggle stats
echo.
python full-test.py --model .\models\combined_model.yml --level lenient
pause
goto menu

:balanced
echo.
echo Starting Face Recognition Test (Balanced Mode)...
echo Press Q or ESC to quit, D to toggle debug, S to toggle stats
echo.
python full-test.py --model .\models\combined_model.yml --level balanced
pause
goto menu

:strict
echo.
echo Starting Face Recognition Test (Strict Mode)...
echo Press Q or ESC to quit, D to toggle debug, S to toggle stats
echo.
python full-test.py --model .\models\combined_model.yml --level strict
pause
goto menu

:simple
echo.
echo Starting Simple Recognition Test...
echo Press Q to quit, S to take screenshot
echo.
python test_system.py
pause
goto menu

:panels
echo.
echo Starting Panel Visibility Test...
echo This will show colored rectangles to test panel visibility
echo.
python debug_panels.py
pause
goto menu

:recognition_test
echo.
echo Starting Recognition Panel Test...
echo This will test if the recognition panel is working correctly
echo Look for a GREEN panel in the top-left corner
echo.
python test_recognition_panel.py
pause
goto menu

:compatibility
echo.
echo Starting OpenCV Compatibility Check...
echo This will verify all modules work with OpenCV 4.10.0
echo.
python check_opencv_compatibility.py
pause
goto menu

:status
echo.
echo ========================================
echo           SYSTEM STATUS CHECK
echo ========================================
echo Checking Python version...
python --version
echo.
echo Checking OpenCV installation...
python -c "import cv2; print('OpenCV version:', cv2.__version__); print('Face module available:', hasattr(cv2, 'face'))"
echo.
echo Checking model files...
if exist "models\combined_model.yml" (
    echo ✓ combined_model.yml found
) else (
    echo ✗ combined_model.yml not found
)
if exist "models\final_omar_model.yml" (
    echo ✓ final_omar_model.yml found
) else (
    echo ✗ final_omar_model.yml not found
)
echo.
echo Checking camera access...
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"
echo.
pause
goto menu

:exit
echo.
echo Thank you for using Face Recognition System!
echo Goodbye!
pause
exit
