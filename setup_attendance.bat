@echo off
REM Attendance Management System - Setup Script for Windows
REM This script installs dependencies and sets up the web interface

echo ================================================================
echo ATTENDANCE MANAGEMENT SYSTEM - SETUP
echo ================================================================
echo.

echo [1/3] Installing Flask...
python -m pip install flask --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Flask
    pause
    exit /b 1
)
echo     ✅ Flask installed successfully
echo.

echo [2/3] Creating templates directory...
if not exist "templates" mkdir templates
echo     ✅ Templates directory created
echo.

echo [3/3] Initializing database...
python -c "from examples.attendance_database import AttendanceDatabase; db = AttendanceDatabase('attendance.db'); print('✅ Database initialized')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to initialize database
    pause
    exit /b 1
)
echo.

echo ================================================================
echo ✅ SETUP COMPLETE!
echo ================================================================
echo.
echo Next steps:
echo 1. Run: python create_templates.py  (to create HTML templates)
echo 2. Run: python attendance_manager.py (to start web server)
echo 3. Open browser to: http://127.0.0.1:5000
echo.
echo ================================================================
pause
