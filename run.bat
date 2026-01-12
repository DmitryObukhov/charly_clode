@echo off
REM Charly Neuromorphic Simulation - Run Script
REM Usage: run.bat [days] [output_dir]

setlocal

set DAYS=%1
set OUTPUT=%2

if "%DAYS%"=="" set DAYS=1
if "%OUTPUT%"=="" set OUTPUT=output

echo Running Charly simulation for %DAYS% day(s)...
echo Output directory: %OUTPUT%
echo.

cd /d "%~dp0src"
python main.py --config ../config/config.yaml --days %DAYS% --output %OUTPUT%

echo.
echo Done.
REM pause
