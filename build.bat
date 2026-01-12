@echo off
REM Charly Neuromorphic Simulation - Build Script
REM Executes Claude Code to regenerate source files from prompts
REM Usage: build.bat

setlocal

echo ============================================
echo Charly Build - Regenerating sources
echo ============================================
echo.
echo This will invoke Claude Code to rebuild all
echo source files in src/ from the prompt files.
echo Config files in config/ will be preserved.
echo.

cd /d "%~dp0"

claude --print "Generate all source files in src/ directory based on the prompt files (physics.md, neuron.md, substrate.md, application.md). Clean src/ directory first but keep config/ files unchanged. Follow the specifications exactly."

echo.
echo Build complete.
pause
