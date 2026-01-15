@echo off
REM Test script to verify version extraction from pyproject.toml
REM This can be deleted after verification

setlocal enabledelayedexpansion

echo Testing version extraction from pyproject.toml...
echo.

REM Extract version
set VERSION=
for /f "tokens=2 delims==" %%i in ('findstr /r "^version *= *" pyproject.toml') do (
    set VERSION=%%i
)

REM Remove quotes and spaces
set VERSION=%VERSION:"=%
set VERSION=%VERSION: =%

if "%VERSION%"=="" (
    echo [FAILED] Could not extract version
    exit /b 1
) else (
    echo [SUCCESS] Extracted version: %VERSION%
    echo.
    echo Expected files:
    echo   - dist\ins_pricing-%VERSION%-py3-none-any.whl
    echo   - dist\ins_pricing-%VERSION%.tar.gz
)

endlocal
