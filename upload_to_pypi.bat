@echo off
REM upload_to_pypi.bat - Upload ins_pricing package to PyPI with auto version detection
REM Usage: upload_to_pypi.bat
REM
REM Prerequisites:
REM   - Set TWINE_PASSWORD environment variable with your PyPI token
REM   - Run `python -m build` to build the distribution files

setlocal enabledelayedexpansion

REM Extract version from pyproject.toml
set VERSION=
for /f "tokens=2 delims==" %%i in ('findstr /r "^version *= *" pyproject.toml') do (
    set VERSION=%%i
)

REM Remove quotes and spaces
set VERSION=%VERSION:"=%
set VERSION=%VERSION: =%

if "%VERSION%"=="" (
    echo ERROR: Could not extract version from pyproject.toml
    exit /b 1
)

set PACKAGE_NAME=ins_pricing
set WHEEL_FILE=dist\%PACKAGE_NAME%-%VERSION%-py3-none-any.whl
set TARBALL_FILE=dist\%PACKAGE_NAME%-%VERSION%.tar.gz

echo ========================================
echo Uploading %PACKAGE_NAME% %VERSION% to PyPI
echo ========================================
echo.

REM Check if TWINE_PASSWORD is set
if "%TWINE_PASSWORD%"=="" (
    echo ERROR: TWINE_PASSWORD environment variable is not set
    echo Please set it with your PyPI token:
    echo   set TWINE_PASSWORD=your_token_here
    echo.
    echo Or set it inline:
    echo   set TWINE_PASSWORD=your_token ^&^& upload_to_pypi.bat
    exit /b 1
)

REM Check if wheel file exists
if not exist "%WHEEL_FILE%" (
    echo ERROR: Wheel file not found: %WHEEL_FILE%
    echo Please run: python -m build
    exit /b 1
)

REM Check if tarball file exists
if not exist "%TARBALL_FILE%" (
    echo ERROR: Source tarball not found: %TARBALL_FILE%
    echo Please run: python -m build
    exit /b 1
)

echo Files to upload:
echo   - %WHEEL_FILE%
echo   - %TARBALL_FILE%
echo.

echo Uploading to PyPI...
echo.

python -m twine upload -r pypi -u __token__ -p "%TWINE_PASSWORD%" "%WHEEL_FILE%" "%TARBALL_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Package uploaded to PyPI
    echo ========================================
    echo.
    echo You can now install it with:
    echo   pip install --upgrade %PACKAGE_NAME%
    echo.
    echo View on PyPI:
    echo   https://pypi.org/project/%PACKAGE_NAME%/%VERSION%/
) else (
    echo.
    echo ========================================
    echo FAILED! Upload failed with error code %ERRORLEVEL%
    echo ========================================
    echo.
    echo Common issues:
    echo   - Version %VERSION% already exists on PyPI
    echo   - Invalid TWINE_PASSWORD token
    echo   - Network connection issues
    exit /b %ERRORLEVEL%
)

endlocal
