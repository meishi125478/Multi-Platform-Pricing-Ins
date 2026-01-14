@echo off
REM Upload ins_pricing to PyPI
REM 使用方法:
REM 1. 设置环境变量 TWINE_PASSWORD 为你的 PyPI token
REM 2. 运行此脚本

echo ========================================
echo Uploading ins_pricing 0.2.8 to PyPI
echo ========================================
echo.

REM 检查环境变量
if "%TWINE_PASSWORD%"=="" (
    echo ERROR: TWINE_PASSWORD environment variable is not set
    echo Please set it with your PyPI token:
    echo set TWINE_PASSWORD=your_token_here
    exit /b 1
)

REM 检查 dist 目录
if not exist "dist\ins_pricing-0.2.8-py3-none-any.whl" (
    echo ERROR: Wheel file not found. Please run: python -m build
    exit /b 1
)

if not exist "dist\ins_pricing-0.2.8.tar.gz" (
    echo ERROR: Source tarball not found. Please run: python -m build
    exit /b 1
)

echo Uploading to PyPI...
echo.

python -m twine upload -r pypi -u __token__ -p "%TWINE_PASSWORD%" dist/ins_pricing-0.2.8-py3-none-any.whl dist/ins_pricing-0.2.8.tar.gz

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Package uploaded to PyPI
    echo ========================================
    echo.
    echo You can now install it with:
    echo pip install --upgrade ins_pricing
) else (
    echo.
    echo ========================================
    echo FAILED! Upload failed with error code %ERRORLEVEL%
    echo ========================================
    exit /b %ERRORLEVEL%
)
