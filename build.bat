@echo off
REM DLER Build Script
REM Usage: build.bat [ultimate|basic|all]

setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: build.bat [ultimate^|basic^|all]
    echo.
    echo   ultimate  - Build DLER.exe with GPU/CUDA support (~1 GB)
    echo   basic     - Build DLER_Basic.exe CPU-only (~30 MB)
    echo   all       - Build both versions
    echo.
    exit /b 1
)

if /i "%1"=="ultimate" goto :build_ultimate
if /i "%1"=="basic" goto :build_basic
if /i "%1"=="all" goto :build_all

echo Unknown option: %1
exit /b 1

:build_ultimate
echo.
echo ============================================================
echo   Building DLER Ultimate (GPU/CUDA)
echo ============================================================
echo.
set DLER_BUILD=ultimate
python -m PyInstaller dler_tk.spec --noconfirm
if errorlevel 1 (
    echo BUILD FAILED!
    exit /b 1
)
echo.
echo Ultimate build complete: dist\DLER.exe
for %%A in (dist\DLER.exe) do echo Size: %%~zA bytes (%%~zA / 1048576 = ~!SIZE! MB^)
goto :eof

:build_basic
echo.
echo ============================================================
echo   Building DLER Basic (CPU-only)
echo ============================================================
echo.
set DLER_BUILD=basic
python -m PyInstaller dler_tk.spec --noconfirm
if errorlevel 1 (
    echo BUILD FAILED!
    exit /b 1
)
echo.
echo Basic build complete: dist\DLER_Basic.exe
goto :eof

:build_all
echo.
echo ============================================================
echo   Building BOTH versions
echo ============================================================
echo.

REM Build Basic first (faster)
call :build_basic
if errorlevel 1 exit /b 1

REM Then Ultimate
call :build_ultimate
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo   ALL BUILDS COMPLETE
echo ============================================================
echo.
echo   dist\DLER.exe        - Ultimate (GPU/CUDA)
echo   dist\DLER_Basic.exe  - Basic (CPU-only)
echo.
dir dist\DLER*.exe
goto :eof
