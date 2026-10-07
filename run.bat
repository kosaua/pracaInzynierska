@echo off
cd /d "%~dp0"

set "VENV_DIR=venv"

rem 1) Jesli venv jest juz aktywny w tym oknie, uzywamy go
if defined VIRTUAL_ENV (
    echo Wykryto aktywne srodowisko: %VIRTUAL_ENV%
    python main.py
    goto koniec
)

rem 2) Jesli folder venv juz istnieje, uruchamiamy go
if exist "%VENV_DIR%\Scripts\python.exe" goto start

rem 3) Sprawdzamy obecnosc Pythona 3.12
echo Nie znaleziono srodowiska venv - szukanie Pythona 3.12...

set "PY_CMD="

rem Sprawdzenie przez Python Launcher (py -3.12)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.12"
    goto create_venv
)

rem Sprawdzenie domyslnego polecenia python
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
    if "%%i"=="3.12" (
        set "PY_CMD=python"
        goto create_venv
    )
)

:error_no_python
echo.
echo BLED: Nie znaleziono Pythona w wersji 3.12 w systemie!
echo Zainstaluj Python 3.12 (python.org) i upewnij sie, ze jest dodany do PATH / Py Launcher.
pause
exit /b 1

:create_venv
echo Tworzenie nowego srodowiska venv przy uzyciu %PY_CMD%...
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo Nie udalo sie utworzyc srodowiska venv.
    pause
    exit /b 1
)

:start
"%VENV_DIR%\Scripts\python.exe" main.py

:koniec
if errorlevel 1 (
    echo.
    echo Program zakonczyl sie bledem.
    pause
)