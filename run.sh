#!/bin/bash
cd "$(dirname "$0")"

VENV_DIR="venv"

# 1) Jeśli venv jest już aktywny w tej sesji
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Wykryto aktywne środowisko: $VIRTUAL_ENV"
    python main.py
    exit $?
fi

# 2) Jeśli folder venv już istnieje, uruchamiamy go
if [ -f "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" main.py
    exit $?
fi

# 3) Poszukiwanie Pythona 3.12 w systemie
PYTHON_CMD=""

if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$VER" = "3.12" ]; then
        PYTHON_CMD="python3"
    fi
elif command -v python &> /dev/null; then
    VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$VER" = "3.12" ]; then
        PYTHON_CMD="python"
    fi
fi

# 4) Sprawdzanie czy odnaleziono Pythona 3.12
if [ -z "$PYTHON_CMD" ]; then
    echo "BŁĄD: Wymagany Python w wersji 3.12 nie został znaleziony w systemie."
    echo "Zainstaluj Python 3.12 i spróbuj ponownie."
    exit 1
fi

echo "Nie znaleziono środowiska venv - tworzę nowe przy użyciu ($PYTHON_CMD)..."
$PYTHON_CMD -m venv "$VENV_DIR"

if [ $? -ne 0 ]; then
    echo "Nie udało się utworzyć środowiska venv. Upewnij się, że masz zainstalowany pakiet python3.12-venv."
    exit 1
fi

# 5) Uruchomienie programu
"$VENV_DIR/bin/python" main.py