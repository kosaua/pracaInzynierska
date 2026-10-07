"""
Tłumacz mowy PL <-> TR.   Uruchomienie:  python main.py

Przy pierwszym starcie sam dociąga biblioteki, FFmpeg i modele.
Modele i ich precyzję zmieniasz w config.py.
"""

import glob
import logging
import os
import sys
import time
import warnings

# wyciszamy konsolę, zanim cokolwiek się zaimportuje
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
warnings.filterwarnings("ignore")
for noisy in ("httpx", "faster_whisper", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


def drop_foreign_cudnn():
    """Inna wersja cuDNN w PATH (Windows) albo LD_LIBRARY_PATH (Linux, np. z ręcznie
    zainstalowanego CUDA Toolkit) potrafi się wmieszać i daje błąd "Could not load symbol
    cudnnGetLibConfig". Wyrzucamy takie foldery tylko dla tego procesu - torch ma własne biblioteki."""
    if os.name != "nt":
        # Linux czyta LD_LIBRARY_PATH tylko przy starcie procesu, więc po zmianie uruchamiamy się ponownie
        paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
        kept = [p for p in paths if not glob.glob(os.path.join(p, "libcudnn*"))]
        if len(kept) != len(paths) and not os.environ.get("_APKA_REEXEC"):
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
            os.environ["_APKA_REEXEC"] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])
        return
    kept = [p for p in os.environ.get("PATH", "").split(os.pathsep)
            if not (p and glob.glob(os.path.join(p, "cudnn*.dll")))]
    os.environ["PATH"] = os.pathsep.join(kept)


drop_foreign_cudnn()

# logi włączamy dopiero po ewentualnym restarcie, żeby nie powstały dwa pliki
import logs
logs.init()
from logs import log, timed

import setup_env
from config import PROFILES
from startup_window import Checklist

STEPS = [
    ("hw",     "Sprzęt",       "Sprawdzam, czy jest karta NVIDIA"),
    ("libs",   "Biblioteki",   "PyTorch, transformers, gradio, faster-whisper..."),
    ("ffmpeg", "FFmpeg",       "Obsługa nagrań audio"),
    ("models", "Modele",       "Sprawdzam, czy modele są na dysku"),
    ("load",   "Wczytywanie",  "Ładowanie modeli do pamięci"),
    ("ui",     "Interfejs",    "Uruchamianie aplikacji w przeglądarce"),
]
checklist = Checklist(STEPS)
app = {}  # wypełniane po udanym starcie


def start():
    note = checklist.note

    start_time = time.perf_counter()

    with timed("Etap 1/6 sprzęt"):
        has_gpu, cuda = setup_env.detect_gpu()
    profile = "gpu" if has_gpu else "cpu"
    log.info("Wybrany profil: %s", profile.upper())
    checklist.done("hw", f"Karta NVIDIA (CUDA {cuda[0]}.{cuda[1]})" if cuda
                   else "Karta NVIDIA" if has_gpu else "Brak karty NVIDIA - tryb CPU")

    with timed("Etap 2/6 biblioteki"):
        setup_env.ensure_packages(has_gpu, cuda, lambda text: note("libs", text))
    logs.log_hardware(has_gpu, cuda)
    checklist.done("libs", "Wszystko zainstalowane")

    with timed("Etap 3/6 FFmpeg"):
        setup_env.ensure_ffmpeg(lambda text: note("ffmpeg", text))
    checklist.done("ffmpeg", "Gotowy")

    model_names = sorted({name for name, _ in PROFILES[profile].values()})
    with timed("Etap 4/6 sprawdzenie plików modeli"):
        setup_env.ensure_models(model_names, lambda text: note("models", text))
    checklist.done("models", ", ".join(model_names))

    import engines  # dopiero teraz, bo ciągnie torcha
    with timed("Etap 5/6 wczytanie modeli"):
        engines.load_all(profile, lambda text: note("load", text))
    checklist.done("load", f"Profil {profile.upper()}")

    import ui
    with timed("Etap 6/6 interfejs"):
        demo = ui.build_ui()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            quiet=True,
            prevent_thread_lock=True,  # serwer działa w tle, a główny wątek obsłuży okno zasobów
            css=ui.CSS,
            theme=ui.gr.themes.Soft(),
        )
    app["demo"] = demo
    app["has_gpu"] = has_gpu

    log.info("Cały start programu - %.2f s", time.perf_counter() - start_time)
    checklist.done("ui", "Gotowe")
    checklist.close()
    log.info("Aplikacja działa: http://127.0.0.1:7860")
    print("Aplikacja działa: http://127.0.0.1:7860  (zamknij okno zasobów albo Ctrl+C, żeby zakończyć)")


if __name__ == "__main__":
    checklist.run(start)

    if "demo" in app:  # start się udał
        from monitor import ResourceMonitor
        if ResourceMonitor(app["has_gpu"]).run():
            log.info("Okno zasobów zamknięte - koniec programu")
            os._exit(0)  # okno zasobów zamknięte = koniec programu
        app["demo"].block_thread()  # bez tkintera: po prostu czekamy w konsoli
