"""Logi programu. Każde uruchomienie dostaje własny plik w folderze logs/."""

import csv
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from importlib import metadata

from config import CONVERSATIONS_FILE, LOGS_DIR

log = logging.getLogger("tlumacz")

session = ""  # nazwa pliku logu bieżącego uruchomienia, łączy wiersze CSV z logiem
csv_lock = threading.Lock()  # Gradio obsługuje żądania w wątkach

CSV_COLUMNS = ["data", "sesja", "nr", "mowi", "jezyk_zrodlowy", "jezyk_docelowy",
               "tekst_oryginalny", "tlumaczenie",
               "model_asr", "precyzja_asr", "czas_asr_s",
               "model_tlumaczenia", "precyzja_tlumaczenia", "czas_tlumaczenia_s", "czas_razem_s"]

LIBRARIES = ["torch", "torchaudio", "transformers", "ctranslate2",
             "faster-whisper", "librosa", "gradio"]


def init():
    """Wywołać raz, na samym początku programu."""
    global session
    os.makedirs(LOGS_DIR, exist_ok=True)
    session = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = session + ".log"

    handler = logging.FileHandler(os.path.join(LOGS_DIR, name), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False  # nie mieszamy z konsolą

    log.info("Uruchomienie: %s", " ".join([sys.executable, *sys.argv]))
    log.info("System: %s, Python %s", platform.platform(), platform.python_version())


def log_hardware(has_gpu, cuda):
    """Sprzęt i wersje bibliotek. Wołamy po instalacji pakietów, bo korzysta z psutil."""
    import psutil

    log.info("CPU: %s (rdzenie fizyczne: %s, logiczne: %s)",
             platform.processor() or platform.machine(),
             psutil.cpu_count(logical=False), psutil.cpu_count())
    log.info("RAM: %.1f GB", psutil.virtual_memory().total / 1024**3)

    if has_gpu:
        log.info("Wersja CUDA sterownika: %s", f"{cuda[0]}.{cuda[1]}" if cuda else "nieznana")
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            for line in out.stdout.strip().splitlines():
                name, driver, vram = (x.strip() for x in line.split(","))
                log.info("GPU: %s, sterownik %s, VRAM %s", name, driver, vram)
        except Exception as exc:
            log.warning("Nie udało się odczytać danych GPU z nvidia-smi: %s", exc)
    else:
        log.info("GPU: brak karty NVIDIA")

    versions = []
    for lib in LIBRARIES:
        try:
            versions.append(f"{lib} {metadata.version(lib)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{lib} (brak)")
    log.info("Biblioteki: %s", ", ".join(versions))


def save_turn(number, speaker, src, tgt, original, translation,
              asr, asr_seconds, translator, translator_seconds):
    """Dopisuje jedną wypowiedź (tylko tekst, bez audio) do logs/rozmowy.csv.
    Plik jest wspólny dla wszystkich uruchomień, kolumna 'sesja' mówi, z którego pochodzi wiersz."""
    row = [
        datetime.now().isoformat(sep=" ", timespec="seconds"), session, number,
        speaker.upper(), src.whisper, tgt.whisper,
        original, translation,
        asr.name, asr.precision, round(asr_seconds, 2),
        translator.name, translator.precision, round(translator_seconds, 2),
        round(asr_seconds + translator_seconds, 2),
    ]
    try:
        with csv_lock:
            is_new = not os.path.exists(CONVERSATIONS_FILE) or os.path.getsize(CONVERSATIONS_FILE) == 0
            # utf-8-sig, żeby Excel poprawnie pokazał polskie i tureckie znaki
            with open(CONVERSATIONS_FILE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(CSV_COLUMNS)
                writer.writerow(row)
    except Exception:
        log.exception("Nie udało się zapisać wypowiedzi do %s", CONVERSATIONS_FILE)


class timed:
    """Stoper: with timed("opis") as t: ...  - zapisuje czas w logu, a t.seconds zostaje do użycia.
    Bez opisu tylko mierzy, nic nie zapisuje."""

    def __init__(self, label=None):
        self.label = label
        self.seconds = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.perf_counter() - self.start
        if self.label:
            status = "BŁĄD po" if exc_type else "-"
            log.info("%s %s %.2f s", self.label, status, self.seconds)
        return False  # wyjątki lecą dalej
