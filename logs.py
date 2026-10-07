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

from config import LOGS_DIR

log = logging.getLogger("tlumacz")

conversation_file = ""  # logs/<data_godzina>.csv, ta sama nazwa co plik .log tego uruchomienia
csv_lock = threading.Lock()  # Gradio obsługuje żądania w wątkach

CSV_COLUMNS = ["data", "nr", "mowi", "jezyk_zrodlowy", "jezyk_docelowy",
               "tekst_oryginalny", "tlumaczenie",
               "model_asr", "kwantyzacja_asr", "czas_asr_s",
               "model_tlumaczenia", "kwantyzacja_tlumaczenia", "czas_tlumaczenia_s", "czas_razem_s"]

LIBRARIES = ["torch", "torchaudio", "transformers", "ctranslate2",
             "faster-whisper", "librosa", "gradio"]


def init():
    """Wywołać raz, na samym początku programu."""
    global conversation_file
    os.makedirs(LOGS_DIR, exist_ok=True)
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    conversation_file = os.path.join(LOGS_DIR, name + ".csv")

    handler = logging.FileHandler(os.path.join(LOGS_DIR, name + ".log"), encoding="utf-8")
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
    """Dopisuje jedną wypowiedź (tylko tekst, bez audio) do pliku CSV tego uruchomienia.
    Plik powstaje przy pierwszej wypowiedzi, więc puste sesje nie zostawiają pustych CSV-ek."""
    row = [
        datetime.now().isoformat(sep=" ", timespec="seconds"), number,
        speaker.upper(), src.whisper, tgt.whisper,
        original, translation,
        asr.name, asr.precision, round(asr_seconds, 2),
        translator.name, translator.precision, round(translator_seconds, 2),
        round(asr_seconds + translator_seconds, 2),
    ]
    try:
        with csv_lock:
            is_new = not os.path.exists(conversation_file)
            # utf-8-sig, żeby Excel poprawnie pokazał polskie i tureckie znaki
            with open(conversation_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(CSV_COLUMNS)
                writer.writerow(row)
    except Exception:
        log.exception("Nie udało się zapisać wypowiedzi do %s", conversation_file)


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
