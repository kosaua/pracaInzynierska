"""Przygotowanie środowiska: sprzęt, biblioteki, FFmpeg, pliki modeli.
Wszystko działa po cichu - pip i pobieranie nic nie wypisują do konsoli."""

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

from config import BIN_DIR, MODELS, MODELS_DIR
from logs import log, timed
from startup_window import SetupError

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --- sprzęt ------------------------------------------------------------------

def detect_gpu():
    """Zwraca (jest_karta_nvidia, wersja_cuda_sterownika jako krotka albo None)."""
    if not shutil.which("nvidia-smi"):
        log.info("Nie znaleziono nvidia-smi - zakładam brak karty NVIDIA")
        return False, None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    except Exception as exc:
        log.warning("nvidia-smi nie uruchomiło się: %s", exc)
        return False, None
    if out.returncode != 0:
        log.warning("nvidia-smi zwróciło kod %s", out.returncode)
        return False, None
    found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    return True, (int(found[1]), int(found[2])) if found else None


# --- biblioteki --------------------------------------------------------------

PACKAGES = {  # nazwa w pip -> nazwa przy imporcie
    "transformers": "transformers",
    "faster-whisper": "faster_whisper",
    "huggingface_hub": "huggingface_hub",
    "librosa": "librosa",
    "gradio": "gradio",
    "sentencepiece": "sentencepiece",
    "accelerate": "accelerate",
    "psutil": "psutil",  # używa go monitor.py i logi
}
CUDA_BUILDS = [((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]


def _installed(module):
    # find_spec nie importuje modułu, więc nie ładujemy torcha, którego zaraz możemy wymieniać
    return importlib.util.find_spec(module) is not None


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    log.info("Polecenie: %s", " ".join(cmd))
    with timed("   pip zakończony"):
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("pip zwrócił kod %s:\n%s", result.returncode, result.stderr[-1500:])
        raise SetupError("Błąd instalacji", result.stderr[-1500:])


def _torch_sees_gpu():
    # osobny proces, żeby nie ładować torcha do tego
    out = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                         capture_output=True, text=True)
    sees = out.stdout.strip() == "True"
    log.info("Czy zainstalowany PyTorch widzi GPU: %s", sees)
    return sees


def _torch_index_url(has_gpu, cuda):
    build = "cpu"
    if has_gpu:
        build = "cu121"  # wartość domyślna, gdy nie znamy wersji sterownika
        for version, name in CUDA_BUILDS:
            if cuda and cuda >= version:
                build = name
                break
    return f"https://download.pytorch.org/whl/{build}"


def ensure_packages(has_gpu, cuda, note):
    torch_ok = _installed("torch") and _installed("torchaudio")
    log.info("PyTorch zainstalowany: %s", torch_ok)

    if torch_ok and has_gpu and not _torch_sees_gpu():
        note("Zainstalowany PyTorch nie widzi karty - wymieniam na wersję CUDA")
        _pip("uninstall", "-y", "torch", "torchaudio")
        torch_ok = False

    if not torch_ok:
        note("Pobieram PyTorch - pierwsze uruchomienie może potrwać kilkanaście minut")
        _pip("install", "-q", "torch", "torchaudio", "--index-url", _torch_index_url(has_gpu, cuda))
        if has_gpu and not _torch_sees_gpu():
            raise SetupError(
                "PyTorch nie widzi GPU",
                "Po instalacji torch.cuda.is_available() nadal zwraca False.\n"
                "Zwykle to za stary sterownik NVIDIA albo kilka środowisk Pythona.\n"
                "Zaktualizuj sterownik i uruchom program ponownie.",
            )

    missing = [pip_name for pip_name, module in PACKAGES.items() if not _installed(module)]
    if missing:
        note("Pobieram: " + ", ".join(missing))
        _pip("install", "-q", *missing)
    else:
        log.info("Wszystkie pozostałe pakiety już są")

    importlib.invalidate_caches()


# --- FFmpeg ------------------------------------------------------------------

FFMPEG_WINDOWS = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_LINUX = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def ensure_ffmpeg(note):
    if shutil.which("ffmpeg"):
        log.info("FFmpeg znaleziony w systemie: %s", shutil.which("ffmpeg"))
        return

    exe = os.path.join(BIN_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(exe):
        note("Pobieram FFmpeg")
        log.info("Pobieram FFmpeg do %s", exe)
        archive = os.path.join(BIN_DIR, "_ffmpeg_tmp")
        try:
            with timed("   FFmpeg pobrany"):
                if os.name == "nt":
                    urllib.request.urlretrieve(FFMPEG_WINDOWS, archive)
                    with zipfile.ZipFile(archive) as zf:
                        name = next(n for n in zf.namelist() if n.endswith("bin/ffmpeg.exe"))
                        with zf.open(name) as src, open(exe, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                elif sys.platform.startswith("linux"):
                    urllib.request.urlretrieve(FFMPEG_LINUX, archive)
                    with tarfile.open(archive, "r:xz") as tf:
                        member = next(m for m in tf.getmembers() if m.name.endswith("/ffmpeg"))
                        with tf.extractfile(member) as src, open(exe, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    os.chmod(exe, 0o755)
                else:
                    raise SetupError("Brak FFmpeg", "Zainstaluj ffmpeg ręcznie (np. brew install ffmpeg).")
        except SetupError:
            raise
        except Exception as exc:
            log.exception("Błąd pobierania FFmpeg")
            raise SetupError("Błąd pobierania FFmpeg", str(exc))
        finally:
            if os.path.exists(archive):
                os.remove(archive)
    else:
        log.info("FFmpeg już jest w folderze bin: %s", exe)

    os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")


# --- modele ------------------------------------------------------------------

WEIGHT_FILES = {
    "model.bin",                                            # faster-whisper
    "model.safetensors", "model.safetensors.index.json",    # transformers
    "pytorch_model.bin", "pytorch_model.bin.index.json",
}


def _on_disk(folder):
    # folder bez pliku z wagami = przerwane pobieranie, więc samo "istnieje" nie wystarcza
    path = os.path.join(MODELS_DIR, folder)
    return os.path.isdir(path) and bool(WEIGHT_FILES & set(os.listdir(path)))


def ensure_models(names, note):
    for name in names:
        model = MODELS[name]
        if _on_disk(model.folder):
            log.info("Model %s jest na dysku (%s)", name, model.folder)
            continue

        note(f"Pobieram {name}")
        log.info("Model %s: brak na dysku, pobieram ze źródła %s", name, model.source)
        path = os.path.join(MODELS_DIR, model.folder)
        shutil.rmtree(path, ignore_errors=True)

        with timed(f"   Pobrano {name}"):
            if model.kind == "whisper":
                from faster_whisper import download_model
                download_model(model.source, output_dir=path)
            else:
                from huggingface_hub import list_repo_files, snapshot_download
                skip = ["*.h5", "*.msgpack"]
                if any(f.endswith(".safetensors") for f in list_repo_files(model.source)):
                    skip.append("*.bin")  # nie ciągniemy tego samego dwa razy
                snapshot_download(model.source, local_dir=path, ignore_patterns=skip)

        if not _on_disk(model.folder):
            log.error("Po pobraniu %s w folderze %s nadal nie ma wag", name, model.folder)
            raise SetupError("Pobieranie modelu nie powiodło się",
                             f"W folderze '{model.folder}' nie ma pliku z wagami.\n"
                             "Sprawdź internet i miejsce na dysku, potem uruchom program ponownie.")
