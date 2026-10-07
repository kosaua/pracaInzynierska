import os
from dataclasses import dataclass
from typing import Any
import torch  # musi być pierwszy: ładuje własne DLL-e CUDA/cuDNN, których potem używa ctranslate2
import gpu_spill
gpu_spill.enable()  # Linux: przelew VRAM -> RAM (na Windowsie robi to sterownik)

import ctranslate2
import numpy as np
import librosa
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, NllbTokenizerFast, SeamlessM4TModel

from config import MODELS, MODELS_DIR, POLISH, PROFILES, TURKISH
from logs import log, timed


@dataclass
class Engine:
    kind: str
    model: Any
    processor: Any = None
    name: str = ""       # np. "NLLB-600M", do logów
    precision: str = ""  # kwantyzacja, w jakiej model faktycznie działa

engines = {}

def _whisper_compute_type(precision, device):
    # wybór najlepszego działającego typu obliczeń dla CTranslate2 (Whisper)
    supported = ctranslate2.get_supported_compute_types(device)
    for candidate in (precision, "float16", "int8_float16", "int8", "float32"):
        if candidate in supported:
            return candidate
    return precision


def _torch_precision(precision, device):
    if device == "cpu":
        return "float32" if precision == "float16" else precision
    if precision == "int8":  # kwantyzacja int8 jest tylko dla CPU
        precision = "float16"
    # karty z compute capability < 7 (np. GTX 10xx) nie radzą sobie z fp16
    if precision == "float16" and torch.cuda.get_device_capability()[0] < 7:
        return "float32"
    return precision


# --- ładowanie ---------------------------------------------------------------

def _load(name, precision, device):
    info = MODELS[name]
    path = os.path.join(MODELS_DIR, info.folder)

    if info.kind == "whisper":
        compute_type = _whisper_compute_type(precision, device)
        log.info("Ładuję %s: urządzenie=%s, kwantyzacja z configu=%s, faktyczna=%s",
                 name, device, precision, compute_type)
        with timed(f"Wczytano {name}"):
            model = WhisperModel(path, device=device, local_files_only=True,
                                 compute_type=compute_type)
        return Engine("whisper", model, name=name, precision=compute_type)

    used = _torch_precision(precision, device)
    log.info("Ładuję %s: urządzenie=%s, kwantyzacja z configu=%s, faktyczna=%s",
             name, device, precision, used)
    dtype = torch.float16 if used == "float16" else torch.float32

    with timed(f"Wczytano {name}"):
        if info.kind == "nllb":
            processor = NllbTokenizerFast.from_pretrained(path, local_files_only=True)
            model_class = AutoModelForSeq2SeqLM
        else:
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model_class = SeamlessM4TModel

        model = model_class.from_pretrained(path, dtype=dtype, local_files_only=True).to(device)
        if used == "int8":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return Engine(info.kind, model, processor, name, used)


def load_all(profile_name, note):
    device = "cuda" if profile_name == "gpu" else "cpu"
    log.info("Profil %s, urządzenie %s", profile_name.upper(), device)
    log.info("PyTorch %s (CUDA %s), CTranslate2 %s",
             torch.__version__, torch.version.cuda, ctranslate2.__version__)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU w PyTorchu: %s, compute capability %d.%d, VRAM %.1f GB",
                 props.name, props.major, props.minor, props.total_memory / 1024**3)

    loaded = {}  # ten sam model w tej samej precyzji ładujemy tylko raz

    # Whisper (CTranslate2) nie korzysta z przelewu do RAM-u, więc ładujemy go pierwszego,
    # żeby dostał twardy VRAM, zanim modele z PyTorcha zajmą resztę
    slots = sorted(PROFILES[profile_name].items(), key=lambda item: MODELS[item[1][0]].kind != "whisper")

    for slot, (name, precision) in slots:
        if (name, precision) not in loaded:
            note(f"Wczytuję {name} ({precision})")
            loaded[(name, precision)] = _load(name, precision, device)
        else:
            log.info("Slot %s używa już wczytanego modelu %s (%s)", slot, name, precision)
        engines[slot] = loaded[(name, precision)]

    for slot, engine in sorted(engines.items()):
        log.info("Przypisanie: %-10s -> %s (%s)", slot, engine.name, engine.precision)


# --- rozpoznawanie i tłumaczenie ---------------------------------------------

def transcribe(audio_path, engine, lang):
    # audio wczytujemy sami (librosa), dzięki temu faster-whisper nie dekoduje pliku
    # przez bibliotekę av, która potrafi być niezgodna wersją
    audio, _ = librosa.load(audio_path, sr=16000)
    return _recognize(audio, engine, lang)


def _recognize(audio, engine, lang):
    if engine.kind == "whisper":
        segments, _ = engine.model.transcribe(audio, language=lang.whisper, beam_size=5)
        return " ".join(s.text for s in segments).strip()

    inputs = engine.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(engine.model.device)
    with torch.inference_mode():
        out = engine.model.generate(**inputs, tgt_lang=lang.seamless, generate_speech=False)
    return engine.processor.batch_decode(out[0], skip_special_tokens=True)[0].strip()


def translate(text, engine, src, tgt):
    if not text.strip():
        return ""

    if engine.kind == "nllb":
        engine.processor.src_lang = src.nllb
        inputs = engine.processor(text, return_tensors="pt").to(engine.model.device)
        tgt_id = engine.processor.convert_tokens_to_ids(tgt.nllb)
        with torch.inference_mode():
            tokens = engine.model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=256)
        return engine.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    inputs = engine.processor(text=text, src_lang=src.seamless, return_tensors="pt")
    inputs = inputs.to(engine.model.device)
    with torch.inference_mode():
        out = engine.model.generate(**inputs, tgt_lang=tgt.seamless, generate_speech=False)
    return engine.processor.batch_decode(out[0], skip_special_tokens=True)[0].strip()


# --- rozgrzewka --------------------------------------------------------------

WARMUP = [
    ("PL", POLISH, TURKISH, "asr_pl", "txt_pl_tr", "Dzień dobry, jak się masz?"),
    ("TR", TURKISH, POLISH, "asr_tr", "txt_tr_pl", "Merhaba, nasılsın?"),
]


def warmup():
    """Pierwszy przebieg modelu jest wolny (kernele CUDA, alokacje), więc robimy go
    na sucho przed startem interfejsu - dla obu kierunków, żeby pierwsza prawdziwa
    rozmowa nie czekała. ASR dostaje 2 s ciszy, tłumaczenie krótkie zdanie."""
    silence = np.zeros(2 * 16000, dtype=np.float32)

    for speaker, src, tgt, asr_slot, text_slot, sentence in WARMUP:
        try:
            with timed(f"Rozgrzewka {speaker}: ASR ({engines[asr_slot].name})"):
                _recognize(silence, engines[asr_slot], src)
            with timed(f"Rozgrzewka {speaker}: tłumaczenie ({engines[text_slot].name})"):
                translate(sentence, engines[text_slot], src, tgt)
        except Exception:
            # nieudana rozgrzewka nie powinna blokować aplikacji
            log.exception("Rozgrzewka %s nie powiodła się", speaker)
