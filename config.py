from collections import namedtuple
import os

Model = namedtuple("Model", "kind folder source")

MODELS = {
    "WSP-BASE":     Model("whisper", "whisper-base", "base"),
    "WSP-SMALL":    Model("whisper", "whisper-small", "small"),
    "WSP-MEDIUM":   Model("whisper", "whisper-medium", "medium"),
    "WSP-LARGE-V3": Model("whisper", "whisper-large-v3", "large-v3"),

    "NLLB-600M": Model("nllb", "nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M"),
    "NLLB-1.3B": Model("nllb", "nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-1.3B"),

    "SML-MEDIUM": Model("seamless", "hf-seamless-m4t-medium", "facebook/hf-seamless-m4t-medium"),
}


PROFILES = {
    "gpu": {
        "asr_pl":    ("SML-MEDIUM", "float32"),
        "asr_tr":    ("WSP-MEDIUM", "int8"),
        "txt_pl_tr": ("NLLB-600M",  "float32"),
        "txt_tr_pl": ("NLLB-600M",  "float32"),
    },
    "cpu": {
        "asr_pl":    ("WSP-BASE",   "int8"),
        "asr_tr":    ("WSP-BASE",   "int8"),
        "txt_pl_tr": ("NLLB-600M",  "int8"),
        "txt_tr_pl": ("SML-MEDIUM", "float32"),
    },
}

Language = namedtuple("Language", "whisper seamless nllb")
POLISH = Language("pl", "pol", "pol_Latn")
TURKISH = Language("tr", "tur", "tur_Latn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "bin")
MODELS_DIR = os.path.join(BASE_DIR, "modele_offline")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
