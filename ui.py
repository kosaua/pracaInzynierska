import os
import traceback
from html import escape

import gradio as gr

import engines
from config import BASE_DIR, POLISH, TURKISH
import logs
from logs import log, timed

with open(os.path.join(BASE_DIR, "styles.css"), encoding="utf-8") as f:
    CSS = f.read()

READY = "Gotowy do rozmowy. / Sohbete hazır."

# kto mówi -> (język źródłowy, docelowy, model ASR, model tłumaczenia)
FLOW = {
    "pl": (POLISH, TURKISH, "asr_pl", "txt_pl_tr"),
    "tr": (TURKISH, POLISH, "asr_tr", "txt_tr_pl"),
}

FLAGS = {"pl": "🇵🇱", "tr": "🇹🇷"}


def render_conversation(history):
    if not history:
        return ("<div class='conv-empty'>Tu pojawi się wasza rozmowa 👇"
                "<br>Konuşmanız burada görünecek 👇</div>")

    html = ""
    # najnowsza wypowiedź ma być na samej górze, więc idziemy od końca
    for turn in reversed(history):
        side = turn["speaker"]
        translation = escape(turn["translation"] or "—")
        original = escape(turn["original"] or "—")
        html += (
            f"<div class='conv-row {side}'><div class='conv-bubble {side}'>"
            f"<div class='conv-primary'>{FLAGS[side]} {translation}</div>"
            f"<div class='conv-secondary'>{original}</div>"
            f"</div></div>"
        )
    return f"<div class='conv-container'>{html}</div>"


def handle_speech(speaker):
    src, tgt, asr_slot, text_slot = FLOW[speaker]

    def handler(audio_file, history):
        history = history or []

        if not audio_file:
            log.warning("Koniec nagrywania (%s), ale nie ma pliku z nagraniem", speaker.upper())
            error = "Nie podano nagrania. / Kayıt bulunamadı."
            return history, render_conversation(history), "", error, None

        asr = engines.engines[asr_slot]
        translator = engines.engines[text_slot]
        log.info("Nowe nagranie (%s -> %s)", src.whisper.upper(), tgt.whisper.upper())

        try:
            with timed() as asr_time:
                original = engines.transcribe(audio_file, asr, src)
            with timed() as translate_time:
                translation = engines.translate(original, translator, src, tgt)
        except Exception as exc:
            log.exception("Błąd przy przetwarzaniu nagrania %s", audio_file)
            traceback.print_exc()
            error = f"Coś poszło nie tak / Bir şeyler ters gitti: {exc}"
            return history, render_conversation(history), "", error, None

        history.append({"speaker": speaker, "original": original, "translation": translation})

        # cała wypowiedź w logu: modele, czasy i teksty
        log.info("Wypowiedź #%d | mówi: %s | ASR: %s (%s) %.2f s | tłumaczenie: %s (%s) %.2f s | razem %.2f s",
                 len(history), speaker.upper(),
                 asr.name, asr.precision, asr_time.seconds,
                 translator.name, translator.precision, translate_time.seconds,
                 asr_time.seconds + translate_time.seconds)
        log.info("    ASR (%s):          %s", src.whisper.upper(), original or "—")
        log.info("    Tłumaczenie (%s):  %s", tgt.whisper.upper(), translation or "—")

        logs.save_turn(len(history), speaker, src, tgt, original, translation,
                       asr, asr_time.seconds, translator, translate_time.seconds)

        caption = f"**{translation or '—'}**  \n_{original or '—'}_"

        # ostatni element (None) czyści nagranie w widżecie
        return history, render_conversation(history), caption, "", None

    return handler


def clear_conversation():
    log.info("Rozmowa wyczyszczona przyciskiem")
    return [], render_conversation([]), "", "", READY


def build_ui():
    with gr.Blocks(title="Tłumacz PL↔TR / PL↔TR Çevirmen") as demo:
        gr.Markdown("# Tłumacz rozmowy PL ↔ TR  /  PL ↔ TR Konuşma Çevirmeni", elem_id="app-title")
        gr.Markdown(
            "Każda osoba nagrywa po swojej stronie, "
            "tłumaczenie pojawia się automatycznie po zakończeniu nagrania.\n\n"
            "Her kişi kendi tarafında konuşmayı kaydeder, "
            "kayıt bittiğinde çeviri otomatik olarak görünür.",
            elem_id="app-subtitle",
        )

        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=["speaker-panel", "panel-pl"]):
                gr.Markdown("🇵🇱 **Mówi po polsku** *(Lehçe konuşuyor)*", elem_classes=["speaker-title"])
                audio_pl = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                caption_pl = gr.Markdown("", elem_classes=["live-caption"])

            with gr.Column(elem_classes=["speaker-panel", "panel-tr"]):
                gr.Markdown("🇹🇷 **Türkçe konuşuyor** *(Mówi po turecku)*", elem_classes=["speaker-title"])
                audio_tr = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                caption_tr = gr.Markdown("", elem_classes=["live-caption"])

        history = gr.State([])
        chat = gr.HTML(render_conversation([]))
        status = gr.Markdown(READY, elem_id="status-caption")

        with gr.Row(elem_id="clear-row"):
            clear_btn = gr.Button("🗑️ Wyczyść rozmowę / Sohbeti temizle", size="sm")

        audio_pl.stop_recording(handle_speech("pl"), [audio_pl, history],
                                [history, chat, caption_pl, status, audio_pl])
        audio_tr.stop_recording(handle_speech("tr"), [audio_tr, history],
                                [history, chat, caption_tr, status, audio_tr])
        clear_btn.click(clear_conversation, outputs=[history, chat, caption_pl, caption_tr, status])

    return demo
