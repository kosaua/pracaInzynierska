import os
import queue
import sys
import threading
import traceback
from config import VERBOSE


class SetupError(Exception):
    """Błąd, który ma się pokazać użytkownikowi w okienku."""

    def __init__(self, title, message):
        super().__init__(message)
        self.title = title
        self.message = message


class Checklist:
    def __init__(self, steps):
        """steps: lista (klucz, tytuł, krótki opis)"""
        self.steps = steps
        self.titles = {key: title for key, title, _ in steps}
        self.events = queue.Queue()
        self.root = None
        try:
            self._build_window()
        except Exception:
            self.root = None  # brak tkintera/ekranu -> tylko konsola

    def _build_window(self):
        import tkinter as tk

        root = tk.Tk()
        root.title("Tłumacz PL-TR — przygotowanie")
        root.resizable(False, False)
        root.configure(padx=8, pady=12)
        root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0))

        self.rows = {}
        for i, (key, title, desc) in enumerate(self.steps):
            box = tk.Label(root, text="☐", font=("Segoe UI", 16))
            box.grid(row=i, column=0, padx=(16, 8), pady=5, sticky="n")

            text = tk.Frame(root)
            text.grid(row=i, column=1, padx=(0, 24), sticky="w")
            tk.Label(text, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
            detail = tk.Label(text, text=desc, font=("Segoe UI", 9), fg="gray",
                              wraplength=360, justify="left")
            detail.pack(anchor="w")

            self.rows[key] = (box, detail)
        self.root = root

    # --- to wywołujemy z kodu startowego (z dowolnego wątku) ---

    def note(self, key, text):
        """Etap trwa - pokazujemy, co się teraz dzieje."""
        self._send("note", key, text)
        if VERBOSE:
            print(f"[..] {self.titles[key]}: {text}")

    def done(self, key, text=None):
        """Etap skończony - ptaszek i (opcjonalnie) nowy opis."""
        self._send("done", key, text)
        if not self.root or VERBOSE:
            print(f"[ok] {self.titles[key]}" + (f" - {text}" if text else ""))

    def close(self):
        self._send("close")

    def error(self, title, message):
        self._send("error", title, message)

    def _send(self, *event):
        if self.root:
            self.events.put(event)

    # --- obsługa okna (wątek główny) ---

    def _poll(self):
        try:
            while True:
                kind, *args = self.events.get_nowait()
                if kind == "note":
                    self.rows[args[0]][1].config(text=args[1], fg="#1a6fd6")
                elif kind == "done":
                    box, detail = self.rows[args[0]]
                    box.config(text="☑", fg="#16a34a")
                    detail.config(fg="gray", **({"text": args[1]} if args[1] else {}))
                elif kind == "error":
                    from tkinter import messagebox
                    messagebox.showerror(args[0], args[1])
                    self.root.destroy()
                    return
                elif kind == "close":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self, worker):
        """Odpala worker w tle, a okno obsługuje w głównym wątku (dzięki temu nie zawiesza się)."""

        def guarded():
            try:
                worker()
            except SetupError as exc:
                print(f"{exc.title}: {exc.message}", file=sys.stderr)
                self.error(exc.title, exc.message)
            except Exception as exc:
                traceback.print_exc()
                self.error("Błąd uruchomienia aplikacji", str(exc))
            finally:
                self.close()

        if not self.root:
            guarded()
            return

        thread = threading.Thread(target=guarded)
        thread.start()
        self.root.after(100, self._poll)
        self.root.mainloop()
        thread.join()
