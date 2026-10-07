"""Okienko z bieżącym zużyciem zasobów przez aplikację (odświeżane co sekundę)."""

import os
import subprocess
import threading
import time

import psutil

REFRESH_MS = 1000
NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # żeby na Windowsie nie mrugało okno cmd

# To samo źródło danych, z którego korzysta Menedżer zadań (kolumny "Dedykowana/Udostępniona
# pamięć GPU" w zakładce Szczegóły). Klasa WMI ma angielskie nazwy także na polskim Windowsie.
GPU_MEMORY_QUERY = (
    "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
    "| Where-Object { $_.Name -like 'pid_%d_*' } "
    "| ForEach-Object { '{0},{1}' -f $_.DedicatedUsage, $_.SharedUsage }"
)
SPILL_WARNING_GB = 0.5  # trochę współdzielonej pamięci jest zawsze, ostrzegamy dopiero powyżej


def _nvidia_smi(*query):
    try:
        out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3, creationflags=NO_CONSOLE)
        return out.stdout.strip().splitlines() if out.returncode == 0 else []
    except Exception:
        return []


def read_gpu(pid):
    """Zwraca (obciążenie %, VRAM użyty MB, VRAM razem MB, VRAM aplikacji MB lub None).
    Pierwsza karta w systemie. Na Windowsie nvidia-smi zwykle nie podaje VRAM-u per proces."""
    try:
        util, used, total = (float(x) for x in _nvidia_smi("--query-gpu=utilization.gpu,memory.used,memory.total")[0].split(","))
    except Exception:
        return None

    app_mb = None
    for line in _nvidia_smi("--query-compute-apps=pid,used_memory"):
        try:
            line_pid, memory = (x.strip() for x in line.split(","))
            if line_pid == str(pid):
                app_mb = float(memory)
        except ValueError:
            pass  # np. "[N/A]"
    return util, used, total, app_mb


def read_process_gpu_memory(pid):
    """(dedykowana, współdzielona) pamięć GPU tego procesu w bajtach - tylko Windows.
    Dedykowana = prawdziwy VRAM, współdzielona = pamięć RAM używana przez kartę
    (tam trafia nadmiar, gdy VRAM się skończy). None, gdy nie da się odczytać."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", GPU_MEMORY_QUERY % pid],
                             capture_output=True, text=True, timeout=10, creationflags=NO_CONSOLE)
    except Exception:
        return None

    dedicated = shared = 0
    found = False
    for line in out.stdout.splitlines():
        try:
            d, s = (int(x) for x in line.split(","))
        except ValueError:
            continue
        dedicated += d
        shared += s
        found = True
    return (dedicated, shared) if found else None


class ResourceMonitor:
    def __init__(self, has_gpu):
        self.has_gpu = has_gpu
        self.process = psutil.Process()
        self.process.cpu_percent(None)  # pierwszy odczyt zawsze zwraca 0, więc "rozgrzewamy"
        self.cores = psutil.cpu_count() or 1
        self.total_ram = psutil.virtual_memory().total
        self.gpu_memory = None  # (dedykowana, współdzielona) w bajtach, aktualizowane w tle

        try:
            self._build_window()
        except Exception:
            self.root = None  # brak tkintera/ekranu

    def _build_window(self):
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Zasoby aplikacji")
        root.resizable(False, False)
        root.configure(padx=16, pady=12)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        rows = [("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"),
                ("vram", "VRAM"), ("spill", "VRAM → RAM")]
        self.rows = {}
        for i, (key, title) in enumerate(rows):
            name = tk.Label(root, text=title, font=("Segoe UI", 10, "bold"), anchor="w", width=11)
            name.grid(row=i, column=0, sticky="w", pady=6)
            bar = ttk.Progressbar(root, length=220, maximum=100)
            bar.grid(row=i, column=1, padx=10)
            value = tk.Label(root, text="...", font=("Segoe UI", 10), anchor="w", width=34)
            value.grid(row=i, column=2, sticky="w")
            self.rows[key] = (name, bar, value)

        tk.Label(root, text="Dotyczy tylko tej aplikacji. Zamknięcie okna zamyka program.",
                 font=("Segoe UI", 8), fg="gray").grid(row=len(rows), column=0, columnspan=3, pady=(10, 0))
        self.root = root

    def _show(self, key, percent, text):
        _, bar, value = self.rows[key]
        bar["value"] = max(0, min(100, percent))
        value.config(text=text)

    def _sample_gpu_memory(self):
        # PowerShell chwilę trwa, więc odpytujemy go w tle, żeby okno się nie zacinało
        while True:
            self.gpu_memory = read_process_gpu_memory(self.process.pid)
            time.sleep(1)

    def _tick(self):
        cpu = self.process.cpu_percent(None) / self.cores  # 100% = wszystkie rdzenie
        self._show("cpu", cpu, f"{cpu:.0f}%  ({self.cores} rdzeni)")

        rss = self.process.memory_info().rss
        self._show("ram", rss / self.total_ram * 100,
                   f"{rss / 1024**3:.1f} GB z {self.total_ram / 1024**3:.0f} GB")

        self._tick_gpu()
        self.root.after(REFRESH_MS, self._tick)

    def _tick_gpu(self):
        gpu = read_gpu(self.process.pid) if self.has_gpu else None
        if gpu is None:
            text = "brak karty NVIDIA" if not self.has_gpu else "brak danych z nvidia-smi"
            for key in ("gpu", "vram", "spill"):
                self._show(key, 0, text)
            return

        util, used, total, app_mb = gpu
        total_gb = total / 1024
        self._show("gpu", util, f"{util:.0f}%")

        memory = self.gpu_memory
        if memory:
            app_mb = memory[0] / 1024**2  # dokładniejsze niż nvidia-smi na Windowsie

        if app_mb is not None:
            self._show("vram", app_mb / total * 100, f"{app_mb / 1024:.1f} GB z {total_gb:.0f} GB")
        else:  # brak danych per proces - pokazujemy całą kartę
            self._show("vram", used / total * 100, f"{used / 1024:.1f} GB z {total_gb:.0f} GB (cała karta)")

        if memory:
            spill_gb = memory[1] / 1024**3
            warning = "  ⚠ przelew do RAM-u" if spill_gb > SPILL_WARNING_GB else ""
            self._show("spill", memory[1] / (total * 1024**2) * 100, f"{spill_gb:.1f} GB{warning}")
        else:
            self._show("spill", 0, "tylko Windows" if os.name != "nt" else "zbieram dane...")

    def run(self):
        """Blokuje do zamknięcia okna. Zwraca False, jeśli okna nie dało się utworzyć."""
        if not self.root:
            return False
        if self.has_gpu and os.name == "nt":
            threading.Thread(target=self._sample_gpu_memory, daemon=True).start()
        self._tick()
        self.root.mainloop()
        return True
