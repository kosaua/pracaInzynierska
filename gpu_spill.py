"""Linux: przelew pamięci GPU do RAM-u (odpowiednik "Udostępnionej pamięci GPU" z Windowsa).

Alokator PyTorcha oparty o cudaMallocManaged (CUDA Unified Memory): pamięć może być większa
niż VRAM, a sterownik sam przenosi strony między VRAM a RAM-em.
Wymaga: gcc (apt install build-essential) i karty Pascal+ (GTX 10xx lub nowsza).
Wyłączenie: NO_GPU_SPILL=1 python main.py
"""

import os
import shutil
import subprocess
import sys

import torch

from config import BIN_DIR

C_SOURCE = r"""
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>
#include <dlfcn.h>

/* Runtime API CUDA (libcudart, ładowany już przez PyTorcha) - w przeciwieństwie do API
   sterownika sam podpina kontekst CUDA w każdym wątku (Gradio obsługuje żądania w wątkach roboczych). */
typedef int cudaError_t;
static cudaError_t (*cuda_malloc_managed)(void **, size_t, unsigned int);
static cudaError_t (*cuda_free)(void *);
static cudaError_t (*cuda_set_device)(int);

static int init(void) {
    static const char *names[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL};
    void *lib = RTLD_DEFAULT;
    cuda_malloc_managed = (cudaError_t (*)(void **, size_t, unsigned int)) dlsym(lib, "cudaMallocManaged");
    for (int i = 0; !cuda_malloc_managed && names[i]; i++) {
        lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (lib) cuda_malloc_managed = (cudaError_t (*)(void **, size_t, unsigned int)) dlsym(lib, "cudaMallocManaged");
    }
    if (!cuda_malloc_managed) return 0;
    cuda_free = (cudaError_t (*)(void *)) dlsym(lib, "cudaFree");
    cuda_set_device = (cudaError_t (*)(int)) dlsym(lib, "cudaSetDevice");
    return cuda_free && cuda_set_device;
}

void *managed_malloc(ssize_t size, int device, void *stream) {
    if (!cuda_malloc_managed && !init()) {
        fprintf(stderr, "[gpu_spill] nie znaleziono libcudart\n");
        return NULL;
    }
    void *ptr = NULL;
    cuda_set_device(device);
    cudaError_t err = cuda_malloc_managed(&ptr, (size_t)size, 1 /* cudaMemAttachGlobal */);
    if (err != 0) {
        fprintf(stderr, "[gpu_spill] cudaMallocManaged(%zd) -> błąd %d\n", (ssize_t)size, err);
        return NULL;
    }
    return ptr;
}

void managed_free(void *ptr, ssize_t size, int device, void *stream) {
    if (ptr && cuda_free) cuda_free(ptr);
}
"""


def _build():
    os.makedirs(BIN_DIR, exist_ok=True)
    so_path = os.path.join(BIN_DIR, "managed_alloc.so")
    c_path = os.path.join(BIN_DIR, "managed_alloc.c")

    if os.path.exists(so_path) and os.path.exists(c_path):
        with open(c_path) as f:
            if f.read() == C_SOURCE:
                return so_path  # już zbudowane

    if not shutil.which("gcc"):
        raise RuntimeError("brak kompilatora gcc - zainstaluj: sudo apt install build-essential")
    with open(c_path, "w") as f:
        f.write(C_SOURCE)
    result = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", so_path, c_path, "-ldl"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("kompilacja nie powiodła się:\n" + result.stderr[-800:])
    return so_path


def enable():
    """Wywołać zaraz po 'import torch', zanim cokolwiek trafi na GPU."""
    if not sys.platform.startswith("linux"):
        return  # Windows robi to sam
    if os.environ.get("NO_GPU_SPILL") or not torch.cuda.is_available():
        return
    try:
        allocator = torch.cuda.memory.CUDAPluggableAllocator(_build(), "managed_malloc", "managed_free")
        torch.cuda.memory.change_current_allocator(allocator)
        print("[ok] Przelew VRAM -> RAM włączony (Unified Memory)")
    except Exception as exc:
        print(f"[!] Nie udało się włączyć przelewu VRAM -> RAM: {exc}", file=sys.stderr)
