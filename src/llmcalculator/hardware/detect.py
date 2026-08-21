"""Detect what this machine actually has.

Every probe is read-only and defensive: a missing vendor tool, a permission
error or an unparseable line degrades to "not found" rather than raising,
because a partial hardware picture is still useful.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import List, Optional

from .base import GB, CPU, Accelerator, HardwareProfile
from . import gpu_db

_TIMEOUT = 8.0


def _run(cmd: List[str], timeout: float = _TIMEOUT) -> Optional[str]:
    """Run a command, returning stdout or None if it fails in any way."""
    if not shutil.which(cmd[0]):
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout


def detect() -> HardwareProfile:
    """Build a HardwareProfile for the current machine."""
    system = platform.system()
    profile = HardwareProfile(
        cpu=_detect_cpu(),
        ram_bytes=_detect_ram(),
        platform=system,
        os_version=platform.release(),
        disk_free_bytes=_detect_disk(),
    )

    accels: List[Accelerator] = []
    if system == "Darwin" and platform.machine() == "arm64":
        accels += _detect_apple(profile)
    accels += _detect_nvidia()
    accels += _detect_amd()
    if system == "Linux":
        accels += _detect_intel()

    profile.accelerators = accels
    if not accels:
        profile.notes.append(
            "No GPU detected. Everything below assumes CPU inference, which works "
            "but runs roughly 3-10x slower than a GPU of the same memory size."
        )
    profile.cpu.bandwidth_gbs = gpu_db.cpu_bandwidth(platform.machine(), profile.cpu.name)
    return profile


# --- CPU / RAM / disk -----------------------------------------------------

def _detect_cpu() -> CPU:
    system = platform.system()
    name = platform.processor() or platform.machine()
    cores = os.cpu_count() or 1
    threads = cores
    perf = 0

    if system == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            name = brand.strip()
        phys = _run(["sysctl", "-n", "hw.physicalcpu"])
        if phys and phys.strip().isdigit():
            cores = int(phys.strip())
        p = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if p and p.strip().isdigit():
            perf = int(p.strip())
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                text = fh.read()
            m = re.search(r"model name\s*:\s*(.+)", text)
            if m:
                name = m.group(1).strip()
            ids = set(re.findall(r"core id\s*:\s*(\d+)", text))
            if ids:
                cores = len(ids)
        except OSError:
            pass
    elif system == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER", name)

    return CPU(name=name.strip(), cores=cores, threads=threads,
               performance_cores=perf, arch=platform.machine())


def _detect_ram() -> int:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            return int(out.strip())
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as fh:
                m = re.search(r"MemTotal:\s*(\d+)\s*kB", fh.read())
            if m:
                return int(m.group(1)) * 1024
        except OSError:
            pass
    elif system == "Windows":
        try:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = Status()
            st.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        except Exception:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 8 * GB


def _detect_disk() -> int:
    try:
        return shutil.disk_usage(os.path.expanduser("~")).free
    except OSError:
        return 0


# --- Apple Silicon --------------------------------------------------------

def _detect_apple(profile: HardwareProfile) -> List[Accelerator]:
    """Apple Silicon: the GPU shares system RAM, capped by a wired limit."""
    chip = profile.cpu.name
    out = _run(["system_profiler", "-json", "SPHardwareDataType"])
    if out:
        try:
            data = json.loads(out)["SPHardwareDataType"][0]
            chip = data.get("chip_type") or data.get("cpu_type") or chip
        except (KeyError, IndexError, ValueError):
            pass

    cores = 0
    disp = _run(["system_profiler", "-json", "SPDisplaysDataType"])
    if disp:
        try:
            gpus = json.loads(disp)["SPDisplaysDataType"]
            cores = int(gpus[0].get("sppci_cores", 0) or 0)
        except (KeyError, IndexError, ValueError, TypeError):
            pass

    limit = _apple_gpu_limit(profile.ram_bytes)
    bw, tflops = gpu_db.lookup(chip)
    if cores and tflops:
        # Scale the reference figure by actual core count where we know it.
        ref_cores = {"pro": 18, "max": 40, "ultra": 80}
        base = next((v for k, v in ref_cores.items() if k in chip.lower()), 10)
        tflops *= cores / base

    profile.notes.append(
        "Apple Silicon shares one memory pool between CPU and GPU. macOS lets the "
        "GPU wire down about {:.0f} GB of your {:.0f} GB; raise it with "
        "`sudo sysctl iogpu.wired_limit_mb=N` if you need more.".format(limit / GB, profile.ram_gb)
    )
    return [Accelerator(name=chip, vendor="apple", memory_bytes=limit, bandwidth_gbs=bw,
                        fp16_tflops=tflops, unified_memory=True, cores=cores)]


def _apple_gpu_limit(ram_bytes: int) -> int:
    """How much unified memory macOS will let the GPU hold."""
    out = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    if out and out.strip().isdigit() and int(out.strip()) > 0:
        return int(out.strip()) * 1024 * 1024
    ram_gb = ram_bytes / GB
    if ram_gb > 36:
        return int((ram_gb - 8) * GB)
    return int(ram_gb * 0.75 * GB)


# --- NVIDIA ---------------------------------------------------------------

def _detect_nvidia() -> List[Accelerator]:
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    accels = []
    for i, line in enumerate(out.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            mem = int(float(parts[1])) * 1024 * 1024  # nvidia-smi reports MiB
        except ValueError:
            continue
        cap = parts[2] if len(parts) > 2 else ""
        bw, tflops = gpu_db.lookup(name)
        accels.append(Accelerator(name=name, vendor="nvidia", memory_bytes=mem,
                                  bandwidth_gbs=bw, fp16_tflops=tflops,
                                  compute_capability=cap, index=i))
    return accels


# --- AMD ------------------------------------------------------------------

def _detect_amd() -> List[Accelerator]:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            data = json.loads(out)
        except ValueError:
            data = {}
        accels = []
        for i, (card, info) in enumerate(sorted(data.items())):
            if not isinstance(info, dict):
                continue
            name = (info.get("Card Series") or info.get("Card model")
                    or info.get("Card SKU") or card)
            mem = 0
            for key, val in info.items():
                if "Total" in key and "vram" in key.lower():
                    try:
                        mem = int(val)
                    except (TypeError, ValueError):
                        pass
            if not mem:
                continue
            bw, tflops = gpu_db.lookup(str(name))
            accels.append(Accelerator(name=str(name), vendor="amd", memory_bytes=mem,
                                      bandwidth_gbs=bw, fp16_tflops=tflops, index=i))
        if accels:
            return accels
    return []


# --- Intel ----------------------------------------------------------------

def _detect_intel() -> List[Accelerator]:
    out = _run(["xpu-smi", "discovery", "-j"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    accels = []
    for i, dev in enumerate(data.get("device_list", [])):
        name = dev.get("device_name", "Intel GPU")
        mem = 0
        try:
            mem = int(float(dev.get("memory_physical_size_byte", 0)))
        except (TypeError, ValueError):
            pass
        if not mem:
            continue
        bw, tflops = gpu_db.lookup(name)
        accels.append(Accelerator(name=name, vendor="intel", memory_bytes=mem,
                                  bandwidth_gbs=bw, fp16_tflops=tflops, index=i))
    return accels


# --- manual override ------------------------------------------------------

def manual(vram_gb: float, ram_gb: float, gpu_name: str = "Custom GPU",
           cores: int = 8) -> HardwareProfile:
    """Build a profile by hand, for planning a machine you do not own yet."""
    bw, tflops = gpu_db.lookup(gpu_name)
    accels = []
    if vram_gb > 0:
        accels.append(Accelerator(name=gpu_name, vendor="nvidia",
                                  memory_bytes=int(vram_gb * GB),
                                  bandwidth_gbs=bw, fp16_tflops=tflops))
    return HardwareProfile(
        cpu=CPU(name="Specified CPU", cores=cores, threads=cores * 2, bandwidth_gbs=50.0),
        ram_bytes=int(ram_gb * GB),
        accelerators=accels,
        platform="manual",
    )
