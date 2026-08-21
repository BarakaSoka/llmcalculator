"""Hardware description shared by every platform detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

GB = 1024 ** 3


@dataclass
class Accelerator:
    """A GPU, or an Apple Silicon SoC acting as one."""

    name: str
    vendor: str
    """apple | nvidia | amd | intel | cpu"""

    memory_bytes: int
    """Memory the accelerator may actually use for a model."""

    bandwidth_gbs: float = 0.0
    """Memory bandwidth in GB/s. This sets token generation speed."""

    fp16_tflops: float = 0.0
    """Dense fp16 throughput. This sets prefill and training speed."""

    unified_memory: bool = False
    """True when the accelerator shares system RAM (Apple Silicon, iGPUs)."""

    compute_capability: str = ""
    cores: int = 0
    index: int = 0

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / GB

    def supports_bf16(self) -> bool:
        if self.vendor == "nvidia":
            try:
                return float(self.compute_capability) >= 8.0
            except ValueError:
                return True
        return self.vendor in ("apple", "amd")

    def supports_flash_attention(self) -> bool:
        if self.vendor == "nvidia":
            try:
                return float(self.compute_capability) >= 8.0
            except ValueError:
                return False
        return self.vendor in ("apple", "amd")


@dataclass
class CPU:
    name: str
    cores: int
    threads: int
    performance_cores: int = 0
    arch: str = ""
    bandwidth_gbs: float = 0.0


@dataclass
class HardwareProfile:
    """Everything llmcalculator needs to know about a machine."""

    cpu: CPU
    ram_bytes: int
    accelerators: List[Accelerator] = field(default_factory=list)
    platform: str = ""
    os_version: str = ""
    disk_free_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / GB

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_bytes / GB

    @property
    def has_gpu(self) -> bool:
        return any(a.vendor != "cpu" for a in self.accelerators)

    @property
    def primary(self) -> Optional[Accelerator]:
        """The accelerator a single-device run would land on."""
        real = [a for a in self.accelerators if a.vendor != "cpu"]
        if not real:
            return self.accelerators[0] if self.accelerators else None
        return max(real, key=lambda a: a.memory_bytes)

    @property
    def total_vram_bytes(self) -> int:
        """Combined accelerator memory. Only meaningful with tensor parallelism."""
        return sum(a.memory_bytes for a in self.accelerators if a.vendor != "cpu")

    @property
    def is_unified(self) -> bool:
        p = self.primary
        return bool(p and p.unified_memory)

    def budget_bytes(self, device: str = "auto") -> int:
        """Memory available to hold a model and its working set.

        On unified-memory machines the GPU and CPU draw on the same pool, so
        the two options differ only in the reserve taken by the OS.
        """
        if device == "cpu":
            return max(0, self.ram_bytes - self._os_reserve())
        if device == "gpu":
            p = self.primary
            return p.memory_bytes if p else 0
        if device == "all-gpus":
            return self.total_vram_bytes
        p = self.primary
        if p and p.vendor != "cpu":
            return p.memory_bytes
        return max(0, self.ram_bytes - self._os_reserve())

    def _os_reserve(self) -> int:
        """RAM the operating system and running apps need to stay responsive."""
        if self.ram_gb <= 8:
            return int(3 * GB)
        if self.ram_gb <= 16:
            return int(4 * GB)
        if self.ram_gb <= 32:
            return int(6 * GB)
        return int(8 * GB)

    def summary_lines(self) -> List[str]:
        out = [
            "CPU        {} ({} cores / {} threads)".format(
                self.cpu.name, self.cpu.cores, self.cpu.threads),
            "RAM        {:.1f} GB".format(self.ram_gb),
        ]
        for a in self.accelerators:
            if a.vendor == "cpu":
                continue
            line = "GPU        {} - {:.1f} GB".format(a.name, a.memory_gb)
            if a.bandwidth_gbs:
                line += ", {:.0f} GB/s".format(a.bandwidth_gbs)
            if a.unified_memory:
                line += " (unified)"
            out.append(line)
        if not self.has_gpu:
            out.append("GPU        none detected - CPU inference only")
        if self.disk_free_bytes:
            out.append("Disk free  {:.0f} GB".format(self.disk_free_gb))
        return out
