---
name: Hardware not detected
about: Your GPU was missed or reported incorrectly
labels: hardware
---

**What was detected**

```
$ llmcalculator scan --json
<paste>
```

**What you actually have**

GPU model, VRAM, and how you normally check it (`nvidia-smi`, `rocm-smi`,
Task Manager, System Information).

**Vendor tool output**, if you have one:

```
$ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
<paste>
```
