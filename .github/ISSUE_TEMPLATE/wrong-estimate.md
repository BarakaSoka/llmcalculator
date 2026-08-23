---
name: Wrong estimate
about: The tool said something fits when it does not, or vice versa
labels: estimate
---

**What the tool said**

```
$ llmcalculator check <model>
<paste the output>
```

**What actually happened**

For example: ran out of memory at 12 GB, or it fit fine with room to spare.

**Your hardware**

```
$ llmcalculator scan --json
<paste>
```

**Runtime used**

Ollama / llama.cpp / vLLM / transformers / MLX, and the version.
