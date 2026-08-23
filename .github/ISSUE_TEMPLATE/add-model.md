---
name: Add a model
about: Request a model be added to the built-in catalog
labels: catalog
---

**Model**

Hugging Face repo id, for example `Qwen/Qwen3-8B`.

**Note:** you may not need this. Any model on the Hub already works:

```
llmcalculator check Qwen/Qwen3-8B
```

The catalog exists so common models are available offline and appear in
`recommend`. If this model is widely used locally, it is worth adding.

**Architecture** (optional — copy from the model's `config.json`)

```json
{
  "num_hidden_layers": ,
  "hidden_size": ,
  "num_attention_heads": ,
  "num_key_value_heads": ,
  "head_dim": ,
  "vocab_size": ,
  "max_position_embeddings":
}
```
