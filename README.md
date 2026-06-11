# learn-llm-by-doing

[中文版 →](README.zh-CN.md)

A self-study curriculum for learning how large language models actually work — by building the core pieces from scratch rather than reading about them.

It covers three areas:

- **Training** — write a GPT from scratch, then scale it across multiple GPUs.
- **Inference** — build a minimal version of a serving engine like vLLM, and compare it to the real thing.
- **Agents** — the design decisions behind multi-agent systems, with a controlled experiment.

Each area is broken into small modules. Every module gives you what to read, something to build, and a short self-check to confirm you understood it.

## What's here

```
courses/        the curriculum (markdown): one folder per module
build_site.py   turns the curriculum into a browsable website
site/           the generated website — open site/index.html to start
```

## Using it

```bash
open site/index.html        # browse the curriculum
```

To edit a module, change the markdown under `courses/` and regenerate:

```bash
python3 build_site.py
```

---

> A personal learning project. The goal is to understand the mechanics, not to chase benchmark numbers.
