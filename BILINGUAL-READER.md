# Bilingual Reader (fork patch)

Adds a paragraph-level translation toggle to the Reader view of
`obsidianmd/obsidian-clipper`. Designed for self-host LLM backends but works with
any OpenAI-compatible chat completions endpoint (DeepL/Anthropic/OpenAI/cloud).

## Why this fork

The upstream Reader is great for clean reading, but offers no built-in way to
read foreign-language articles bilingually. Browser-level translators (Google
Translate, Immersive Translate) inject their own DOM that gets clipped into the
vault as a tangle. Doing it inside the Reader pipeline keeps the vault export
clean: paragraphs stay paired, headings dedup, and a `Save File` from the
Reader writes a tidy `> 原文 \n\n 譯文` markdown for bilingual notes.

Related upstream issue: [#276 — Web Clipper bilingual format readability](https://github.com/obsidianmd/obsidian-clipper/issues/276)

## What it adds

- **Reader nav button (`A`-icon)** + `T` hotkey. Cycles through three states:
  - `off` — original article
  - `bilingual` — original paragraph + translated paragraph (smaller, indented)
  - `target_only` — translation in place of original
- **Heading translation** — `<h1>` title is included; duplicates are deduped.
- **Vault save respects the active state**:
  - `bilingual` state → `> 原文 \n\n 譯文` blockquote-wrapped pairs
  - `target_only` → translation only
  - `off` → original
- **Settings panel** (collapsible, in Reader settings overlay) for endpoint /
  model / target language / system prompt / auto-on toggle.
- **Persistent translation cache** (chrome.storage.local) keyed by
  `(text, model, endpoint, system-prompt)` — reopening the same article
  doesn't re-translate.
- **Backend-agnostic** — defaults to local llama-server on
  `http://127.0.0.1:8843/v1/chat/completions` running Qwen3-4B-Instruct, but
  any OpenAI-compatible endpoint works.

## Architecture

```
content-script ──┬──► reader-script.js (Reader.cycleTranslateState)
                 │       │
                 │       └─► applyTranslationState(article, state)
                 │              │
                 │              └─► translateTexts(batch of 5)
                 │                     │
                 │                     └─► chrome.runtime.sendMessage
                 │                            │
                 │                            ▼
background service worker ──► fetch(endpoint, {model, messages, …})
                                          │
                                          ▼
                                    llama-server / cloud
```

The MV3 background worker is the fetch site so we bypass page-level CSP
(content scripts inherit the page origin and can't reach `localhost` on
strict-CSP sites like note.com).

## Default backend (recommended self-host)

```bash
llama-server \
  -m ~/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q8_0.gguf \
  --jinja \
  --host 127.0.0.1 --port 8843 \
  -c 32768 -ngl 999 --parallel 4 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.3 --top-p 0.9 --top-k 40 --min-p 0.0 \
  --repeat-penalty 1.05 --repeat-last-n 1024 \
  --dry-multiplier 0.5 --dry-penalty-last-n -1 \
  --alias qwen3-4b-translator
```

`Qwen3-4B-Instruct-2507` covers 119 languages and translates EN/JP→繁中
cleanly (verified against technical and literary samples). Q8 GGUF is ~4 GB.
You can swap any OpenAI-compatible endpoint via the in-Reader settings panel.

## Build

Standard upstream toolchain — Node 18+, Webpack:

```bash
npm install
npm run dev:chrome   # → ./dev/ (load unpacked)
npm run build:chrome # → ./build/ (production)
```

Load the `dev/` folder via `chrome://extensions/` (developer mode → Load
unpacked).

## Phase scope

- **Phase 1 (current)**: self-use, hardcoded local backend default,
  in-memory + persistent cache, Reader UI toggle, Reader-internal vault save.
- **Phase 1.5**: settings panel, persistent cache, popup-iframe save hook,
  Localized strings (this branch).
- **Phase 2 (planned, only if upstream interest)**: provider abstraction
  reusing the existing Interpreter framework, fully localized labels, schema-
  validated batch JSON instead of separator parsing, Lazy-load on viewport.

## Files touched

| File | Why |
|---|---|
| `src/utils/translator.ts` | Translation client, batch + retry/fallback, cache |
| `src/utils/reader.ts` | Nav button, hotkey, state machine, settings panel |
| `src/styles/reader/_bilingual.scss` | Translated-paragraph + settings panel styling |
| `src/reader.scss` | Import bilingual partial |
| `src/background.ts` | Service-worker proxy for translator fetch |
| `src/content.ts` | Refresh save snapshot before opening Obsidian iframe |
| `src/types/types.ts` | `ReaderSettings` translator fields |
| `src/utils/storage-utils.ts` | Default values + load/save mapping |
