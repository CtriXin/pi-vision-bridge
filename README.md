# pi-vision-bridge

A [pi coding agent](https://github.com/earendil-works/pi) extension that gives a
**text-only model access to vision** — by delegation, not by pretending.

The main model calls a `describe_image` tool. The tool relays the image to a
vision-capable model discovered from your own `models.json`, and the text answer
comes back. The main model never sees pixels.

## Why

Many cheap or fast models (deepseek-chat, glm, qwen text tiers, small local
models) are text-only. When the agent hits a screenshot, diagram or error
popup, it normally cannot look at it. This extension registers `describe_image`
and delegates to a vision model you already pay for / already configured.

## How it works

- Scans `<agent-dir>/models.json` (default `~/.pi/agent/models.json`, or
  `$PI_CODING_AGENT_DIR/models.json`) for every model whose `input` includes
  `"image"`.
- Resolves each candidate's provider `api`, `baseUrl` and `apiKey` (literal,
  `"$ENV_VAR"`, or `"!command"` — same interpolation as pi).
- Activates the tool **only while the selected model lacks image input**.
  Switch to a vision model and the tool hides itself; other tools are untouched.
- On each call, shuffles the pool and falls back on failure until one model
  answers. Supported protocols: `anthropic-messages`, `openai-completions`,
  `openai-responses` (+ `azure-openai-responses`, `openai-codex-responses`).

## Install

Copy `extension.ts` into your pi extensions dir:

```bash
mkdir -p ~/.pi/agent/extensions
curl -O https://raw.githubusercontent.com/CtriXin/pi-vision-bridge/main/extension.ts \
  --output-dir ~/.pi/agent/extensions
```

Requires a `models.json` entry with `"input": ["text", "image"]` on at least one
model, e.g.

```jsonc
{
  "providers": {
    "openrouter": {
      "api": "openai-completions",
      "baseUrl": "https://openrouter.ai/api",
      "apiKey": "$OPENROUTER_API_KEY",
      "models": [
        { "id": "google/gemini-2.5-flash", "input": ["text", "image"] }
      ]
    }
  }
}
```

Restart pi, pick any text-only model, and the agent gets `describe_image`.

## Configuration

| Env var | Effect |
| --- | --- |
| `PI_VISION_POOL` | JSON array of model ids to pin the delegation pool, e.g. `'["google/gemini-2.5-flash"]'`. `[]` disables the extension. Unset → discover all image-input models. |
| `PI_CODING_AGENT_DIR` | Where to find `models.json` (pi's own convention). |

## Notes

- Only `apiKey` values leave the process — inside request headers to the
  model's own provider. Nothing is logged or returned to the model.
- `describe_image` takes a **local path**, not a URL. For text files, code,
  configs or logs the agent is told to use `read` instead — the tool is only
  for genuinely visual content.
- Requests use streaming (`stream: true`) because some OpenAI-compatible
  relays require it; both SSE parsers handle the standard delta shapes.

## License

MIT
