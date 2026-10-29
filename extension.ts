/**
 * pi-vision-bridge
 *
 * Gives a text-only primary model vision ability by delegation: the model calls
 * a `describe_image` tool, the tool forwards the image to a vision-capable model
 * from your own models.json, and the text answer is fed back. The main model
 * never touches pixels.
 *
 * Design notes:
 *  - No hardcoded model list: the candidate pool is discovered at runtime from
 *    <agent-dir>/models.json (every model whose `input` includes "image" and
 *    whose provider key resolves).
 *  - Follows the live model: the tool is only active while the selected model
 *    lacks image input; other tools are untouched.
 *  - Pool order is shuffled per call; failures fall through to the next model.
 *  - Credentials: apiKey values are only ever sent in request headers, never
 *    returned to the model or written to logs.
 *  - Optional override: set PI_VISION_POOL to a JSON array of model ids to pin
 *    the pool, or [] to disable.
 */

import { Type } from "typebox";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

// Resolve models.json apiKey fields: literal, "$ENV_VAR", or "!command".
function resolveApiKey(value) {
  if (typeof value !== "string" || !value) return null;
  if (value.startsWith("$")) return process.env[value.slice(1)] || null;
  if (value.startsWith("!")) {
    try {
      const out = execFileSync(value.slice(1), { shell: true, timeout: 5000, encoding: "utf8" });
      return out.trim() || null;
    } catch {
      return null;
    }
  }
  return value;
}

function visionPool(models) {
  const injected = process.env.PI_VISION_POOL;
  if (injected) {
    try {
      const parsed = JSON.parse(injected);
      // An empty array means "computed: nothing can see", not "not set".
      if (Array.isArray(parsed))
        return parsed.filter((m) => typeof m === "string" && m.trim());
    } catch {
      // Broken override: fall back to scanning models.json.
    }
  }
  const found = [];
  for (const prov of Object.values(models?.providers || {})) {
    for (const m of Array.isArray(prov.models) ? prov.models : []) {
      if (m?.id && Array.isArray(m.input) && m.input.includes("image"))
        found.push({ modelId: m.id, model: m });
    }
  }
  return found;
}

function shuffled(items) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

const MIME_BY_EXT = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
};

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function trimTrailingSlash(s) {
  return typeof s === "string" ? s.replace(/\/+$/, "") : s;
}

// Find which provider hosts a model id and merge its overrides/headers.
function findModelProvider(models, modelId) {
  if (!models?.providers) return null;
  for (const [provider, prov] of Object.entries(models.providers)) {
    const ms = Array.isArray(prov.models) ? prov.models : [];
    const m = ms.find((x) => x && x.id === modelId);
    if (!m) continue;
    return {
      provider,
      api: m.api || prov.api,
      baseUrl: m.baseUrl || prov.baseUrl,
      apiKey: resolveApiKey(prov.apiKey),
      headers: { ...(prov.headers || {}), ...(m.headers || {}) },
      modelId,
    };
  }
  return null;
}

function modelInput(models, modelId) {
  if (!models?.providers) return null;
  for (const prov of Object.values(models.providers)) {
    const ms = Array.isArray(prov.models) ? prov.models : [];
    const m = ms.find((x) => x && x.id === modelId);
    if (m) return Array.isArray(m.input) ? m.input : ["text"];
  }
  return null;
}

function buildEndpoint(api, baseUrl) {
  const b = trimTrailingSlash(baseUrl);
  if (!b) return null;
  if (api === "anthropic-messages") return `${b}/v1/messages`;
  if (api === "openai-completions")
    return b.endsWith("/v1") ? `${b}/chat/completions` : `${b}/v1/chat/completions`;
  if (api === "openai-responses" || api === "azure-openai-responses" || api === "openai-codex-responses")
    return b.endsWith("/v1") ? `${b}/responses` : `${b}/v1/responses`;
  return null;
}

// ---- per-protocol calls, unified { ok, text, error } ----
// Streamed requests are used because some relays mandate stream:true; both
// protocols parse the same SSE line shape they emit.

async function callAnthropic(endpoint, key, headers, model, b64, mime, question) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": key,
      authorization: `Bearer ${key}`,
      "anthropic-version": "2023-06-01",
      ...headers,
    },
    signal: AbortSignal.timeout(60000),
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: question },
            { type: "image", source: { type: "base64", media_type: mime, data: b64 } },
          ],
        },
      ],
    }),
  });
  const j = await res.json().catch(() => ({}));
  const text = Array.isArray(j?.content)
    ? j.content.map((c) => c?.text).filter(Boolean).join("\n").trim()
    : "";
  if (!res.ok || !text) {
    return { ok: false, error: `HTTP ${res.status} ${JSON.stringify(j?.error || j).slice(0, 200)}` };
  }
  return { ok: true, text };
}

async function readSseText(res, pickDelta) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let text = "";
  let streamErr = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line.startsWith("data:")) continue;
      const d = line.slice(5).trim();
      if (d === "" || d === "[DONE]") continue;
      try {
        const o = JSON.parse(d);
        if (o.error) {
          streamErr = JSON.stringify(o.error).slice(0, 200);
          break;
        }
        const delta = pickDelta(o);
        if (typeof delta === "string") text += delta;
      } catch {
        // skip unparseable line
      }
    }
    if (streamErr) break;
  }
  return { text: text.trim(), streamErr };
}

async function callOpenAIChat(endpoint, key, headers, model, b64, mime, question) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${key}`,
      ...headers,
    },
    signal: AbortSignal.timeout(90000),
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      stream: true,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: question },
            { type: "image_url", image_url: { url: `data:${mime};base64,${b64}` } },
          ],
        },
      ],
    }),
  });
  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    return { ok: false, error: `HTTP ${res.status} ${t.slice(0, 200)}` };
  }
  const { text, streamErr } = await readSseText(res, (o) => {
    for (const ch of o.choices || []) {
      const c = ch?.delta?.content;
      if (typeof c === "string") return c;
    }
    return undefined;
  });
  if (streamErr) return { ok: false, error: streamErr };
  if (!text) return { ok: false, error: "empty stream content" };
  return { ok: true, text };
}

async function callOpenAIResponses(endpoint, key, headers, model, b64, mime, question) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${key}`,
      ...headers,
    },
    signal: AbortSignal.timeout(90000),
    body: JSON.stringify({
      model,
      stream: true,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: question },
            { type: "input_image", image_url: `data:${mime};base64,${b64}` },
          ],
        },
      ],
    }),
  });
  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    return { ok: false, error: `HTTP ${res.status} ${t.slice(0, 200)}` };
  }
  // response.output_text.delta carries text increments
  const { text, streamErr } = await readSseText(res, (o) => o.delta);
  if (streamErr) return { ok: false, error: streamErr };
  if (!text) return { ok: false, error: "empty responses stream content" };
  return { ok: true, text };
}

async function callByApi(api, v, b64, mime, question) {
  if (api === "anthropic-messages")
    return callAnthropic(v.endpoint, v.apiKey, v.headers, v.modelId, b64, mime, question);
  if (api === "openai-completions")
    return callOpenAIChat(v.endpoint, v.apiKey, v.headers, v.modelId, b64, mime, question);
  return callOpenAIResponses(v.endpoint, v.apiKey, v.headers, v.modelId, b64, mime, question);
}

export default async function (pi) {
  const piDir = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
  const models = readJson(path.join(piDir, "models.json"));
  if (!models?.providers) {
    // No provider config: stay silent.
    return;
  }

  const pool = [];
  for (const entry of visionPool(models)) {
    const id = typeof entry === "string" ? entry : entry.modelId;
    const p = findModelProvider(models, id);
    if (p && p.apiKey && p.api) {
      const endpoint = buildEndpoint(p.api, p.baseUrl);
      if (endpoint) pool.push({ ...p, endpoint });
    }
  }
  if (pool.length === 0) {
    return;
  }

  // Tool activation follows the actually selected model, including restores.
  function syncModel(model) {
    if (!model) return;
    const input = modelInput(models, model.id) || model.input;
    const hasVision = Array.isArray(input) && input.includes("image");
    const active = pi.getActiveTools().filter((name) => name !== "describe_image");
    if (!hasVision) active.push("describe_image");
    pi.setActiveTools(active);
  }
  pi.on("session_start", (_event, ctx) => syncModel(ctx.model));
  pi.on("model_select", (event) => syncModel(event.model));

  pi.registerTool({
    name: "describe_image",
    label: "Describe Image",
    description:
      "Analyze an image file (screenshot, photo, diagram, chart, UI, error popup, sketch) and return a TEXT description. " +
      "Use this ONLY when you need visual understanding and the current model cannot see images natively. " +
      "Pass a local file path; do not pass URLs. Returns text, not pixels.",
    promptSnippet:
      "Relay image to a configured vision-capable model when the main model has no image input",
    promptGuidelines: [
      "Call describe_image ONLY for genuine visual content: screenshots, photos, diagrams, charts, UI captures, error popups, sketches.",
      "Do NOT call it for code, plain text, configs, logs, or any file whose content is text — use the read tool instead.",
      "If the main model already supports image input (multimodal), do not call this tool; read the image directly.",
      "Pass a precise `question` (e.g. 'Read all visible text', 'Describe the UI layout and colors', 'Diagnose the error in this screenshot').",
    ],
    parameters: Type.Object({
      path: Type.String({
        description:
          "Local file path to the image (png/jpg/jpeg/gif/webp/bmp). Relative paths resolve against the current working directory.",
      }),
      question: Type.Optional(
        Type.String({
          description:
            "What to extract from the image. Defaults to a thorough general description. " +
            "Examples: 'Read all text verbatim', 'Describe the layout', 'What error is shown and how to fix it'.",
        })
      ),
    }),
    async execute(_toolCallId, params, _signal) {
      const imgPath = params?.path;
      const question =
        (params?.question && String(params.question).trim()) ||
        "Describe this image in detail: visible text (verbatim), layout, colors, and notable elements.";
      try {
        if (!imgPath || !fs.existsSync(imgPath)) {
          return {
            content: [{ type: "text", text: `Error: image file not found: ${imgPath}` }],
            details: { ok: false },
          };
        }
        const buf = fs.readFileSync(imgPath);
        const b64 = buf.toString("base64");
        const ext = path.extname(imgPath).toLowerCase();
        const mime = MIME_BY_EXT[ext] || "image/png";

        const tried = [];
        for (const v of shuffled(pool)) {
          const r = await callByApi(v.api, v, b64, mime, question);
          if (r.ok) {
            return {
              content: [{ type: "text", text: `[vision via ${v.modelId}]\n${r.text}` }],
              details: { ok: true, model: v.modelId, provider: v.provider, tried },
            };
          }
          tried.push(`${v.modelId} (${v.api}): ${r.error}`);
        }
        return {
          content: [
            {
              type: "text",
              text: `Every vision-capable model in the pool failed.\nAttempts:\n${tried.map((t) => "- " + t).join("\n")}`,
            },
          ],
          details: { ok: false, tried },
        };
      } catch (e) {
        return {
          content: [{ type: "text", text: `describe_image error: ${e?.message || e}` }],
          details: { ok: false },
        };
      }
    },
  });
}
