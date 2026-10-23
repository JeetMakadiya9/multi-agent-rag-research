// The only module that talks to the studio server. Every failure becomes an ApiError with a readable message.

export class ApiError extends Error {
  constructor(message, { status = 0, kind = "http", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;            // "http" | "network" | "timeout" | "format" | "aborted"
    this.payload = payload;      // the server's JSON body, when it sent one
  }
}

// Job states after which nothing more happens on the server.
export const FINAL_STATES = new Set(["done", "cancelled"]);

const enc = encodeURIComponent;

// API paths are written from the server root ("/api/..."). In a browser they are resolved against the folder of the
// studio page, so the studio also works when a proxy serves it under a sub-path.
export function apiUrl(path) {
  const page = typeof location !== "undefined" && location && typeof location.pathname === "string" ? location.pathname : "/";
  return path.startsWith("/") ? page.slice(0, page.lastIndexOf("/") + 1) + path.slice(1) : path;
}

export async function request(path, { method = "GET", body, headers = {}, timeout = 20000, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeout);
  const onAbort = () => controller.abort("aborted");
  if (signal) {
    if (signal.aborted) controller.abort("aborted");
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  let response;
  try {
    response = await fetch(apiUrl(path), { method, body, headers, signal: controller.signal, credentials: "same-origin" });
  } catch (error) {
    if (controller.signal.aborted && controller.signal.reason === "aborted") {
      throw new ApiError("The request was cancelled.", { kind: "aborted" });
    }
    if (controller.signal.aborted) throw new ApiError("The studio server did not respond in time.", { kind: "timeout" });
    throw new ApiError("The studio server could not be reached. Is `python -m webrag.server` running?", { kind: "network" });
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onAbort);
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(`The server sent an unexpected response (HTTP ${response.status}).`, {
        status: response.status, kind: "format" });
    }
  }
  if (!response.ok) {
    const message = payload && typeof payload.error === "string" ? payload.error : `Request failed (HTTP ${response.status}).`;
    throw new ApiError(message, { status: response.status, payload });
  }
  if (payload === null || typeof payload !== "object") {
    throw new ApiError("The server sent an empty or malformed response.", { status: response.status, kind: "format" });
  }
  return payload;
}

const json = (value) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
const context = (source, passage) => `context?source=${enc(source)}&passage=${enc(passage)}`;

export const api = {
  status: (opts) => request("/api/status", { timeout: 15000, ...opts }),
  jobs: (opts) => request("/api/jobs", opts),
  submit: (input, mode, options) => request("/api/run", json({ input, mode, ...(options || {}) })),
  providers: (opts) => request("/api/providers", { timeout: 30000, ...opts }),
  saveProviderConfig: (patch) => request("/api/providers/config", json(patch)),
  testProvider: (id) => request(`/api/providers/${enc(id)}/test`, { method: "POST", timeout: 60000 }),
  job: (id, opts) => request(`/api/run/${enc(id)}`, { timeout: 30000, ...opts }),
  cancel: (id) => request(`/api/run/${enc(id)}/cancel`, { method: "POST" }),
  jobContext: (id, source, passage) => request(`/api/run/${enc(id)}/${context(source, passage)}`),
  runs: (opts) => request("/api/runs", { timeout: 30000, ...opts }),
  run: (name, opts) => request(`/api/runs/${enc(name)}`, { timeout: 30000, ...opts }),
  runContext: (name, source, passage) => request(`/api/runs/${enc(name)}/${context(source, passage)}`),
  evaluations: (opts) => request("/api/evaluations", { timeout: 30000, ...opts }),
  evaluation: (name, opts) => request(`/api/evaluations/${enc(name)}`, opts),
  evaluationDoc: (name) => request(`/api/evaluations/docs/${enc(name)}`),
  library: (opts) => request("/api/library", { timeout: 60000, ...opts }),
  sources: (opts) => request("/api/sources", { timeout: 60000, ...opts }),
  integrations: (opts) => request("/api/integrations", { timeout: 60000, ...opts }),
  testOllama: () => request("/api/integrations/ollama/test", { method: "POST", timeout: 20000 }),
  obsidian: (opts) => request("/api/integrations/obsidian", opts),
  exportToObsidian: (run) => request("/api/export/obsidian", json({ run })),
  exportUrl: (name, format) => apiUrl(`/api/runs/${enc(name)}/export?format=${enc(format)}`),
  upload: (file) => request("/api/library/upload", {
    method: "POST", body: file, timeout: 120000,
    headers: { "Content-Type": "application/octet-stream", "X-Filename": enc(file.name) } }),
};

// Poll a job until it is done or cancelled. Transient failures are retried with back-off; persistent ones are
// reported. Aborting `signal` stops polling at once (the in-flight request is aborted too) and resolves to null;
// no update is delivered after the abort, so a stale response can never overwrite newer state.
export async function pollJob(id, onUpdate, { interval = 1500, signal, maxErrors = 5 } = {}) {
  let errors = 0;
  const wait = (ms) => new Promise((resolve) => {
    const done = () => { clearTimeout(timer); if (signal) signal.removeEventListener("abort", done); resolve(); };
    const timer = setTimeout(done, ms);
    if (signal) signal.addEventListener("abort", done, { once: true });
  });
  for (;;) {
    if (signal && signal.aborted) return null;
    try {
      const job = await api.job(id, { signal });
      if (signal && signal.aborted) return null;
      errors = 0;
      onUpdate(job);
      if (FINAL_STATES.has(job.state)) return job;
    } catch (error) {
      if (error.kind === "aborted" || (signal && signal.aborted)) return null;
      errors += 1;
      if (errors >= maxErrors || error.status === 404) throw error;
    }
    await wait(errors ? interval * (errors + 1) : interval);
  }
}
