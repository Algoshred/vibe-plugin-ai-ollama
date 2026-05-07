/**
 * vibe-plugin-ai-ollama
 *
 * Ollama AI agent provider for VibeControls Agent.
 * Implements the AIAgentProvider interface with dual-mode support:
 * - SDK mode: Uses the `ollama` npm package (Ollama JS client) for direct API access
 * - CLI mode: Uses the `ollama` binary with `run` subcommand
 *
 * Supports both Ollama Cloud (https://ollama.com) and self-hosted
 * (e.g. http://localhost:11434) via the OLLAMA_HOST env / config var.
 *
 * Auth: OLLAMA_API_KEY (cloud) or none (self-hosted).
 *
 * Mode auto-detection:
 *   - SDK if OLLAMA_API_KEY or OLLAMA_HOST set, or local daemon reachable
 *   - CLI if `ollama` binary is on PATH
 *   - Default: SDK (with localhost host) — healthCheck reports actual state.
 */

import { Elysia } from "elysia";

// ── Locally Redeclared Interfaces ────────────────────────────────────────
// (Avoid hard dependency on @vibecontrols/agent)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

interface VibePlugin {
  name: string;
  version: string;
  description?: string;
  tags?: Array<
    "backend" | "frontend" | "cli" | "provider" | "adapter" | "integration"
  >;
  cliCommand?: string;
  apiPrefix?: string;
  prerequisites?: Array<{
    name: string;
    kind: "binary" | "npm" | "pip" | "cargo" | "manual";
    requiresSudo: boolean;
    description?: string;
  }>;
  createRoutes?: () => unknown;
  providers?: { ai?: AIAgentProvider; [key: string]: unknown };
  onServerStart?: (
    app: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
  onServerStop?: () => void | Promise<void>;
  onCliSetup?: (
    program: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
}

interface HostServices {
  logger?: {
    info: (source: string, msg: string) => void;
    warn: (source: string, msg: string) => void;
    error: (source: string, msg: string) => void;
    debug: (source: string, msg: string) => void;
  };
  serviceRegistry?: {
    getService: <T>(pluginName: string, serviceName: string) => T | undefined;
  };
  getConfig: (key: string) => string | undefined | Promise<string | undefined>;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
}

// Log ingester interface (from ai plugin's service registry)
interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Provider Adapter Interface ──────────────────────────────────────────

interface ProviderAdapter {
  readonly mode: ProviderMode;

  sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  healthCheck(): Promise<{ ok: boolean; message?: string }>;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "ollama";
const CLI_COMMAND = "ollama";
const DISPLAY_NAME = "Ollama";
const DEFAULT_MODEL = "llama3.2";
const DEFAULT_MAX_TOKENS = 4096;
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
const DEFAULT_CLOUD_HOST = "https://ollama.com";
const DEFAULT_LOCAL_HOST = "http://localhost:11434";
// Per https://docs.ollama.com/cli the canonical install method on
// Linux/macOS is the upstream install script.
const CLI_INSTALL_COMMAND = [
  "sh",
  "-c",
  "curl -fsSL https://ollama.com/install.sh | sh",
];

// Reasonable static fallback for environments where listModels()/SDK
// list() cannot be reached (e.g., before any host is configured).
const OLLAMA_FALLBACK_MODELS: AIModelInfo[] = [
  {
    id: "llama3.2",
    name: "Llama 3.2",
    provider: PROVIDER_NAME,
    contextWindow: 128_000,
    maxOutputTokens: 4_096,
    supportsVision: false,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  },
  {
    id: "llama3.1",
    name: "Llama 3.1",
    provider: PROVIDER_NAME,
    contextWindow: 128_000,
    maxOutputTokens: 4_096,
    supportsVision: false,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  },
  {
    id: "qwen2.5",
    name: "Qwen 2.5",
    provider: PROVIDER_NAME,
    contextWindow: 128_000,
    maxOutputTokens: 4_096,
    supportsVision: false,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  },
];

// ── SDK Adapter ─────────────────────────────────────────────────────────

/**
 * Minimal structural type for the `Ollama` class from the `ollama` npm
 * package. We only model what we use here so we don't pull the type
 * surface in at module level.
 */
interface OllamaListModelEntry {
  name: string;
  model?: string;
  modified_at?: string | Date;
  size?: number;
  details?: {
    family?: string;
    parameter_size?: string;
    quantization_level?: string;
  };
}

interface OllamaListResponse {
  models: OllamaListModelEntry[];
}

interface OllamaChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

interface OllamaChatResponse {
  model: string;
  message: { role: string; content: string };
  done: boolean;
  prompt_eval_count?: number;
  eval_count?: number;
}

interface OllamaChatStream extends AsyncIterable<OllamaChatResponse> {
  abort?: () => void;
}

interface OllamaClient {
  list(): Promise<OllamaListResponse>;
  chat(params: {
    model: string;
    messages: OllamaChatMessage[];
    stream?: false;
    options?: Record<string, unknown>;
  }): Promise<OllamaChatResponse>;
  chat(params: {
    model: string;
    messages: OllamaChatMessage[];
    stream: true;
    options?: Record<string, unknown>;
  }): Promise<OllamaChatStream>;
}

interface OllamaConnection {
  host: string;
  apiKey?: string;
}
type OllamaConnectionResolver = () => Promise<OllamaConnection>;

class OllamaSdkAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "sdk";
  private client: OllamaClient | null = null;
  private clientHost: string | null = null;
  private clientApiKey: string | undefined;
  private resolveConnection: OllamaConnectionResolver;

  constructor(resolveConnection: OllamaConnectionResolver) {
    this.resolveConnection = resolveConnection;
  }

  private async getClient(): Promise<OllamaClient> {
    const conn = await this.resolveConnection();
    if (
      this.client &&
      this.clientHost === conn.host &&
      this.clientApiKey === conn.apiKey
    ) {
      return this.client;
    }

    let mod: unknown;
    try {
      mod = await import("ollama");
    } catch {
      throw new Error(
        "Failed to load ollama npm package. Install it with: bun add ollama",
      );
    }

    const m = mod as {
      Ollama?: new (opts: {
        host?: string;
        headers?: Record<string, string>;
      }) => OllamaClient;
      default?: unknown;
    };
    const OllamaCtor = m.Ollama;
    if (!OllamaCtor) {
      throw new Error(
        "ollama npm package did not export the Ollama class as expected",
      );
    }

    const headers: Record<string, string> | undefined = conn.apiKey
      ? { Authorization: `Bearer ${conn.apiKey}` }
      : undefined;

    this.client = new OllamaCtor({ host: conn.host, headers });
    this.clientHost = conn.host;
    this.clientApiKey = conn.apiKey;
    return this.client;
  }

  async listModels(): Promise<AIModelInfo[]> {
    const client = await this.getClient();
    const resp = await client.list();
    return resp.models.map((m) => ({
      id: m.name,
      name: m.name,
      provider: PROVIDER_NAME,
      // Ollama does not advertise context windows via /api/tags. Use a
      // sensible default; callers can override per-session.
      contextWindow: 128_000,
      maxOutputTokens: 4_096,
      supportsVision: false,
      supportsStreaming: true,
      inputPricePerMToken: 0,
      outputPricePerMToken: 0,
    }));
  }

  private buildMessages(
    prompt: string,
    config: AISessionConfig,
  ): OllamaChatMessage[] {
    const messages: OllamaChatMessage[] = [];
    if (config.systemPrompt) {
      messages.push({ role: "system", content: config.systemPrompt });
    }
    messages.push({ role: "user", content: prompt });
    return messages;
  }

  private buildOptions(config: AISessionConfig): Record<string, unknown> {
    const opts: Record<string, unknown> = {};
    if (config.maxTokens) opts["num_predict"] = config.maxTokens;
    if (typeof config.temperature === "number") {
      opts["temperature"] = config.temperature;
    }
    return opts;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const response = await client.chat({
      model,
      messages: this.buildMessages(prompt, config),
      stream: false,
      options: this.buildOptions(config),
    });

    const durationMs = Date.now() - startTime;

    if (signal?.aborted) {
      throw new Error("Request aborted");
    }

    return {
      content: response.message?.content ?? "",
      model: response.model,
      inputTokens: response.prompt_eval_count ?? Math.ceil(prompt.length / 4),
      outputTokens:
        response.eval_count ??
        Math.ceil((response.message?.content ?? "").length / 4),
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const stream = await client.chat({
      model,
      messages: this.buildMessages(prompt, config),
      stream: true,
      options: this.buildOptions(config),
    });

    if (signal) {
      const onAbort = () => {
        try {
          stream.abort?.();
        } catch {
          // best-effort; ignore
        }
      };
      signal.addEventListener("abort", onAbort, { once: true });
    }

    let content = "";
    let outputTokens = 0;
    let inputTokens = 0;
    let finalModel = model;

    try {
      for await (const part of stream) {
        const piece = part.message?.content ?? "";
        if (piece) {
          content += piece;
          onChunk({ type: "text", content: piece });
        }
        if (part.model) finalModel = part.model;
        if (part.done) {
          inputTokens = part.prompt_eval_count ?? inputTokens;
          outputTokens = part.eval_count ?? outputTokens;
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "stream error";
      onChunk({ type: "error", content: msg });
      throw err;
    }

    const durationMs = Date.now() - startTime;
    onChunk({ type: "done", content: "" });

    return {
      content,
      model: finalModel,
      inputTokens: inputTokens || Math.ceil(prompt.length / 4),
      outputTokens: outputTokens || Math.ceil(content.length / 4),
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const client = await this.getClient();
      await client.list();
      const host = this.clientHost ?? "(unknown)";
      return {
        ok: true,
        message: `${DISPLAY_NAME} SDK ready (host: ${host}${
          this.clientApiKey ? ", auth: bearer" : ""
        })`,
      };
    } catch (err) {
      return {
        ok: false,
        message:
          err instanceof Error ? err.message : "SDK initialization failed",
      };
    }
  }
}

// ── CLI Adapter ─────────────────────────────────────────────────────────

class OllamaCliAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "cli";

  private buildEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (typeof v === "string") env[k] = v;
    }
    // Surface OLLAMA_HOST / OLLAMA_API_KEY (already in env if set);
    // nothing extra to inject here.
    return env;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    // `ollama run <model> <prompt>` prints the response to stdout and
    // exits. (No streaming when prompt is passed positionally.)
    const args = ["run", model, prompt];

    const proc = Bun.spawn([CLI_COMMAND, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      env: this.buildEnv(),
      timeout: (config.providerConfig?.["timeoutMs"] as number) || 300_000,
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;
    const durationMs = Date.now() - startTime;

    if (exitCode !== 0 && !stdout) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    const content = stdout.trim() || stderr.trim();
    // CLI does not provide real token counts; approximate from char lengths.
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);

    return {
      content,
      model,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { exitCode, provider: PROVIDER_NAME, mode: "cli" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    // `ollama run` does stream tokens to stdout; we forward each decoded
    // chunk as a text event so the UI sees incremental output.
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;
    const args = ["run", model, prompt];

    const proc = Bun.spawn([CLI_COMMAND, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      env: this.buildEnv(),
      timeout: (config.providerConfig?.["timeoutMs"] as number) || 300_000,
    });

    const decoder = new TextDecoder();
    let content = "";

    try {
      for await (const piece of proc.stdout as ReadableStream<Uint8Array>) {
        const text = decoder.decode(piece, { stream: true });
        if (text) {
          content += text;
          onChunk({ type: "text", content: text });
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "cli stream error";
      onChunk({ type: "error", content: msg });
      throw err;
    }

    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;
    const durationMs = Date.now() - startTime;

    if (exitCode !== 0 && !content) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    onChunk({ type: "done", content: "" });

    const trimmed = content.trim();
    return {
      content: trimmed,
      model,
      inputTokens: Math.ceil(prompt.length / 4),
      outputTokens: Math.ceil(trimmed.length / 4),
      durationMs,
      metadata: { exitCode, provider: PROVIDER_NAME, mode: "cli" },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} CLI ${proc.stdout.toString().trim()}`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not available (exit code ${proc.exitCode})`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not installed or not in PATH`,
      };
    }
  }
}

// ── Provider Implementation ─────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  abortController: AbortController | null;
  files: AIFileAttachment[];
  createdAt: string;
  updatedAt: string;
}

class OllamaProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private activeMode: ProviderMode | null = null;
  private adapter: ProviderAdapter | null = null;
  private cachedApiKey: string | undefined;
  private cachedHost: string | undefined;

  setHostServices(hs: HostServices): void {
    this.hostServices = hs;
    this.logIngester =
      hs.serviceRegistry?.getService<LogIngester>("ai", "log-ingester") ?? null;

    // Warm the cache so detectMode() can see DB-stored credentials.
    void Promise.all([
      Promise.resolve(hs.getConfig("OLLAMA_API_KEY")),
      Promise.resolve(hs.getConfig("OLLAMA_HOST")),
    ])
      .then(([apiKey, host]) => {
        const trimmedKey = apiKey?.trim();
        const trimmedHost = host?.trim();
        if (trimmedKey) this.cachedApiKey = trimmedKey;
        if (trimmedHost) this.cachedHost = trimmedHost;
      })
      .catch(() => {});
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  private async resolveConnection(): Promise<OllamaConnection> {
    const envApiKey = process.env["OLLAMA_API_KEY"]?.trim();
    const envHost = process.env["OLLAMA_HOST"]?.trim();

    let apiKey = envApiKey || this.cachedApiKey;
    let host = envHost || this.cachedHost;

    if ((!apiKey || !host) && this.hostServices) {
      try {
        if (!apiKey) {
          const dbKey = (
            await this.hostServices.getConfig("OLLAMA_API_KEY")
          )?.trim();
          if (dbKey) {
            this.cachedApiKey = dbKey;
            apiKey = dbKey;
          }
        }
        if (!host) {
          const dbHost = (
            await this.hostServices.getConfig("OLLAMA_HOST")
          )?.trim();
          if (dbHost) {
            this.cachedHost = dbHost;
            host = dbHost;
          }
        }
      } catch {
        // swallow — treat as no config
      }
    }

    // If an API key is provided and host is unset, default to Ollama Cloud.
    // Otherwise default to local self-hosted.
    if (!host) {
      host = apiKey ? DEFAULT_CLOUD_HOST : DEFAULT_LOCAL_HOST;
    }

    return apiKey ? { host, apiKey } : { host };
  }

  // ── Mode Management ──────────────────────────────────────────────────

  getMode(): ProviderMode {
    if (this.activeMode) return this.activeMode;
    return this.detectMode();
  }

  setMode(mode: ProviderMode): void {
    if (!SUPPORTED_MODES.includes(mode)) {
      throw new Error(`${DISPLAY_NAME} does not support ${mode} mode`);
    }
    this.activeMode = mode;
    this.adapter = null; // Force re-creation on next use
    this.log("info", `Mode explicitly set to: ${mode}`);
  }

  private detectMode(): ProviderMode {
    if (
      process.env["OLLAMA_API_KEY"]?.trim() ||
      process.env["OLLAMA_HOST"]?.trim() ||
      this.cachedApiKey ||
      this.cachedHost
    ) {
      return "sdk";
    }

    try {
      const finder = process.platform === "win32" ? "where.exe" : "which";
      const proc = Bun.spawnSync([finder, CLI_COMMAND], {
        timeout: 3000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) return "cli";
    } catch {
      // CLI not found
    }

    // Default to SDK mode pointing at localhost; healthCheck will report
    // whether the local daemon is actually reachable.
    return "sdk";
  }

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    const mode = this.getMode();
    this.adapter =
      mode === "sdk"
        ? new OllamaSdkAdapter(() => this.resolveConnection())
        : new OllamaCliAdapter();
    this.activeMode = mode;
    this.log("info", `Adapter initialized in ${mode} mode`);
    return this.adapter;
  }

  // ── Session Management ───────────────────────────────────────────────

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return this.toAISession(existing);
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      abortController: null,
      files: [],
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log("info", `Session created: ${id} (${config.name})`);

    return this.toAISession(session);
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const result = await adapter.sendPrompt(
        fullPrompt,
        session.config,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const chunkHandler = onChunk ?? ((_c: AIStreamChunk) => {});

      const result = await adapter.streamPrompt(
        fullPrompt,
        session.config,
        chunkHandler,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  // ── Extended Methods ─────────────────────────────────────────────────

  async listModels(): Promise<AIModelInfo[]> {
    // Try a live SDK list() first regardless of active mode (we can build
    // a temporary SDK adapter from cached/env config). If that fails, fall
    // back to a static recommended set so the UI still has something.
    try {
      const tmp = new OllamaSdkAdapter(() => this.resolveConnection());
      return await tmp.listModels();
    } catch {
      return [...OLLAMA_FALLBACK_MODELS];
    }
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);

    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session: ${sessionId}`);
    }
  }

  getCapabilities(): AIProviderCapabilities {
    return {
      streaming: true,
      vision: false,
      fileAttachments: true,
      // Ollama supports tool calling for many models (e.g. llama3.1+,
      // qwen2.5, mistral-nemo); expose it as a capability.
      toolUse: true,
      mcpSupport: false,
      voiceMode: false,
      cancelSupport: true,
      modelListing: true,
    };
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.getSession(sessionId);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  // ── Standard Methods ─────────────────────────────────────────────────

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.getSession(sessionId);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) {
        session.abortController.abort();
        session.abortController = null;
      }
      session.status = "terminated";
      session.files = [];
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => this.toAISession(s));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── Private Helpers ──────────────────────────────────────────────────

  private getSession(sessionId: string): ManagedSession {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");
    return session;
  }

  private buildFullPrompt(
    prompt: string,
    context?: AIContext[],
    files?: AIFileAttachment[],
  ): string {
    let fullPrompt = prompt;

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      fullPrompt = `${prompt}\n\n${contextStr}`;
    }

    if (files && files.length > 0) {
      const fileStr = files
        .map((f) => {
          const textContent =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}, ${f.size} bytes) ---\n${textContent}`;
        })
        .join("\n\n");
      fullPrompt = `${fullPrompt}\n\n${fileStr}`;
    }

    return fullPrompt;
  }

  private updateSessionStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
  ): void {
    const model = session.config.model || DEFAULT_MODEL;

    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;
    // Ollama is free for self-hosted; cloud pricing varies and is not
    // exposed via API. Leave estimatedCostUsd at 0 unless caller wires
    // pricing in via providerConfig in the future.

    if (!session.stats.modelBreakdown) {
      session.stats.modelBreakdown = {};
    }
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;

    session.status = "active";
    session.updatedAt = new Date().toISOString();
  }

  private toAISession(s: ManagedSession): AISession {
    return {
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    };
  }

  private log(level: "info" | "error" | "debug", msg: string): void {
    this.hostServices?.logger?.[level]?.(`${PROVIDER_NAME}-provider`, msg);
  }
}

// Re-export to silence "declared but never read" lint on the constant
// (used in jsdoc + future cost wiring).
void DEFAULT_MAX_TOKENS;

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: "binary" as const,
                requiresSudo: false,
                description: `${DISPLAY_NAME} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }

      const proc = Bun.spawnSync(CLI_INSTALL_COMMAND, {
        timeout: 300_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [
          {
            name: CLI_COMMAND,
            message:
              proc.stderr.toString().trim() ||
              `Run manually: ${CLI_INSTALL_COMMAND.join(" ")}`,
          },
        ],
      };
    });
}

const provider = new OllamaProvider();

export const vibePlugin: VibePlugin = {
  name: "ollama",
  version: "1.0.0",
  description:
    "Ollama AI agent provider for VibeControls (dual-mode: SDK + CLI, Cloud + self-hosted)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "binary",
      requiresSudo: false,
      description: `${DISPLAY_NAME} CLI for CLI mode`,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),

  onServerStart(_app, hostServices) {
    if (hostServices) provider.setHostServices(hostServices);
  },

  onServerStop() {
    for (const [id] of (provider as OllamaProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
};

export default vibePlugin;
