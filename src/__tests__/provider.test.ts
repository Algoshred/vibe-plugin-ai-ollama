/**
 * vibe-plugin-ai-ollama Provider Tests
 *
 * Tests for the OllamaProvider class exported via the vibePlugin.
 */
import { beforeEach, describe, expect, it, mock } from "bun:test";

// Mock the `ollama` npm package before importing the plugin
mock.module("ollama", () => {
  class MockOllama {
    constructor(_opts: { host?: string; headers?: Record<string, string> }) {
      // ignore opts
    }
    list = mock(() =>
      Promise.resolve({
        models: [
          {
            name: "llama3.2",
            modified_at: new Date().toISOString(),
            size: 1234,
          },
          {
            name: "qwen2.5",
            modified_at: new Date().toISOString(),
            size: 5678,
          },
        ],
      }),
    );
    chat = mock(
      (params: {
        stream?: boolean;
        model: string;
        messages: { role: string; content: string }[];
      }) => {
        if (params.stream) {
          // Return an async iterable simulating streamed chunks
          const stream = {
            async *[Symbol.asyncIterator]() {
              yield {
                model: params.model,
                message: { role: "assistant", content: "stream " },
                done: false,
              };
              yield {
                model: params.model,
                message: { role: "assistant", content: "text" },
                done: true,
                prompt_eval_count: 5,
                eval_count: 15,
              };
            },
            abort: () => {},
          };
          return Promise.resolve(stream);
        }
        return Promise.resolve({
          model: params.model,
          message: { role: "assistant", content: "Hello from Ollama" },
          done: true,
          prompt_eval_count: 10,
          eval_count: 20,
        });
      },
    );
  }
  return { Ollama: MockOllama, default: MockOllama };
});

const { vibePlugin } = await import("../index.js");

// Extract the provider from the plugin
const provider = vibePlugin.providers!.ai!;

describe("OllamaProvider", () => {
  const sessionConfig = {
    name: "test-session",
    agentType: "ollama",
    model: "llama3.2",
    maxTokens: 4096,
  };

  beforeEach(() => {
    // Ensure SDK mode is used + a host is configured so resolveConnection
    // returns deterministically.
    process.env["OLLAMA_API_KEY"] = "test-key-123";
    process.env["OLLAMA_HOST"] = "http://localhost:11434";
    provider.setMode!("sdk");
  });

  // ── Session Lifecycle ───────────────────────────────────────────

  describe("createSession", () => {
    it("creates a new session with generated ID", async () => {
      const session = await provider.createSession(sessionConfig);

      expect(session.id).toBeDefined();
      expect(session.name).toBe("test-session");
      expect(session.agentType).toBe("ollama");
      expect(session.provider).toBe("ollama");
      expect(session.status).toBe("active");
      expect(session.stats.inputTokens).toBe(0);
      expect(session.stats.outputTokens).toBe(0);
      expect(session.stats.requestCount).toBe(0);
      expect(session.createdAt).toBeDefined();
    });

    it("uses provided sessionId from providerConfig", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "custom-id-001" },
      });

      expect(session.id).toBe("custom-id-001");
    });

    it("returns existing session if ID already exists", async () => {
      const session1 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });
      const session2 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });

      expect(session1.id).toBe(session2.id);
      expect(session2.status).toBe("active");
    });
  });

  describe("configureSession", () => {
    it("updates session config", async () => {
      const session = await provider.createSession({ ...sessionConfig });
      await provider.configureSession(session.id, { model: "qwen2.5" });

      const sessions = await provider.listSessions();
      const updated = sessions.find((s) => s.id === session.id);
      expect(updated?.config.model).toBe("qwen2.5");
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.configureSession("does-not-exist", { model: "x" }),
      ).rejects.toThrow("not found");
    });
  });

  describe("destroySession", () => {
    it("terminates session and cleans up", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "destroy-me" },
      });

      await provider.destroySession(session.id);

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("terminated");
    });

    it("no-ops for unknown session ID", async () => {
      await provider.destroySession("nonexistent-session");
    });
  });

  describe("listSessions", () => {
    it("returns all sessions", async () => {
      const id = `list-test-${Date.now()}`;
      await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: id },
      });

      const sessions = await provider.listSessions();
      expect(sessions.length).toBeGreaterThanOrEqual(1);
      expect(sessions.some((s) => s.id === id)).toBe(true);
    });
  });

  describe("getSessionStatus", () => {
    it("returns status for existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `status-${Date.now()}` },
      });

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("active");
    });

    it("returns terminated for unknown session", async () => {
      const status = await provider.getSessionStatus("totally-unknown");
      expect(status).toBe("terminated");
    });
  });

  // ── sendPrompt ──────────────────────────────────────────────────

  describe("sendPrompt", () => {
    it("sends prompt via SDK adapter and returns response", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `prompt-${Date.now()}` },
      });

      const response = await provider.sendPrompt(session.id, "What is 2+2?");

      expect(response.content).toBe("Hello from Ollama");
      expect(response.model).toBe("llama3.2");
      expect(response.inputTokens).toBe(10);
      expect(response.outputTokens).toBe(20);
      expect(response.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("accumulates usage stats across multiple prompts", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `multi-prompt-${Date.now()}` },
      });

      await provider.sendPrompt(session.id, "First prompt");
      await provider.sendPrompt(session.id, "Second prompt");

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(20); // 10 + 10
      expect(stats.outputTokens).toBe(40); // 20 + 20
      expect(stats.requestCount).toBe(2);
    });

    it("throws for non-existent session", async () => {
      await expect(provider.sendPrompt("ghost", "Hello")).rejects.toThrow(
        "not found",
      );
    });

    it("throws for terminated session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `terminated-prompt-${Date.now()}` },
      });
      await provider.destroySession(session.id);

      await expect(provider.sendPrompt(session.id, "Hello")).rejects.toThrow(
        "terminated",
      );
    });
  });

  // ── streamPrompt ────────────────────────────────────────────────

  describe("streamPrompt", () => {
    it("streams chunks via SDK adapter", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `stream-${Date.now()}` },
      });

      const chunks: string[] = [];
      const response = await provider.streamPrompt!(
        session.id,
        "Hi",
        undefined,
        (c) => {
          if (c.type === "text") chunks.push(c.content);
        },
      );

      expect(chunks.join("")).toBe("stream text");
      expect(response.content).toBe("stream text");
      expect(response.inputTokens).toBe(5);
      expect(response.outputTokens).toBe(15);
    });
  });

  // ── getUsageStats ───────────────────────────────────────────────

  describe("getUsageStats", () => {
    it("returns zero stats for fresh session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `fresh-stats-${Date.now()}` },
      });

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(0);
      expect(stats.outputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
      expect(stats.estimatedCostUsd).toBe(0);
    });

    it("returns default stats for unknown session", async () => {
      const stats = await provider.getUsageStats("no-such-session");
      expect(stats.inputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
    });
  });

  // ── healthCheck ─────────────────────────────────────────────────

  describe("healthCheck", () => {
    it("returns ok when SDK is available", async () => {
      const result = await provider.healthCheck();
      expect(result.ok).toBe(true);
      expect(result.message).toContain("SDK");
    });
  });

  // ── getCapabilities ─────────────────────────────────────────────

  describe("getCapabilities", () => {
    it("returns expected capabilities", () => {
      provider.setMode!("sdk");
      const caps = provider.getCapabilities!();

      expect(caps.streaming).toBe(true);
      expect(caps.vision).toBe(false);
      expect(caps.fileAttachments).toBe(true);
      expect(caps.toolUse).toBe(true);
      expect(caps.mcpSupport).toBe(false);
      expect(caps.cancelSupport).toBe(true);
      expect(caps.modelListing).toBe(true);
    });
  });

  // ── getMode / setMode ───────────────────────────────────────────

  describe("getMode / setMode", () => {
    it("defaults to sdk when OLLAMA_API_KEY is set", () => {
      process.env["OLLAMA_API_KEY"] = "key";
      const mode = provider.getMode!();
      expect(mode).toBe("sdk");
    });

    it("allows explicit mode switching", () => {
      provider.setMode!("cli");
      expect(provider.getMode!()).toBe("cli");

      provider.setMode!("sdk");
      expect(provider.getMode!()).toBe("sdk");
    });
  });

  // ── listModels ──────────────────────────────────────────────────

  describe("listModels", () => {
    it("returns models reported by the SDK", async () => {
      const models = await provider.listModels!();

      expect(models.length).toBeGreaterThanOrEqual(2);
      expect(models.every((m) => m.provider === "ollama")).toBe(true);

      const llama = models.find((m) => m.id === "llama3.2");
      expect(llama).toBeDefined();
      expect(llama!.supportsStreaming).toBe(true);
    });
  });

  // ── cancelRequest ───────────────────────────────────────────────

  describe("cancelRequest", () => {
    it("throws for unknown session", async () => {
      await expect(provider.cancelRequest!("missing")).rejects.toThrow(
        "not found",
      );
    });
  });

  // ── attachFiles ─────────────────────────────────────────────────

  describe("attachFiles", () => {
    it("attaches files to an existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `files-${Date.now()}` },
      });

      await provider.attachFiles!(session.id, [
        {
          filename: "test.txt",
          mimeType: "text/plain",
          content: "hello",
          size: 5,
        },
      ]);
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.attachFiles!("none", [
          { filename: "f.txt", mimeType: "text/plain", content: "x", size: 1 },
        ]),
      ).rejects.toThrow("not found");
    });
  });
});
