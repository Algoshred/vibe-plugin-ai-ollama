import { describe, expect, it } from "bun:test";

const { permissionFlags } = await import("../index.js");

describe("ollama permissionFlags", () => {
  // ollama `run` is a text generator with no tool execution — every mode is a no-op.
  it("every mode → no flag (no autonomy concept)", () => {
    expect(permissionFlags("plan")).toEqual([]);
    expect(permissionFlags("acceptEdits")).toEqual([]);
    expect(permissionFlags("fullAuto")).toEqual([]);
    expect(permissionFlags(undefined)).toEqual([]);
    expect(permissionFlags("bogus" as never)).toEqual([]);
  });
});
