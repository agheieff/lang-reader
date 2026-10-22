import { describe, expect, it } from "vitest";

import { SentenceRevealState, shouldHandleSentenceClick } from "./reveal-state.js";

describe("sentence reveal state", () => {
  it("records a sentence only on its first reveal", () => {
    const state = new SentenceRevealState();

    expect(state.reveal("sentence-1")).toBe(true);
    expect(state.reveal("sentence-1")).toBe(false);
    expect(state.has("sentence-1")).toBe(true);
  });

  it("restores prior reveals without recording them again", () => {
    const state = new SentenceRevealState(["句子-一", "sentence-2"]);

    expect(state.has("句子-一")).toBe(true);
    expect(state.reveal("句子-一")).toBe(false);
    expect(state.reveal("新-句子")).toBe(true);
  });
});

describe("sentence click intent", () => {
  it("leaves ordinary word clicks to the word gloss", () => {
    expect(shouldHandleSentenceClick(false, true, false)).toBe(false);
    expect(shouldHandleSentenceClick(false, false, false)).toBe(true);
    expect(shouldHandleSentenceClick(false, false, true)).toBe(false);
  });

  it("claims a word click only while sentence selection mode is active", () => {
    expect(shouldHandleSentenceClick(true, true, false)).toBe(true);
    expect(shouldHandleSentenceClick(true, false, false)).toBe(true);
    expect(shouldHandleSentenceClick(true, false, true)).toBe(false);
  });

  it("never opens sentence help from a plain tap on touch", () => {
    expect(shouldHandleSentenceClick(false, false, false, true)).toBe(false);
    expect(shouldHandleSentenceClick(false, true, false, true)).toBe(false);
    expect(shouldHandleSentenceClick(true, false, false, true)).toBe(true);
  });
});
