import { describe, expect, it } from "vitest";

import {
  parseInterestInput,
  parseProfileActivation,
  QUESTIONNAIRE_CONFIDENCES,
} from "./activation.js";

describe("profile activation", () => {
  it("parses an inactive profile", () => {
    expect(
      parseProfileActivation({
        active: false,
        learning_language: "zh-Hans",
        activated_at: null,
        questionnaire_completed: false,
        starting_point: null,
        confidence: null,
      }).active,
    ).toBe(false);
  });

  it("normalizes optional comma-separated interests", () => {
    expect(parseInterestInput("history, science fiction, history, , cooking")).toEqual([
      "history",
      "science fiction",
      "cooking",
    ]);
  });

  it("rejects an unknown starting point", () => {
    expect(() =>
      parseProfileActivation({
        active: false,
        learning_language: "de-DE",
        activated_at: null,
        questionnaire_completed: true,
        starting_point: "expert-ish",
        confidence: "medium",
      }),
    ).toThrow(/starting point/);
  });

  it("accepts every questionnaire confidence from its runtime source of truth", () => {
    for (const confidence of QUESTIONNAIRE_CONFIDENCES) {
      expect(
        parseProfileActivation({
          active: true,
          learning_language: "de-DE",
          activated_at: "2026-07-19T10:00:00Z",
          questionnaire_completed: true,
          starting_point: "unsure",
          confidence,
        }).confidence,
      ).toBe(confidence);
    }
  });
});
