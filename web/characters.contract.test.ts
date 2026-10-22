import { describe, expect, it } from "vitest";

import { parseCharacters } from "./characters.contract.js";

function response(character = "湖") {
  return {
    learning_language: "zh-Hans",
    translation_language: "en",
    characters: [
      {
        character,
        occurrence_count: 4,
        exposed_lesson_count: 2,
        raw_reveal_count: 3,
        counted_reveal_sessions: 2,
        distinct_word_contexts: 3,
        qualified_exposures: 2,
        inferred_failure_sessions: 1,
        inferred_failure_mass: 0.25,
        direct_successes: 0,
        direct_failures: 0,
        mastery: 0.68,
        mastery_uncertainty: 0.12,
        retrievability: 0.61,
        stability_days: 4,
        first_exposed_at: "2026-07-10T12:00:00Z",
        last_exposed_at: "2026-07-12T12:00:00Z",
        last_evidence_at: "2026-07-12T12:00:00Z",
        next_due_at: "2026-07-16T12:00:00Z",
      },
    ],
  };
}

describe("parseCharacters", () => {
  it("accepts one Han Unicode scalar, including supplementary-plane characters", () => {
    expect(parseCharacters(response()).characters[0]?.character).toBe("湖");
    expect(parseCharacters(response("𠀀")).characters[0]?.character).toBe("𠀀");
    expect(parseCharacters(response("〇")).characters[0]?.character).toBe("〇");
  });

  it("rejects non-Han text and duplicate characters", () => {
    expect(() => parseCharacters(response("A"))).toThrow(/one Han character/);
    expect(() => parseCharacters(response("学习"))).toThrow(/one Han character/);
    const duplicate = response();
    duplicate.characters.push({ ...duplicate.characters[0] });
    expect(() => parseCharacters(duplicate)).toThrow(/unique/);
  });

  it("rejects impossible counters and exposure chronology", () => {
    const tooManyLessons = response();
    tooManyLessons.characters[0] = {
      ...tooManyLessons.characters[0],
      occurrence_count: 1,
      exposed_lesson_count: 2,
    };
    expect(() => parseCharacters(tooManyLessons)).toThrow(/more lessons/);

    const tooManyCountedChecks = response();
    tooManyCountedChecks.characters[0] = {
      ...tooManyCountedChecks.characters[0],
      raw_reveal_count: 1,
      counted_reveal_sessions: 2,
    };
    expect(() => parseCharacters(tooManyCountedChecks)).toThrow(/more counted checks/);

    const backwards = response();
    backwards.characters[0] = {
      ...backwards.characters[0],
      first_exposed_at: "2026-07-13T12:00:00Z",
    };
    expect(() => parseCharacters(backwards)).toThrow(/last seen before/);

    const tooManyContexts = response();
    tooManyContexts.characters[0] = {
      ...tooManyContexts.characters[0],
      distinct_word_contexts: 5,
    };
    expect(() => parseCharacters(tooManyContexts)).toThrow(/more word contexts/);

    const tooManyDifficultySessions = response();
    tooManyDifficultySessions.characters[0] = {
      ...tooManyDifficultySessions.characters[0],
      inferred_failure_sessions: 3,
    };
    expect(() => parseCharacters(tooManyDifficultySessions)).toThrow(/more difficulty sessions/);
  });

  it("parses inferred recognition and allows characters without qualified evidence", () => {
    const parsed = parseCharacters(response()).characters[0];
    expect(parsed).toMatchObject({
      distinct_word_contexts: 3,
      qualified_exposures: 2,
      inferred_failure_mass: 0.25,
      mastery: 0.68,
      retrievability: 0.61,
    });

    const emptyEvidence = response();
    expect(
      parseCharacters({
        ...emptyEvidence,
        characters: [
          {
            ...emptyEvidence.characters[0],
            qualified_exposures: 0,
            inferred_failure_sessions: 0,
            inferred_failure_mass: 0,
            last_evidence_at: null,
            next_due_at: null,
          },
        ],
      }).characters[0],
    ).toMatchObject({
      last_evidence_at: undefined,
      next_due_at: undefined,
    });
  });

  it("rejects invalid recognition estimates", () => {
    const invalidMastery = response();
    invalidMastery.characters[0] = {
      ...invalidMastery.characters[0],
      mastery: 1.01,
    };
    expect(() => parseCharacters(invalidMastery)).toThrow(/mastery must be between 0 and 1/);

    const invalidStability = response();
    invalidStability.characters[0] = {
      ...invalidStability.characters[0],
      stability_days: -0.1,
    };
    expect(() => parseCharacters(invalidStability)).toThrow(/stability_days/);
  });

  it("allows an empty inactive-language projection", () => {
    expect(
      parseCharacters({
        learning_language: "de",
        translation_language: "en",
        characters: [],
      }).characters,
    ).toEqual([]);
  });
});
