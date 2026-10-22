import { describe, expect, it } from "vitest";

import {
  CONSTRUCTION_STATUS_DETAILS,
  constructionDifficultySignals,
  constructionStatus,
  DEFAULT_GRAMMAR_SORT,
  formatGrammarDue,
  formatGrammarStatus,
  type GrammarConstruction,
  parseGrammar,
  selectGrammar,
} from "./grammar.js";

const rawConstruction = {
  key: "es:si-imperfect-subjunctive",
  label: "Si + imperfect subjunctive",
  description: "Frames an unlikely condition and its imagined result.",
  category: "conditional clauses",
  difficulty: 0.62,
  occurrence_count: 6,
  exposed_lesson_count: 2,
  raw_help_count: 2,
  counted_help_sessions: 1,
  inferred_difficulty_signals: 0,
  qualified_exposures: 2,
  mastery: 0.55,
  stability_days: 3.5,
  first_exposed_at: "2026-07-10T12:00:00Z",
  last_exposed_at: "2026-07-15T12:00:00Z",
  last_helped_at: "2026-07-10T12:03:00Z",
  next_due_at: null,
  examples: [
    {
      lesson_id: 4,
      title: "Un viaje imaginario",
      sentence_key: "s4",
      text: "Si tuviera tiempo, viajaría más.",
      translation: "If I had time, I would travel more.",
      note: "A hypothetical present situation.",
    },
  ],
};

function construction(overrides: Partial<GrammarConstruction> = {}): GrammarConstruction {
  return {
    ...rawConstruction,
    next_due_at: undefined,
    examples: rawConstruction.examples,
    ...overrides,
  };
}

describe("parseGrammar", () => {
  it("parses evidence, optional timestamps, and contextual examples", () => {
    const parsed = parseGrammar({
      learning_language: "es-ES",
      translation_language: "en",
      constructions: [rawConstruction],
    });

    expect(parsed).toMatchObject({
      learning_language: "es-ES",
      translation_language: "en",
      constructions: [
        {
          key: "es:si-imperfect-subjunctive",
          counted_help_sessions: 1,
          last_helped_at: "2026-07-10T12:03:00Z",
          next_due_at: undefined,
          examples: [
            {
              lesson_id: 4,
              sentence_key: "s4",
              translation: "If I had time, I would travel more.",
            },
          ],
        },
      ],
    });
  });

  it("selects nothing from an empty construction list", () => {
    expect(selectGrammar([], "", DEFAULT_GRAMMAR_SORT)).toEqual([]);
  });

  it("accepts an empty state without inventing constructions", () => {
    expect(
      parseGrammar({
        learning_language: "zh-Hans",
        translation_language: "en",
        constructions: [],
      }).constructions,
    ).toEqual([]);
  });

  it("rejects malformed roots, lists, and duplicate construction keys", () => {
    expect(() => parseGrammar(null)).toThrow(/response must be an object/);
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: {},
      }),
    ).toThrow(/constructions must be a list/);
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [rawConstruction, rawConstruction],
      }),
    ).toThrow(/keys must be unique/);
  });

  it.each([
    ["difficulty", -0.01, /difficulty must be between 0 and 1/],
    ["mastery", 1.01, /mastery must be between 0 and 1/],
    ["stability_days", -1, /stability_days must be a non-negative number/],
    ["qualified_exposures", 0.5, /qualified_exposures must be an integer/],
    ["raw_help_count", -1, /raw_help_count must be an integer/],
  ])("rejects invalid %s values", (field, value, message) => {
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [{ ...rawConstruction, [field]: value }],
      }),
    ).toThrow(message);
  });

  it("checks count relationships without treating repeated raw taps as sessions", () => {
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [{ ...rawConstruction, occurrence_count: 1, exposed_lesson_count: 2 }],
      }),
    ).toThrow(/fewer instances than lessons/);
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [{ ...rawConstruction, raw_help_count: 1, counted_help_sessions: 2 }],
      }),
    ).toThrow(/counted help exceeds raw help/);
  });

  it("rejects invalid dates and malformed examples", () => {
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [{ ...rawConstruction, last_exposed_at: "recently" }],
      }),
    ).toThrow(/last_exposed_at must be a timestamp/);
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [{ ...rawConstruction, examples: "not a list" }],
      }),
    ).toThrow(/examples must be a list/);
    expect(() =>
      parseGrammar({
        learning_language: "es-ES",
        translation_language: "en",
        constructions: [
          {
            ...rawConstruction,
            examples: [{ ...rawConstruction.examples[0], lesson_id: 0 }],
          },
        ],
      }),
    ).toThrow(/lesson_id must be an integer of at least 1/);
  });
});

describe("construction status", () => {
  it("uses deduplicated and inferred signals, never repeated raw help taps", () => {
    const repeatedTap = construction({
      raw_help_count: 12,
      counted_help_sessions: 1,
      inferred_difficulty_signals: 0,
      qualified_exposures: 0,
      mastery: 0.1,
    });

    expect(constructionDifficultySignals(repeatedTap)).toBe(1);
    expect(constructionStatus(repeatedTap)).toBe("seen");
  });

  it("requires repeated signals, a high signal share, and low mastery for attention", () => {
    expect(
      constructionStatus(
        construction({
          raw_help_count: 2,
          counted_help_sessions: 2,
          inferred_difficulty_signals: 1,
          qualified_exposures: 1,
          mastery: 0.35,
        }),
      ),
    ).toBe("needs_attention");
    expect(
      constructionStatus(
        construction({
          raw_help_count: 2,
          counted_help_sessions: 2,
          inferred_difficulty_signals: 0,
          qualified_exposures: 5,
          mastery: 0.35,
        }),
      ),
    ).toBe("developing");
    expect(
      constructionStatus(
        construction({
          raw_help_count: 2,
          counted_help_sessions: 2,
          inferred_difficulty_signals: 0,
          qualified_exposures: 1,
          mastery: 0.7,
        }),
      ),
    ).toBe("developing");
  });

  it("reserves comfortable for stable, broad, mostly clean evidence", () => {
    const comfortable = construction({
      occurrence_count: 10,
      exposed_lesson_count: 3,
      raw_help_count: 1,
      counted_help_sessions: 1,
      qualified_exposures: 5,
      mastery: 0.82,
      stability_days: 8,
    });

    expect(constructionStatus(comfortable)).toBe("comfortable");
    expect(constructionStatus({ ...comfortable, exposed_lesson_count: 2 })).toBe("developing");
    expect(constructionStatus({ ...comfortable, stability_days: 6.9 })).toBe("developing");
    expect(constructionStatus({ ...comfortable, mastery: 0.79 })).toBe("developing");
  });

  it("uses deliberately non-absolute user-facing labels", () => {
    expect(CONSTRUCTION_STATUS_DETAILS.comfortable.label).toBe("Comfortable");
    expect(Object.values(CONSTRUCTION_STATUS_DETAILS).map((detail) => detail.label)).not.toContain(
      "Mastered",
    );
    expect(formatGrammarStatus(construction({ qualified_exposures: 0 }))).toBe("Seen");
  });
});

describe("grammar due labels", () => {
  const now = new Date("2026-07-16T12:00:00Z");

  it("formats unscheduled, overdue, and future review times", () => {
    expect(formatGrammarDue(undefined, now)).toBe("Not scheduled");
    expect(formatGrammarDue("2026-07-16T11:00:00Z", now)).toBe("Due now");
    expect(formatGrammarDue("2026-07-17T11:00:00Z", now)).toBe("Due within a day");
    expect(formatGrammarDue("2026-07-18T12:00:00Z", now)).toBe("Due in 2 days");
  });
});

describe("grammar selection", () => {
  it("searches labels, descriptions, categories, examples, translations, and notes", () => {
    const conditional = construction({
      key: "conditional",
      label: "Condición irreal",
      category: "Cláusulas condicionales",
      description: "Combina si con el imperfecto de subjuntivo.",
      examples: [
        {
          lesson_id: 4,
          title: "Un día distinto",
          sentence_key: "s1",
          text: "Si pudiera, iría al mar.",
          translation: "If I could, I would go to the sea.",
          note: "Used for an imagined result.",
        },
      ],
    });
    const perfect = construction({
      key: "perfect",
      label: "Pretérito perfecto",
      description: "haber + participio",
      category: "past narration",
    });

    for (const query of [
      "condicion",
      "subjuntivo",
      "clausulas",
      "pudiera",
      "go to the sea",
      "imagined result",
    ]) {
      expect(selectGrammar([perfect, conditional], query, "alphabetical")).toEqual([conditional]);
    }
    expect(selectGrammar([conditional, perfect], "haberparticipio", "alphabetical")).toEqual([
      perfect,
    ]);
  });

  it("sorts weak attention first, while due reviews take precedence", () => {
    const now = new Date("2026-07-16T12:00:00Z");
    const needsAttention = construction({
      key: "attention",
      label: "Attention",
      raw_help_count: 3,
      counted_help_sessions: 2,
      inferred_difficulty_signals: 1,
      qualified_exposures: 1,
      mastery: 0.3,
    });
    const seen = construction({
      key: "seen",
      label: "Seen",
      raw_help_count: 0,
      counted_help_sessions: 0,
      qualified_exposures: 1,
    });
    const due = construction({
      key: "due",
      label: "Due",
      next_due_at: "2026-07-15T12:00:00Z",
    });

    expect(
      selectGrammar([seen, needsAttention], "", "attention", now).map((item) => item.key),
    ).toEqual(["attention", "seen"]);
    expect(selectGrammar([seen, needsAttention, due], "", "attention", now)[0]).toBe(due);
  });

  it("does not let repeated raw taps change attention order", () => {
    const first = construction({
      key: "a",
      label: "A",
      raw_help_count: 20,
      counted_help_sessions: 1,
    });
    const second = construction({
      key: "b",
      label: "B",
      raw_help_count: 1,
      counted_help_sessions: 1,
    });

    expect(selectGrammar([second, first], "", "attention")).toEqual([first, second]);
  });

  it("sorts encounters, clean evidence, low mastery, category, and label deterministically", () => {
    const clauses = construction({
      key: "clauses",
      label: "Z clauses",
      category: "clauses",
      occurrence_count: 9,
      exposed_lesson_count: 3,
      qualified_exposures: 5,
      mastery: 0.8,
    });
    const aspect = construction({
      key: "aspect",
      label: "A aspect",
      category: "aspect",
      occurrence_count: 3,
      exposed_lesson_count: 1,
      qualified_exposures: 1,
      mastery: 0.2,
    });

    expect(selectGrammar([aspect, clauses], "", "encounters")[0]).toBe(clauses);
    expect(selectGrammar([aspect, clauses], "", "clean")[0]).toBe(clauses);
    expect(selectGrammar([aspect, clauses], "", "mastery")[0]).toBe(aspect);
    expect(selectGrammar([clauses, aspect], "", "category")[0]).toBe(aspect);
    expect(selectGrammar([clauses, aspect], "", "alphabetical")[0]).toBe(aspect);
  });

  it("does not mutate the caller's array while filtering or sorting", () => {
    const first = construction({ key: "z", label: "Z" });
    const second = construction({ key: "a", label: "A" });
    const input = [first, second];

    expect(selectGrammar(input, "", "alphabetical")).toEqual([second, first]);
    expect(input).toEqual([first, second]);
  });
});
