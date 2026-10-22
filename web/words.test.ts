import { describe, expect, it } from "vitest";

import type { WordView } from "./contracts.js";
import { formatDue, selectWords } from "./words.js";

function word(overrides: Partial<WordView> = {}): WordView {
  return {
    key: "zh:gongyuan:NOUN",
    lemma: "公园",
    pos: "noun",
    gloss: "park",
    pronunciation: "gōngyuán",
    frequency_rank: 700,
    surface_forms: [{ text: "公园", occurrences: 2 }],
    related_senses: [],
    occurrence_count: 2,
    exposed_lesson_count: 1,
    raw_reveal_count: 0,
    counted_reveal_sessions: 0,
    qualified_exposures: 1,
    mastery: 0.55,
    stability_days: 0.8,
    reveal_failures: 0,
    first_exposed_at: "2026-07-10T12:00:00Z",
    last_exposed_at: "2026-07-10T12:00:00Z",
    ...overrides,
  };
}

describe("word-list selection", () => {
  it("searches definitions, pronunciation, and observed surface forms", () => {
    const went = word({
      key: "es:ir:VERB",
      lemma: "ir",
      pos: "verb",
      gloss: "to go",
      pronunciation: undefined,
      surface_forms: [{ text: "fui", occurrences: 1 }],
      occurrence_count: 1,
    });

    expect(selectWords([word(), went], "fui", "alphabetical")).toEqual([went]);
    expect(selectWords([word(), went], "PARK", "alphabetical")[0]?.lemma).toBe("公园");
  });

  it("matches unmarked and unspaced pinyin across tone and separator variants", () => {
    const words = [
      word({ key: "one", lemma: "重新", pronunciation: "chóngxīn" }),
      word({ key: "two", lemma: "一次", pronunciation: "yí cì" }),
      word({ key: "three", lemma: "绿色", pronunciation: "lǜsè" }),
      word({ key: "time", lemma: "时", pronunciation: "shí" }),
      word({ key: "be", lemma: "是", pronunciation: "shì" }),
    ];

    expect(selectWords(words, "chongxin", "alphabetical").map((item) => item.key)).toEqual(["one"]);
    expect(selectWords(words, "yici", "alphabetical").map((item) => item.key)).toEqual(["two"]);
    expect(selectWords(words, "luse", "alphabetical").map((item) => item.key)).toEqual(["three"]);
    expect(selectWords(words, "shi", "alphabetical").map((item) => item.key)).toEqual([
      "time",
      "be",
    ]);
  });

  it("searches word classes", () => {
    const noun = word({ key: "noun", pos: "noun" });
    const preposition = word({ key: "preposition", pos: "preposition" });

    expect(selectWords([noun, preposition], "preposition", "alphabetical")).toEqual([preposition]);
  });

  it("puts due repeated checks ahead of untouched words", () => {
    const now = new Date("2026-07-16T12:00:00Z");
    const due = word({
      key: "due",
      lemma: "观察",
      reveal_failures: 2,
      counted_reveal_sessions: 2,
      raw_reveal_count: 3,
      mastery: 0.41,
      next_due_at: "2026-07-15T12:00:00Z",
    });

    expect(selectWords([word(), due], "", "attention", now)[0]).toBe(due);
    expect(selectWords([word(), due], "", "clicks", now)[0]).toBe(due);
  });

  it("sorts meanings alphabetically and clean reads from most to least", () => {
    const park = word({ key: "park", gloss: "park", qualified_exposures: 1 });
    const garden = word({ key: "garden", gloss: "garden", qualified_exposures: 4 });

    expect(selectWords([park, garden], "", "meaning").map((item) => item.key)).toEqual([
      "garden",
      "park",
    ]);
    expect(selectWords([park, garden], "", "clean").map((item) => item.key)).toEqual([
      "garden",
      "park",
    ]);
  });
});

describe("due labels", () => {
  const now = new Date("2026-07-16T12:00:00Z");

  it("formats unscheduled, overdue, and future reviews", () => {
    expect(formatDue(undefined, now)).toBe("Not scheduled");
    expect(formatDue("2026-07-16T11:00:00Z", now)).toBe("Due now");
    expect(formatDue("2026-07-18T12:00:00Z", now)).toBe("Due in 2 days");
  });
});
