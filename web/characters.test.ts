import { describe, expect, it } from "vitest";

import type { CharacterView } from "./characters.contract.js";
import {
  characterSortSelection,
  DEFAULT_CHARACTER_SORT,
  nextCharacterSort,
  parseCharacterSortSelection,
  reverseCharacterSortResult,
  selectCharacters,
  summarizeCharacters,
} from "./characters.js";

function character(character: string, overrides: Partial<CharacterView> = {}): CharacterView {
  return {
    character,
    occurrence_count: 1,
    exposed_lesson_count: 1,
    raw_reveal_count: 0,
    counted_reveal_sessions: 0,
    distinct_word_contexts: 1,
    qualified_exposures: 0,
    inferred_failure_sessions: 0,
    inferred_failure_mass: 0,
    direct_successes: 0,
    direct_failures: 0,
    mastery: 0.5,
    mastery_uncertainty: 0.22,
    retrievability: 0,
    stability_days: 0.5,
    first_exposed_at: "2026-07-10T12:00:00Z",
    last_exposed_at: "2026-07-10T12:00:00Z",
    ...overrides,
  };
}

describe("character-list selection", () => {
  const lake = character("湖", {
    occurrence_count: 8,
    exposed_lesson_count: 3,
    raw_reveal_count: 4,
    counted_reveal_sessions: 2,
    distinct_word_contexts: 4,
    qualified_exposures: 3,
    inferred_failure_sessions: 2,
    inferred_failure_mass: 0.4,
    mastery: 0.72,
    mastery_uncertainty: 0.1,
    retrievability: 0.6,
    last_exposed_at: "2026-07-12T12:00:00Z",
    last_evidence_at: "2026-07-12T12:00:00Z",
    next_due_at: "2026-07-12T12:00:00Z",
  });
  const side = character("边", {
    occurrence_count: 3,
    exposed_lesson_count: 2,
    raw_reveal_count: 1,
    counted_reveal_sessions: 1,
    distinct_word_contexts: 2,
    qualified_exposures: 4,
    inferred_failure_sessions: 1,
    inferred_failure_mass: 0.1,
    mastery: 0.4,
    mastery_uncertainty: 0.2,
    retrievability: 0.25,
    last_exposed_at: "2026-07-14T12:00:00Z",
    last_evidence_at: "2026-07-14T12:00:00Z",
    next_due_at: "2026-07-20T12:00:00Z",
  });
  const temperature = character("温", {
    occurrence_count: 3,
    exposed_lesson_count: 3,
    raw_reveal_count: 2,
    counted_reveal_sessions: 2,
    distinct_word_contexts: 3,
    qualified_exposures: 2,
    inferred_failure_sessions: 2,
    inferred_failure_mass: 0.3,
    mastery: 0.55,
    mastery_uncertainty: 0.15,
    retrievability: 0.45,
    last_exposed_at: "2026-07-11T12:00:00Z",
    last_evidence_at: "2026-07-11T12:00:00Z",
    next_due_at: "2026-07-10T12:00:00Z",
  });

  it("searches the literal character", () => {
    expect(selectCharacters([lake, side], " 湖 ", "alphabetical")).toEqual([lake]);
    expect(selectCharacters([lake, side], "海", "alphabetical")).toEqual([]);
  });

  it("sorts every metric with deterministic character tie breaks", () => {
    const entries = [side, lake, temperature];
    const now = new Date("2026-07-15T12:00:00Z");
    expect(selectCharacters(entries, "", "attention", now)).toEqual([temperature, lake, side]);
    expect(selectCharacters(entries, "", "encounters", now)).toEqual([lake, temperature, side]);
    expect(selectCharacters(entries, "", "contexts", now)).toEqual([lake, temperature, side]);
    expect(selectCharacters(entries, "", "checks")).toEqual([lake, temperature, side]);
    expect(selectCharacters(entries, "", "clean")).toEqual([side, lake, temperature]);
    expect(selectCharacters(entries, "", "mastery")).toEqual([side, temperature, lake]);
    expect(selectCharacters(entries, "", "recent")).toEqual([side, lake, temperature]);
    expect(selectCharacters(entries, "", "alphabetical")).toEqual([temperature, lake, side]);
  });

  it("summarizes observable and reducer evidence without averaging recognition", () => {
    expect(summarizeCharacters([lake, side])).toEqual({
      unique_characters: 2,
      appearances: 11,
      lesson_exposures: 5,
      raw_reveal_count: 5,
      counted_reveal_sessions: 3,
      qualified_exposures: 7,
      inferred_failure_sessions: 3,
      inferred_failure_mass: 0.5,
    });
  });
});

describe("character table sort controls", () => {
  it("defaults text ascending, metrics descending, and toggles the active sort", () => {
    expect(nextCharacterSort(DEFAULT_CHARACTER_SORT, "alphabetical")).toEqual({
      sort: "alphabetical",
      direction: "ascending",
    });
    const ascending = nextCharacterSort(DEFAULT_CHARACTER_SORT, "alphabetical");
    expect(nextCharacterSort(ascending, "alphabetical")).toEqual({
      sort: "alphabetical",
      direction: "descending",
    });
    expect(reverseCharacterSortResult(ascending)).toBe(false);
    expect(nextCharacterSort(DEFAULT_CHARACTER_SORT, "mastery")).toEqual({
      sort: "mastery",
      direction: "ascending",
    });
  });

  it("round trips valid selections and rejects invalid ones", () => {
    const state = { sort: "recent", direction: "ascending" } as const;
    expect(parseCharacterSortSelection(characterSortSelection(state))).toEqual(state);
    expect(parseCharacterSortSelection("unknown:ascending")).toEqual(DEFAULT_CHARACTER_SORT);
  });
});
