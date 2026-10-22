import type { CharacterView } from "./characters.contract.js";
import { isReviewDue as isDue } from "./review-due.js";
import { matchesSearch, normalizeSearchText } from "./search-match.js";
import { createSortSelection, type SortSelectionState } from "./sort-selection.js";

export const CHARACTER_SORTS = [
  "attention",
  "encounters",
  "contexts",
  "checks",
  "clean",
  "mastery",
  "recent",
  "alphabetical",
] as const;

export type CharacterSort = (typeof CHARACTER_SORTS)[number];

export type CharacterSortState = SortSelectionState<CharacterSort>;

export interface CharacterSummary {
  unique_characters: number;
  appearances: number;
  lesson_exposures: number;
  raw_reveal_count: number;
  counted_reveal_sessions: number;
  qualified_exposures: number;
  inferred_failure_sessions: number;
  inferred_failure_mass: number;
}

const characterSortSelectionMachine = createSortSelection<CharacterSort>(
  CHARACTER_SORTS,
  "attention",
  ["alphabetical", "mastery"],
  ["alphabetical", "mastery"],
);

export const DEFAULT_CHARACTER_SORT: CharacterSortState =
  characterSortSelectionMachine.defaultState;
export const nextCharacterSort = characterSortSelectionMachine.next;
export const parseCharacterSortSelection = characterSortSelectionMachine.parse;
export const characterSortSelection = characterSortSelectionMachine.selection;
export const reverseCharacterSortResult = characterSortSelectionMachine.reverseResult;

export function selectCharacters(
  characters: readonly CharacterView[],
  query: string,
  sort: CharacterSort,
  now = new Date(),
): CharacterView[] {
  const needle = normalizeSearchText(query.trim());
  const selected = needle
    ? characters.filter((entry) => matchesSearch(entry.character, needle))
    : [...characters];
  return selected.sort((left, right) => compareCharacters(left, right, sort, now));
}

export function summarizeCharacters(characters: readonly CharacterView[]): CharacterSummary {
  return characters.reduce<CharacterSummary>(
    (summary, entry) => ({
      unique_characters: summary.unique_characters + 1,
      appearances: summary.appearances + entry.occurrence_count,
      lesson_exposures: summary.lesson_exposures + entry.exposed_lesson_count,
      raw_reveal_count: summary.raw_reveal_count + entry.raw_reveal_count,
      counted_reveal_sessions: summary.counted_reveal_sessions + entry.counted_reveal_sessions,
      qualified_exposures: summary.qualified_exposures + entry.qualified_exposures,
      inferred_failure_sessions:
        summary.inferred_failure_sessions + entry.inferred_failure_sessions,
      inferred_failure_mass: summary.inferred_failure_mass + entry.inferred_failure_mass,
    }),
    {
      unique_characters: 0,
      appearances: 0,
      lesson_exposures: 0,
      raw_reveal_count: 0,
      counted_reveal_sessions: 0,
      qualified_exposures: 0,
      inferred_failure_sessions: 0,
      inferred_failure_mass: 0,
    },
  );
}

function compareCharacters(
  left: CharacterView,
  right: CharacterView,
  sort: CharacterSort,
  now: Date,
): number {
  let difference = 0;
  if (sort === "attention") {
    difference =
      Number(isDue(right, now)) - Number(isDue(left, now)) ||
      left.retrievability - right.retrievability ||
      right.inferred_failure_mass - left.inferred_failure_mass ||
      right.mastery_uncertainty - left.mastery_uncertainty;
  } else if (sort === "encounters") {
    difference =
      right.occurrence_count - left.occurrence_count ||
      right.exposed_lesson_count - left.exposed_lesson_count;
  } else if (sort === "contexts") {
    difference =
      right.distinct_word_contexts - left.distinct_word_contexts ||
      right.occurrence_count - left.occurrence_count;
  } else if (sort === "checks") {
    difference =
      right.raw_reveal_count - left.raw_reveal_count ||
      right.counted_reveal_sessions - left.counted_reveal_sessions;
  } else if (sort === "clean") {
    difference =
      right.qualified_exposures - left.qualified_exposures ||
      left.inferred_failure_mass - right.inferred_failure_mass;
  } else if (sort === "mastery") {
    difference =
      left.mastery - right.mastery ||
      left.retrievability - right.retrievability ||
      right.mastery_uncertainty - left.mastery_uncertainty;
  } else if (sort === "recent") {
    difference = evidenceTime(right) - evidenceTime(left);
  }
  return difference || compareCharactersAlphabetically(left.character, right.character);
}

function evidenceTime(character: CharacterView): number {
  return Date.parse(character.last_evidence_at ?? character.last_exposed_at);
}

function compareCharactersAlphabetically(left: string, right: string): number {
  return (left.codePointAt(0) ?? 0) - (right.codePointAt(0) ?? 0);
}
