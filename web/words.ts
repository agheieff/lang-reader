import type { WordView } from "./contracts.js";
import { isReviewDue as isDue } from "./review-due.js";
import { matchesSearch, normalizeSearchText } from "./search-match.js";

export { formatReviewDue as formatDue } from "./review-due.js";

export const WORD_SORTS = [
  "attention",
  "clicks",
  "encounters",
  "clean",
  "mastery",
  "alphabetical",
  "meaning",
] as const;

export type WordSort = (typeof WORD_SORTS)[number];

export function selectWords(
  words: readonly WordView[],
  query: string,
  sort: WordSort,
  now = new Date(),
): WordView[] {
  const needle = normalizeSearchText(query.trim());
  const selected = needle
    ? words.filter((word) => searchableValues(word).some((value) => matchesSearch(value, needle)))
    : [...words];
  return selected.sort((left, right) => compareWords(left, right, sort, now));
}

function searchableValues(word: WordView): readonly string[] {
  return [
    word.lemma,
    word.pos,
    word.gloss,
    word.pronunciation ?? "",
    ...word.surface_forms.map((surface) => surface.text),
  ];
}

function compareWords(left: WordView, right: WordView, sort: WordSort, now: Date): number {
  let difference = 0;
  if (sort === "attention") {
    difference =
      Number(isDue(right, now)) - Number(isDue(left, now)) ||
      right.reveal_failures - left.reveal_failures ||
      right.counted_reveal_sessions - left.counted_reveal_sessions ||
      left.mastery - right.mastery;
  } else if (sort === "clicks") {
    difference =
      right.raw_reveal_count - left.raw_reveal_count ||
      right.counted_reveal_sessions - left.counted_reveal_sessions;
  } else if (sort === "encounters") {
    difference =
      right.occurrence_count - left.occurrence_count ||
      right.exposed_lesson_count - left.exposed_lesson_count;
  } else if (sort === "clean") {
    difference =
      right.qualified_exposures - left.qualified_exposures ||
      left.reveal_failures - right.reveal_failures;
  } else if (sort === "mastery") {
    difference = left.mastery - right.mastery || right.reveal_failures - left.reveal_failures;
  } else if (sort === "meaning") {
    difference = left.gloss.localeCompare(right.gloss) || left.pos.localeCompare(right.pos);
  }
  return (
    difference ||
    left.lemma.localeCompare(right.lemma) ||
    left.pos.localeCompare(right.pos) ||
    left.key.localeCompare(right.key)
  );
}
