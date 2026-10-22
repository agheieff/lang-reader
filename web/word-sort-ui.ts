import { createSortSelection, type SortSelectionState } from "./sort-selection.js";
import { WORD_SORTS, type WordSort } from "./words.js";

export type WordSortState = SortSelectionState<WordSort>;

// Note: header clicks start "mastery" descending while its comparator output is
// treated as ascending; both behaviors are preserved from the original module.
const wordSortSelectionMachine = createSortSelection<WordSort>(
  WORD_SORTS,
  "attention",
  ["alphabetical", "meaning"],
  ["alphabetical", "meaning", "mastery"],
);

export const DEFAULT_WORD_SORT: WordSortState = wordSortSelectionMachine.defaultState;
export const nextWordSort = wordSortSelectionMachine.next;
export const parseWordSortSelection = wordSortSelectionMachine.parse;
export const wordSortSelection = wordSortSelectionMachine.selection;
export const reverseWordSortResult = wordSortSelectionMachine.reverseResult;
