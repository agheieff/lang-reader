import { describe, expect, it } from "vitest";

import {
  DEFAULT_WORD_SORT,
  nextWordSort,
  parseWordSortSelection,
  reverseWordSortResult,
  wordSortSelection,
} from "./word-sort-ui.js";

describe("word table sort controls", () => {
  it("starts text columns ascending and numeric columns descending", () => {
    expect(nextWordSort(DEFAULT_WORD_SORT, "alphabetical")).toEqual({
      sort: "alphabetical",
      direction: "ascending",
    });
    expect(nextWordSort(DEFAULT_WORD_SORT, "meaning")).toEqual({
      sort: "meaning",
      direction: "ascending",
    });
    expect(nextWordSort(DEFAULT_WORD_SORT, "clicks")).toEqual({
      sort: "clicks",
      direction: "descending",
    });
    expect(nextWordSort(DEFAULT_WORD_SORT, "clean")).toEqual({
      sort: "clean",
      direction: "descending",
    });
    expect(nextWordSort(DEFAULT_WORD_SORT, "mastery")).toEqual({
      sort: "mastery",
      direction: "descending",
    });
  });

  it("toggles the current column and reports when the default order must reverse", () => {
    const ascending = nextWordSort(DEFAULT_WORD_SORT, "alphabetical");
    const descending = nextWordSort(ascending, "alphabetical");

    expect(descending).toEqual({ sort: "alphabetical", direction: "descending" });
    expect(reverseWordSortResult(ascending)).toBe(false);
    expect(reverseWordSortResult(descending)).toBe(true);
    expect(reverseWordSortResult(nextWordSort(DEFAULT_WORD_SORT, "mastery"))).toBe(true);
  });

  it("round trips dropdown values and rejects invalid values", () => {
    const state = { sort: "encounters", direction: "ascending" } as const;
    expect(parseWordSortSelection(wordSortSelection(state))).toEqual(state);
    expect(parseWordSortSelection("not-a-sort:ascending")).toEqual(DEFAULT_WORD_SORT);
  });
});
