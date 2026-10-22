import { describe, expect, it } from "vitest";

import { matchesSearch, normalizeSearchText } from "./search-match.js";

describe("search matching", () => {
  it("normalizes case, tone marks, umlaut input, and spacing", () => {
    const needle = normalizeSearchText("LU: SE");

    expect(matchesSearch("lǜsè", needle)).toBe(true);
  });

  it("makes symbol compaction an explicit caller policy", () => {
    const needle = normalizeSearchText("haberparticipio");

    expect(matchesSearch("haber + participio", needle)).toBe(false);
    expect(matchesSearch("haber + participio", needle, { compactSymbols: true })).toBe(true);
  });
});
