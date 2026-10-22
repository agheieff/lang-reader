import { describe, expect, it } from "vitest";

import type { GrammarOccurrence } from "./contracts.js";
import { visibleGrammarOccurrences } from "./grammar-markers.js";

const occurrences: GrammarOccurrence[] = [
  {
    key: "sentence:grammar:1",
    construction_key: "zh:measure-word-phrase",
    run_start: 0,
    run_end: 2,
  },
  {
    key: "sentence:grammar:2",
    construction_key: "zh:de-modifier",
    run_start: 2,
    run_end: 4,
  },
  {
    key: "sentence:grammar:3",
    construction_key: "zh:measure-word-phrase",
    run_start: 4,
    run_end: 6,
  },
];

describe("visibleGrammarOccurrences", () => {
  it("suppresses every occurrence of a comfortable construction", () => {
    const visible = visibleGrammarOccurrences(occurrences, new Set(["zh:measure-word-phrase"]));

    expect(visible.map((occurrence) => occurrence.key)).toEqual(["sentence:grammar:2"]);
  });

  it("returns no occurrences when every construction is comfortable", () => {
    expect(
      visibleGrammarOccurrences(occurrences, new Set(["zh:measure-word-phrase", "zh:de-modifier"])),
    ).toEqual([]);
  });

  it("keeps all occurrences, in order, when no construction is comfortable", () => {
    const visible = visibleGrammarOccurrences(occurrences, new Set());

    expect(visible).toEqual(occurrences);
    expect(visible).not.toBe(occurrences);
  });
});
