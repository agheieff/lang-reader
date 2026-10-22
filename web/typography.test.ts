import { describe, expect, it } from "vitest";

import { splitClosingPunctuation } from "./typography.js";

describe("closing punctuation boundaries", () => {
  it("separates punctuation from following whitespace without changing text", () => {
    const text = "。” ";
    const split = splitClosingPunctuation(text);

    expect(split).toEqual({ attached: "。”", remainder: " " });
    expect(split.attached + split.remainder).toBe(text);
  });

  it("supports Latin and full-width punctuation sequences", () => {
    expect(splitClosingPunctuation(", next")).toEqual({
      attached: ",",
      remainder: " next",
    });
    expect(splitClosingPunctuation("？！…next")).toEqual({
      attached: "？！…",
      remainder: "next",
    });
  });

  it("does not bind whitespace or opening punctuation to the previous word", () => {
    expect(splitClosingPunctuation(" next")).toEqual({ attached: "", remainder: " next" });
    expect(splitClosingPunctuation("「next")).toEqual({ attached: "", remainder: "「next" });
  });
});
