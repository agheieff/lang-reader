import { describe, expect, it } from "vitest";

import { hskBandForDifficulty, readerLevelPresentation } from "./reader-level.js";

describe("HSK difficulty bands", () => {
  it("uses nine equal bands with explicit endpoint behavior", () => {
    expect(hskBandForDifficulty(0)).toBe(1);
    expect(hskBandForDifficulty(1 / 9 - Number.EPSILON)).toBe(1);
    expect(hskBandForDifficulty(1 / 9)).toBe(2);
    expect(hskBandForDifficulty(8 / 9 - Number.EPSILON)).toBe(8);
    expect(hskBandForDifficulty(8 / 9)).toBe(9);
    expect(hskBandForDifficulty(1)).toBe(9);
  });

  it("rejects values outside the shared difficulty scale", () => {
    expect(() => hskBandForDifficulty(-0.01)).toThrow(RangeError);
    expect(() => hskBandForDifficulty(1.01)).toThrow(RangeError);
    expect(() => hskBandForDifficulty(Number.NaN)).toThrow(RangeError);
  });
});

describe("reader level presentation", () => {
  it("shows an explicitly approximate HSK band for Chinese variants", () => {
    expect(readerLevelPresentation("zh-Hans", "A2", 0.19)).toEqual({
      label: "Approx. HSK 2",
      title: "Approximate HSK 2, inferred from this lesson's difficulty",
    });
    expect(readerLevelPresentation("ZH-hANT", "B1", 0.57).label).toBe("Approx. HSK 6");
    expect(readerLevelPresentation("zh", "C2", 1).label).toBe("Approx. HSK 9");
  });

  it("keeps the supplied CEFR label for non-Chinese languages", () => {
    expect(readerLevelPresentation("es-ES", "A2", 0.19)).toEqual({ label: "A2" });
    expect(readerLevelPresentation("ja", "N4", 0.19)).toEqual({ label: "N4" });
  });

  it("does not mistake longer language identifiers for Chinese tags", () => {
    expect(readerLevelPresentation("zho", "A2", 0.19)).toEqual({ label: "A2" });
  });
});
