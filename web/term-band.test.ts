import { describe, expect, it } from "vitest";

import { attachTermBand, parseTermBands, TERM_BANDS } from "./term-band.js";

describe("term-band rendering", () => {
  it("attaches stable data and class tokens for every supported band", () => {
    for (const band of TERM_BANDS) {
      const dataset: { termBand?: string } = {};
      const classes: string[] = [];

      attachTermBand({ dataset, classList: { add: (token) => classes.push(token) } }, band);

      expect(dataset.termBand).toBe(band);
      expect(classes).toEqual([`term-band-${band}`]);
    }
  });

  it("leaves terms without a band unmarked", () => {
    const dataset: { termBand?: string } = {};
    const classes: string[] = [];

    attachTermBand({ dataset, classList: { add: (token) => classes.push(token) } }, undefined);

    expect(dataset.termBand).toBeUndefined();
    expect(classes).toEqual([]);
  });
});

describe("term-band contract", () => {
  it("parses every supported band from its shared runtime list", () => {
    const bands = Object.fromEntries(TERM_BANDS.map((band) => [`term-${band}`, band]));

    expect(parseTermBands(bands)).toEqual(bands);
  });

  it("keeps caller-specific contract errors", () => {
    expect(() =>
      parseTermBands(
        { term: "surprise" },
        {
          prefix: "Invalid texts",
          blankKeyRequirement: "must be non-blank text",
          unsupportedBandRequirement: "is not supported",
        },
      ),
    ).toThrow("Invalid texts: term_bands.term is not supported");
  });
});
