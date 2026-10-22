import { describe, expect, it } from "vitest";

import type { Run } from "./contracts.js";
import { pronunciationForRun } from "./run-pronunciation.js";

function run(overrides: Partial<Run> = {}): Run {
  return {
    text: "行",
    term: {
      key: "zh:行:VERB",
      lemma: "行",
      pos: "verb",
      gloss: "to go",
      pronunciation: "háng",
    },
    ...overrides,
  };
}

describe("contextual run pronunciation", () => {
  it("prefers the encountered reading over the canonical term reading", () => {
    expect(pronunciationForRun(run({ pronunciation: "xíng" }))).toBe("xíng");
  });

  it("falls back to the canonical reading for existing lessons", () => {
    expect(pronunciationForRun(run())).toBe("háng");
    expect(pronunciationForRun({ text: "，" })).toBeUndefined();
  });
});
