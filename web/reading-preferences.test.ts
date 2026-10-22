import { describe, expect, it } from "vitest";

import { contentExperiment, parseReadingPreferences } from "./reading-preferences.js";

describe("reading preferences", () => {
  it("parses the notes view, including blank default notes", () => {
    const parsed = parseReadingPreferences({
      text: "",
      revision_id: null,
      source: "default",
      updated_at: null,
      last_agent_reason: null,
      pending_messages: [{ text: "More poems", created_at: "2026-09-28T20:00:00Z" }],
    });
    expect(parsed.source).toBe("default");
    expect(parsed.pending_messages).toHaveLength(1);
  });

  it("marks only variation and new-subject tests", () => {
    const question = "Do you enjoy maritime history?";
    expect(
      contentExperiment({ content_plan: { move: "new" }, content_hypothesis: question }),
    ).toEqual({ kind: "Testing a new subject", question });
    expect(
      contentExperiment({ content_plan: { move: "variation" }, content_hypothesis: question })
        ?.kind,
    ).toBe("Testing a variation");
    expect(
      contentExperiment({ content_plan: { move: "favourite" }, content_hypothesis: question }),
    ).toBeNull();
    expect(contentExperiment({})).toBeNull();
  });
});
