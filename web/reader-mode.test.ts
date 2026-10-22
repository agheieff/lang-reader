import { describe, expect, it } from "vitest";

import {
  parseReaderLaunch,
  readerApiPath,
  readerModeRecordsEvidence,
  textActionHref,
} from "./reader-mode.js";

describe("reader launch modes", () => {
  it("keeps preview on its read-only endpoint and disables all evidence", () => {
    const launch = parseReaderLaunch("?lesson_id=7&mode=preview");

    expect(launch).toEqual({ lessonId: 7, mode: "preview" });
    expect(readerApiPath("/api/profiles/zh-hans", launch)).toBe("/api/profiles/zh-hans/texts/7");
    expect(readerModeRecordsEvidence(launch.mode)).toBe(false);
  });

  it("requests a fresh evidence-bearing state for rereads", () => {
    const launch = parseReaderLaunch("?lesson_id=7&mode=reread");

    expect(readerApiPath("/api/profiles/zh-hans", launch)).toBe(
      "/api/profiles/zh-hans/reader?lesson_id=7&fresh=true",
    );
    expect(readerModeRecordsEvidence(launch.mode)).toBe(true);
  });

  it("continues a selected text without asking for fresh progress", () => {
    const launch = parseReaderLaunch("?lesson_id=7");
    expect(readerApiPath("/api/profiles/zh-hans", launch)).toBe(
      "/api/profiles/zh-hans/reader?lesson_id=7",
    );
  });

  it("builds durable library action links and rejects invalid modes", () => {
    expect(textActionHref("zh hans", 7, "continue")).toBe("/p/zh%20hans?lesson_id=7");
    expect(textActionHref("zh hans", 7, "preview")).toBe("/p/zh%20hans?lesson_id=7&mode=preview");
    expect(textActionHref("zh-hans", 7, "reread")).not.toContain("fresh");
    expect(() => parseReaderLaunch("?lesson_id=7&mode=secret")).toThrow(/mode/);
    expect(() => parseReaderLaunch("?lesson_id=0")).toThrow(/ID/);
  });
});
