import { describe, expect, it } from "vitest";

import {
  calibrationLabel,
  formatActiveTime,
  formatWpm,
  knowledgeBands,
  levelSourceLabel,
  masteryRange,
  parseStatistics,
} from "./statistics.js";

const payload = {
  learning_language: "zh-Hans",
  translation_language: "en",
  words: {
    total: 10,
    learning: 4,
    expected: 2,
    familiar: 3,
    mastered: 1,
    expected_min_mastery: 0.56,
    familiar_min_mastery: 0.64,
    mastered_min_mastery: 0.85,
  },
  reading: {
    texts_read: 7,
    completed_sessions: 8,
    total_active_seconds: 4_205,
    recent_average_wpm: 81.4,
    recent_wpm_sessions: 5,
    lifetime_average_wpm: 75.8,
    lifetime_wpm_sessions: 6,
    recent_window_size: 10,
    minimum_wpm_active_seconds: 30,
    minimum_wpm_completion_ratio: 0.8,
  },
  level: {
    value: 0.195,
    category: "A2",
    source: "estimated",
    status: "rough",
    lower: 0.04,
    upper: 0.365,
    lower_category: "A1",
    upper_category: "B1",
    qualified_attempts: 2,
    usable_probes: 20,
    qualified_readings: 3,
  },
};

describe("statistics contract", () => {
  it("accepts a coherent populated response", () => {
    const parsed = parseStatistics(payload);

    expect(parsed.words.total).toBe(10);
    expect(parsed.reading.recent_average_wpm).toBe(81.4);
    expect(parsed.level.status).toBe("rough");
  });

  it("accepts the all-zero reading state", () => {
    const parsed = parseStatistics({
      ...payload,
      words: { ...payload.words, total: 0, learning: 0, expected: 0, familiar: 0, mastered: 0 },
      reading: {
        ...payload.reading,
        texts_read: 0,
        completed_sessions: 0,
        total_active_seconds: 0,
        recent_average_wpm: null,
        recent_wpm_sessions: 0,
        lifetime_average_wpm: null,
        lifetime_wpm_sessions: 0,
      },
      level: {
        ...payload.level,
        source: "unknown",
        status: "unstarted",
        lower: null,
        upper: null,
        lower_category: null,
        upper_category: null,
      },
    });

    expect(knowledgeBands(parsed.words).every((band) => band.share === 0)).toBe(true);
    expect(parsed.reading.recent_average_wpm).toBeNull();
  });

  it("rejects inconsistent distributions, cutoffs, WPM, and level ranges", () => {
    expect(() =>
      parseStatistics({
        ...payload,
        words: { ...payload.words, learning: 5 },
      }),
    ).toThrow("must sum");
    expect(() =>
      parseStatistics({
        ...payload,
        words: { ...payload.words, familiar_min_mastery: 0.5 },
      }),
    ).toThrow("strictly increasing");
    expect(() =>
      parseStatistics({
        ...payload,
        reading: { ...payload.reading, recent_average_wpm: null },
      }),
    ).toThrow("WPM values");
    expect(() =>
      parseStatistics({
        ...payload,
        level: { ...payload.level, value: 0.9 },
      }),
    ).toThrow("inside its range");
  });
});

describe("statistics presentation", () => {
  it("formats durations and nullable speeds compactly", () => {
    expect(formatActiveTime(0)).toBe("0s");
    expect(formatActiveTime(59.6)).toBe("1m");
    expect(formatActiveTime(3_600)).toBe("1h");
    expect(formatActiveTime(4_205)).toBe("1h 10m");
    expect(formatWpm(null)).toBe("—");
    expect(formatWpm(81.6)).toBe("82");
  });

  it("describes knowledge ranges and level evidence", () => {
    const bands = knowledgeBands(parseStatistics(payload).words);

    expect(bands.map((band) => [band.label, band.share])).toEqual([
      ["Low confidence", 0.4],
      ["Developing", 0.2],
      ["Familiar", 0.3],
      ["Strong", 0.1],
    ]);
    expect(bands.map(masteryRange)).toEqual(["under 56%", "56–63%", "64–84%", "85% and up"]);
    expect(calibrationLabel("unstarted")).toBe("No qualifying reads yet");
    expect(calibrationLabel("rough")).toBe("Rough estimate");
    expect(levelSourceLabel("self_reported")).toBe("Self-report prior, updated by reading");
  });
});
