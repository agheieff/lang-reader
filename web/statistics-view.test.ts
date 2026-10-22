import { describe, expect, it } from "vitest";

import { parseStatistics, type StatisticsPayload } from "./statistics.js";
import { presentStatistics, renderStatisticsView } from "./statistics-view.js";

const populated = parseStatistics({
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
    recent_wpm_sessions: 2,
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
});

interface FakeNode {
  textContent: string;
}

class FakeElement implements FakeNode {
  readonly attributes = new Map<string, string>();
  readonly style: Record<string, string> = {};
  children: FakeNode[] = [];
  className = "";
  hidden = false;
  textContent = "";
  title = "";

  append(...nodes: FakeNode[]): void {
    this.children.push(...nodes);
  }

  replaceChildren(...nodes: FakeNode[]): void {
    this.children = [...nodes];
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }
}

function statisticsDocument(): {
  documentRoot: Document;
  element: (id: string) => FakeElement;
} {
  const ids = [
    "statistics-word-total",
    "statistics-knowledge-bar",
    "statistics-knowledge-legend",
    "statistics-words-empty",
    "statistics-texts-read",
    "statistics-reading-sessions",
    "statistics-active-time",
    "statistics-recent-wpm",
    "statistics-lifetime-wpm",
    "statistics-recent-wpm-detail",
    "statistics-lifetime-wpm-detail",
    "statistics-reading-note",
    "statistics-level-category",
    "statistics-level-value",
    "statistics-level-source",
    "statistics-level-meter",
    "statistics-level-meter-fill",
    "statistics-level-status",
    "statistics-level-range",
    "statistics-level-evidence",
  ];
  const elements = new Map(ids.map((id) => [id, new FakeElement()]));
  const documentRoot = {
    getElementById: (id: string): FakeElement | null => elements.get(id) ?? null,
    createElement: (): FakeElement => new FakeElement(),
    createTextNode: (text: string): FakeNode => ({ textContent: text }),
  } as unknown as Document;
  return {
    documentRoot,
    element: (id: string): FakeElement => {
      const found = elements.get(id);
      if (!found) throw new Error(`Unknown fake element: ${id}`);
      return found;
    },
  };
}

describe("statistics view presentation", () => {
  it("builds populated vocabulary, reading, and placement copy", () => {
    const view = presentStatistics(populated);

    expect(view.words).toMatchObject({
      totalText: "10",
      empty: false,
      ariaLabel: "Low confidence 4, Developing 2, Familiar 3, Strong 1",
    });
    expect(
      view.words.bands.map((band) => [
        band.key,
        band.shareText,
        band.masteryText,
        band.segmentTitle,
      ]),
    ).toEqual([
      ["learning", "40%", "under 56%", "Low confidence: 4 (40%)"],
      ["expected", "20%", "56–63%", "Developing: 2 (20%)"],
      ["familiar", "30%", "64–84%", "Familiar: 3 (30%)"],
      ["mastered", "10%", "85% and up", "Strong: 1 (10%)"],
    ]);
    expect(view.reading).toMatchObject({
      textsRead: "7",
      sessions: "8 completed sessions · 1 reread",
      activeTime: "1h 10m",
      recentWpm: "81",
      recentWpmDetail: "Last 2 qualifying reads · preliminary",
      lifetimeWpm: "76",
      lifetimeWpmDetail: "6 qualifying reads",
    });
    expect(view.reading.note).toContain(
      "Chinese WPM uses the lesson’s segmented word units, not individual characters.",
    );
    expect(view.level).toEqual({
      category: "Approx. HSK 2",
      categoryTitle: "Approximate HSK 2, inferred from this lesson's difficulty",
      value: "0.20 / 1.00",
      source: "Estimated from reading",
      percent: 20,
      meterText: "Approx. HSK 2, 20% of the shared scale",
      status: "Rough estimate",
      range: "Likely range 0.04–0.36 · Approx. HSK 1 to Approx. HSK 4",
      evidence: "3 qualifying reads · 2 calibration reads · 20 usable probes",
    });
  });

  it("builds the existing empty-state labels without a language-specific pace note", () => {
    const state: StatisticsPayload = {
      ...populated,
      learning_language: "es-ES",
      words: {
        ...populated.words,
        total: 0,
        learning: 0,
        expected: 0,
        familiar: 0,
        mastered: 0,
      },
      reading: {
        ...populated.reading,
        texts_read: 0,
        completed_sessions: 0,
        total_active_seconds: 0,
        recent_average_wpm: null,
        recent_wpm_sessions: 0,
        lifetime_average_wpm: null,
        lifetime_wpm_sessions: 0,
      },
      level: {
        ...populated.level,
        value: 0.15,
        category: "A1",
        source: "unknown",
        status: "unstarted",
        lower: null,
        upper: null,
        lower_category: null,
        upper_category: null,
        qualified_attempts: 0,
        usable_probes: 0,
        qualified_readings: 0,
      },
    };

    const view = presentStatistics(state);

    expect(view.words.empty).toBe(true);
    expect(view.words.ariaLabel).toBe("No word knowledge evidence yet");
    expect(view.words.bands.every((band) => band.shareText === "0%")).toBe(true);
    expect(view.reading).toMatchObject({
      sessions: "No completed sessions",
      activeTime: "0s",
      recentWpm: "—",
      recentWpmDetail: "No measured reads yet",
      lifetimeWpm: "—",
      lifetimeWpmDetail: "No measured reads yet",
    });
    expect(view.reading.note).not.toContain("Chinese WPM");
    expect(view.level).toMatchObject({
      category: "A1",
      categoryTitle: "",
      percent: 15,
      status: "No qualifying reads yet",
      range: "Calibration texts will add a confidence range.",
      evidence: "0 qualifying reads · 0 calibration reads · 0 usable probes",
    });
  });

  it("writes the presentation through the existing template ID contract", () => {
    const view = statisticsDocument();

    renderStatisticsView(populated, view.documentRoot);

    expect(view.element("statistics-word-total").textContent).toBe("10");
    expect(view.element("statistics-knowledge-bar").children).toHaveLength(4);
    expect(view.element("statistics-knowledge-bar").attributes.get("aria-label")).toBe(
      "Low confidence 4, Developing 2, Familiar 3, Strong 1",
    );
    expect(view.element("statistics-knowledge-legend").children).toHaveLength(4);
    expect(view.element("statistics-words-empty").hidden).toBe(true);
    expect(view.element("statistics-reading-sessions").textContent).toBe(
      "8 completed sessions · 1 reread",
    );
    expect(view.element("statistics-level-category").textContent).toBe("Approx. HSK 2");
    expect(view.element("statistics-level-meter").attributes.get("aria-valuenow")).toBe("20");
    expect(view.element("statistics-level-meter-fill").style.width).toBe("20%");
  });
});
