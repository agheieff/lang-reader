import { byId, plural, textElement } from "./dom.js";
import { isChineseLanguage } from "./language-tag.js";
import { readerLevelPresentation } from "./reader-level.js";
import {
  calibrationLabel,
  formatActiveTime,
  formatWpm,
  knowledgeBands,
  levelSourceLabel,
  masteryRange,
  type StatisticsPayload,
} from "./statistics.js";

export interface StatisticsBandPresentation {
  key: "learning" | "expected" | "familiar" | "mastered";
  label: string;
  count: number;
  countText: string;
  shareText: string;
  masteryText: string;
  segmentTitle: string;
}

export interface StatisticsPresentation {
  words: {
    totalText: string;
    empty: boolean;
    ariaLabel: string;
    bands: StatisticsBandPresentation[];
  };
  reading: {
    textsRead: string;
    sessions: string;
    activeTime: string;
    recentWpm: string;
    recentWpmDetail: string;
    lifetimeWpm: string;
    lifetimeWpmDetail: string;
    note: string;
  };
  level: {
    category: string;
    categoryTitle: string;
    value: string;
    source: string;
    percent: number;
    meterText: string;
    status: string;
    range: string;
    evidence: string;
  };
}

export function presentStatistics(state: StatisticsPayload): StatisticsPresentation {
  const bands = knowledgeBands(state.words);
  const reading = state.reading;
  const rereads = Math.max(0, reading.completed_sessions - reading.texts_read);
  const level = state.level;
  const current = readerLevelPresentation(state.learning_language, level.category, level.value);
  const percent = Math.round(level.value * 100);

  return {
    words: {
      totalText: state.words.total.toLocaleString(),
      empty: state.words.total === 0,
      ariaLabel:
        state.words.total === 0
          ? "No word knowledge evidence yet"
          : bands.map((band) => `${band.label} ${band.count.toLocaleString()}`).join(", "),
      bands: bands.map((band) => {
        const shareText = state.words.total === 0 ? "0%" : `${Math.round(band.share * 100)}%`;
        return {
          key: band.key,
          label: band.label,
          count: band.count,
          countText: band.count.toLocaleString(),
          shareText,
          masteryText: masteryRange(band),
          segmentTitle: `${band.label}: ${band.count.toLocaleString()} (${shareText})`,
        };
      }),
    },
    reading: {
      textsRead: reading.texts_read.toLocaleString(),
      sessions:
        reading.completed_sessions === 0
          ? "No completed sessions"
          : `${reading.completed_sessions.toLocaleString()} completed ${plural(
              reading.completed_sessions,
              "session",
            )}${rereads > 0 ? ` · ${rereads.toLocaleString()} ${plural(rereads, "reread")}` : ""}`,
      activeTime: formatActiveTime(reading.total_active_seconds),
      recentWpm: formatWpm(reading.recent_average_wpm),
      recentWpmDetail:
        reading.recent_wpm_sessions === 0
          ? "No measured reads yet"
          : `Last ${reading.recent_wpm_sessions} qualifying ${plural(
              reading.recent_wpm_sessions,
              "read",
            )}${reading.recent_wpm_sessions < 3 ? " · preliminary" : ""}`,
      lifetimeWpm: formatWpm(reading.lifetime_average_wpm),
      lifetimeWpmDetail:
        reading.lifetime_wpm_sessions === 0
          ? "No measured reads yet"
          : `${reading.lifetime_wpm_sessions.toLocaleString()} qualifying ${plural(
              reading.lifetime_wpm_sessions,
              "read",
            )}`,
      note: readingNote(state),
    },
    level: {
      category: current.label,
      categoryTitle: current.title ?? "",
      value: `${level.value.toFixed(2)} / 1.00`,
      source: levelSourceLabel(level.source),
      percent,
      meterText: `${current.label}, ${percent}% of the shared scale`,
      status: calibrationLabel(level.status),
      range: levelRange(state),
      evidence: `${level.qualified_readings.toLocaleString()} qualifying ${plural(
        level.qualified_readings,
        "read",
      )} · ${level.qualified_attempts.toLocaleString()} calibration ${plural(
        level.qualified_attempts,
        "read",
      )} · ${level.usable_probes.toLocaleString()} usable ${plural(level.usable_probes, "probe")}`,
    },
  };
}

export function renderStatisticsView(
  state: StatisticsPayload,
  documentRoot: Document = document,
): void {
  const presentation = presentStatistics(state);
  renderWordStatistics(documentRoot, presentation);
  renderReadingStatistics(documentRoot, presentation);
  renderLevelStatistics(documentRoot, presentation);
}

function renderWordStatistics(documentRoot: Document, presentation: StatisticsPresentation): void {
  const words = presentation.words;
  byId("statistics-word-total", documentRoot).textContent = words.totalText;

  const bar = byId("statistics-knowledge-bar", documentRoot);
  const segments = words.bands
    .filter((band) => band.count > 0)
    .map((band) => {
      const segment = documentRoot.createElement("span");
      segment.className = `statistics-knowledge-segment statistics-knowledge-${band.key}`;
      segment.style.flexGrow = String(band.count);
      segment.title = band.segmentTitle;
      return segment;
    });
  bar.replaceChildren(...segments);
  bar.setAttribute("aria-label", words.ariaLabel);

  const legend = words.bands.map((band) => {
    const item = documentRoot.createElement("li");
    item.className = `statistics-knowledge-item statistics-knowledge-${band.key}`;
    const label = documentRoot.createElement("span");
    label.className = "statistics-knowledge-label";
    label.append(
      textElement("span", "statistics-knowledge-swatch", "", documentRoot),
      textElement("strong", "", band.label, documentRoot),
      textElement("small", "", band.masteryText, documentRoot),
    );
    const count = textElement("span", "statistics-knowledge-count", "", documentRoot);
    count.append(
      textElement("strong", "", band.countText, documentRoot),
      documentRoot.createTextNode(band.shareText),
    );
    item.append(label, count);
    return item;
  });
  byId("statistics-knowledge-legend", documentRoot).replaceChildren(...legend);
  byId("statistics-words-empty", documentRoot).hidden = !words.empty;
}

function renderReadingStatistics(
  documentRoot: Document,
  presentation: StatisticsPresentation,
): void {
  const reading = presentation.reading;
  byId("statistics-texts-read", documentRoot).textContent = reading.textsRead;
  byId("statistics-reading-sessions", documentRoot).textContent = reading.sessions;
  byId("statistics-active-time", documentRoot).textContent = reading.activeTime;
  byId("statistics-recent-wpm", documentRoot).textContent = reading.recentWpm;
  byId("statistics-lifetime-wpm", documentRoot).textContent = reading.lifetimeWpm;
  byId("statistics-recent-wpm-detail", documentRoot).textContent = reading.recentWpmDetail;
  byId("statistics-lifetime-wpm-detail", documentRoot).textContent = reading.lifetimeWpmDetail;
  byId("statistics-reading-note", documentRoot).textContent = reading.note;
}

function renderLevelStatistics(documentRoot: Document, presentation: StatisticsPresentation): void {
  const level = presentation.level;
  const category = byId("statistics-level-category", documentRoot);
  category.textContent = level.category;
  category.title = level.categoryTitle;
  byId("statistics-level-value", documentRoot).textContent = level.value;
  byId("statistics-level-source", documentRoot).textContent = level.source;
  const meter = byId("statistics-level-meter", documentRoot);
  meter.setAttribute("aria-valuenow", String(level.percent));
  meter.setAttribute("aria-valuetext", level.meterText);
  byId("statistics-level-meter-fill", documentRoot).style.width = `${level.percent}%`;
  byId("statistics-level-status", documentRoot).textContent = level.status;
  byId("statistics-level-range", documentRoot).textContent = level.range;
  byId("statistics-level-evidence", documentRoot).textContent = level.evidence;
}

function readingNote(state: StatisticsPayload): string {
  const reading = state.reading;
  const unitNote = isChineseLanguage(state.learning_language)
    ? " Chinese WPM uses the lesson’s segmented word units, not individual characters."
    : "";
  return `Pace uses completed reads with at least ${formatActiveTime(
    reading.minimum_wpm_active_seconds,
  )} active time and ${Math.round(
    reading.minimum_wpm_completion_ratio * 100,
  )}% completion.${unitNote} Active time from unfinished sessions remains only in this browser.`;
}

function levelRange(state: StatisticsPayload): string {
  const level = state.level;
  if (level.lower === null || level.upper === null) {
    return "Calibration texts will add a confidence range.";
  }
  const lower = readerLevelPresentation(
    state.learning_language,
    level.lower_category ?? level.category,
    level.lower,
  );
  const upper = readerLevelPresentation(
    state.learning_language,
    level.upper_category ?? level.category,
    level.upper,
  );
  return (
    `Likely range ${level.lower.toFixed(2)}–${level.upper.toFixed(2)}` +
    ` · ${lower.label} to ${upper.label}`
  );
}
