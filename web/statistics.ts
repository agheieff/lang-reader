import { createJsonContract } from "./json-contract.js";

const LEVEL_SOURCES = ["unknown", "self_reported", "estimated"] as const;
export type LevelSource = (typeof LEVEL_SOURCES)[number];

const CALIBRATION_STATUSES = ["unstarted", "collecting", "rough", "stable"] as const;
export type CalibrationStatus = (typeof CALIBRATION_STATUSES)[number];

export interface StatisticsWords {
  total: number;
  learning: number;
  expected: number;
  familiar: number;
  mastered: number;
  expected_min_mastery: number;
  familiar_min_mastery: number;
  mastered_min_mastery: number;
}

export interface StatisticsReading {
  texts_read: number;
  completed_sessions: number;
  total_active_seconds: number;
  recent_average_wpm: number | null;
  recent_wpm_sessions: number;
  lifetime_average_wpm: number | null;
  lifetime_wpm_sessions: number;
  recent_window_size: number;
  minimum_wpm_active_seconds: number;
  minimum_wpm_completion_ratio: number;
}

export interface StatisticsLevel {
  value: number;
  category: string;
  source: LevelSource;
  status: CalibrationStatus;
  lower: number | null;
  upper: number | null;
  lower_category: string | null;
  upper_category: string | null;
  qualified_attempts: number;
  usable_probes: number;
  qualified_readings: number;
}

export interface StatisticsPayload {
  learning_language: string;
  translation_language: string;
  words: StatisticsWords;
  reading: StatisticsReading;
  level: StatisticsLevel;
}

export interface KnowledgeBand {
  key: "learning" | "expected" | "familiar" | "mastered";
  label: string;
  count: number;
  minimum: number;
  maximum: number;
  share: number;
}

const {
  record: asRecord,
  text: asText,
  integer: asInteger,
  number: asNumber,
  nullableNumber: asNullableNumber,
  nullableText: asNullableText,
  oneOf: asEnum,
} = createJsonContract("Invalid statistics", { enumRequirement: "is unsupported" });

export function parseStatistics(value: unknown): StatisticsPayload {
  const root = asRecord(value, "response");
  const rawWords = asRecord(root.words, "words");
  const rawReading = asRecord(root.reading, "reading");
  const rawLevel = asRecord(root.level, "level");
  const words: StatisticsWords = {
    total: asInteger(rawWords.total, "words.total"),
    learning: asInteger(rawWords.learning, "words.learning"),
    expected: asInteger(rawWords.expected, "words.expected"),
    familiar: asInteger(rawWords.familiar, "words.familiar"),
    mastered: asInteger(rawWords.mastered, "words.mastered"),
    expected_min_mastery: asNumber(
      rawWords.expected_min_mastery,
      "words.expected_min_mastery",
      0,
      1,
    ),
    familiar_min_mastery: asNumber(
      rawWords.familiar_min_mastery,
      "words.familiar_min_mastery",
      0,
      1,
    ),
    mastered_min_mastery: asNumber(
      rawWords.mastered_min_mastery,
      "words.mastered_min_mastery",
      0,
      1,
    ),
  };
  if (words.learning + words.expected + words.familiar + words.mastered !== words.total) {
    throw new Error("Invalid statistics: word knowledge counts must sum to words.total");
  }
  if (
    !(
      words.expected_min_mastery < words.familiar_min_mastery &&
      words.familiar_min_mastery < words.mastered_min_mastery
    )
  ) {
    throw new Error("Invalid statistics: word mastery cutoffs must be strictly increasing");
  }

  const reading: StatisticsReading = {
    texts_read: asInteger(rawReading.texts_read, "reading.texts_read"),
    completed_sessions: asInteger(rawReading.completed_sessions, "reading.completed_sessions"),
    total_active_seconds: asNumber(rawReading.total_active_seconds, "reading.total_active_seconds"),
    recent_average_wpm: asNullableNumber(
      rawReading.recent_average_wpm,
      "reading.recent_average_wpm",
    ),
    recent_wpm_sessions: asInteger(rawReading.recent_wpm_sessions, "reading.recent_wpm_sessions"),
    lifetime_average_wpm: asNullableNumber(
      rawReading.lifetime_average_wpm,
      "reading.lifetime_average_wpm",
    ),
    lifetime_wpm_sessions: asInteger(
      rawReading.lifetime_wpm_sessions,
      "reading.lifetime_wpm_sessions",
    ),
    recent_window_size: asInteger(rawReading.recent_window_size, "reading.recent_window_size", 1),
    minimum_wpm_active_seconds: asNumber(
      rawReading.minimum_wpm_active_seconds,
      "reading.minimum_wpm_active_seconds",
    ),
    minimum_wpm_completion_ratio: asNumber(
      rawReading.minimum_wpm_completion_ratio,
      "reading.minimum_wpm_completion_ratio",
      0,
      1,
    ),
  };
  if (
    reading.texts_read > reading.completed_sessions ||
    reading.recent_wpm_sessions > reading.recent_window_size ||
    reading.recent_wpm_sessions > reading.lifetime_wpm_sessions
  ) {
    throw new Error("Invalid statistics: reading session counts are inconsistent");
  }
  if (
    (reading.recent_average_wpm === null) !== (reading.recent_wpm_sessions === 0) ||
    (reading.lifetime_average_wpm === null) !== (reading.lifetime_wpm_sessions === 0)
  ) {
    throw new Error("Invalid statistics: WPM values do not match their session counts");
  }

  const level: StatisticsLevel = {
    value: asNumber(rawLevel.value, "level.value", 0, 1),
    category: asText(rawLevel.category, "level.category"),
    source: asEnum(rawLevel.source, LEVEL_SOURCES, "level.source"),
    status: asEnum(rawLevel.status, CALIBRATION_STATUSES, "level.status"),
    lower: asNullableNumber(rawLevel.lower, "level.lower", 0, 1),
    upper: asNullableNumber(rawLevel.upper, "level.upper", 0, 1),
    lower_category: asNullableText(rawLevel.lower_category, "level.lower_category"),
    upper_category: asNullableText(rawLevel.upper_category, "level.upper_category"),
    qualified_attempts: asInteger(rawLevel.qualified_attempts, "level.qualified_attempts"),
    usable_probes: asInteger(rawLevel.usable_probes, "level.usable_probes"),
    qualified_readings: asInteger(rawLevel.qualified_readings, "level.qualified_readings"),
  };
  const rangeParts = [level.lower, level.upper];
  if (rangeParts.filter((part) => part !== null).length === 1) {
    throw new Error("Invalid statistics: level range must include both bounds or neither");
  }
  if (
    level.lower !== null &&
    level.upper !== null &&
    (level.lower > level.upper || level.value < level.lower || level.value > level.upper)
  ) {
    throw new Error("Invalid statistics: current level must fall inside its range");
  }

  return {
    learning_language: asText(root.learning_language, "learning_language"),
    translation_language: asText(root.translation_language, "translation_language"),
    words,
    reading,
    level,
  };
}

export function knowledgeBands(words: StatisticsWords): KnowledgeBand[] {
  const raw = [
    {
      key: "learning",
      label: "Low confidence",
      count: words.learning,
      minimum: 0,
      maximum: words.expected_min_mastery,
    },
    {
      key: "expected",
      label: "Developing",
      count: words.expected,
      minimum: words.expected_min_mastery,
      maximum: words.familiar_min_mastery,
    },
    {
      key: "familiar",
      label: "Familiar",
      count: words.familiar,
      minimum: words.familiar_min_mastery,
      maximum: words.mastered_min_mastery,
    },
    {
      key: "mastered",
      label: "Strong",
      count: words.mastered,
      minimum: words.mastered_min_mastery,
      maximum: 1,
    },
  ] as const;
  return raw.map((band) => ({
    ...band,
    share: words.total === 0 ? 0 : band.count / words.total,
  }));
}

export function formatActiveTime(seconds: number): string {
  const rounded = Math.max(0, Math.round(seconds));
  if (rounded < 60) return `${rounded}s`;
  const minutes = Math.floor(rounded / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder === 0 ? `${hours}h` : `${hours}h ${remainder}m`;
}

export function formatWpm(value: number | null): string {
  return value === null ? "—" : Math.round(value).toLocaleString();
}

export function masteryRange(band: KnowledgeBand): string {
  const lower = Math.round(band.minimum * 100);
  const upper = Math.round(band.maximum * 100);
  if (band.minimum === 0) return `under ${upper}%`;
  if (band.maximum === 1) return `${lower}% and up`;
  return `${lower}–${upper - 1}%`;
}

export function calibrationLabel(status: CalibrationStatus): string {
  if (status === "unstarted") return "No qualifying reads yet";
  if (status === "collecting") return "Collecting evidence";
  if (status === "rough") return "Rough estimate";
  return "Stable estimate";
}

export function levelSourceLabel(source: LevelSource): string {
  if (source === "self_reported") return "Self-report prior, updated by reading";
  if (source === "estimated") return "Estimated from reading";
  return "Temporary starting estimate";
}
