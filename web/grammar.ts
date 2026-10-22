import { createJsonContract } from "./json-contract.js";
import { isReviewDue as isDue } from "./review-due.js";
import { matchesSearch, normalizeSearchText } from "./search-match.js";

export { formatReviewDue as formatGrammarDue } from "./review-due.js";

export const GRAMMAR_SORTS = [
  "attention",
  "encounters",
  "clean",
  "mastery",
  "alphabetical",
  "category",
] as const;

export type GrammarSort = (typeof GRAMMAR_SORTS)[number];

export const DEFAULT_GRAMMAR_SORT: GrammarSort = "attention";

export const CONSTRUCTION_STATUSES = [
  "seen",
  "developing",
  "needs_attention",
  "comfortable",
] as const;

export type ConstructionStatus = (typeof CONSTRUCTION_STATUSES)[number];

export interface GrammarExample {
  lesson_id: number;
  title: string;
  sentence_key: string;
  text: string;
  translation?: string | undefined;
  note?: string | undefined;
}

export interface GrammarConstruction {
  key: string;
  label: string;
  description: string;
  category: string;
  difficulty: number;
  occurrence_count: number;
  exposed_lesson_count: number;
  raw_help_count: number;
  counted_help_sessions: number;
  inferred_difficulty_signals: number;
  qualified_exposures: number;
  mastery: number;
  stability_days: number;
  first_exposed_at: string;
  last_exposed_at: string;
  last_helped_at?: string | undefined;
  next_due_at?: string | undefined;
  examples: GrammarExample[];
}

export interface GrammarState {
  learning_language: string;
  translation_language: string;
  constructions: GrammarConstruction[];
}

export interface ConstructionStatusDetail {
  label: string;
  description: string;
}

export const CONSTRUCTION_STATUS_DETAILS: Record<ConstructionStatus, ConstructionStatusDetail> = {
  seen: {
    label: "Seen",
    description: "Encountered, with too little evidence to judge yet.",
  },
  developing: {
    label: "Developing",
    description: "Evidence is mixed or still accumulating.",
  },
  needs_attention: {
    label: "Needs attention",
    description: "Repeated, deduplicated help signals outweigh clean evidence.",
  },
  comfortable: {
    label: "Comfortable",
    description: "Supported by varied, mostly unassisted reading evidence.",
  },
};

const { record, text, optionalText, integer, nonNegativeNumber, boundedNumber, timestamp } =
  createJsonContract("Invalid grammar");

const COMFORTABLE_MIN_QUALIFIED_EXPOSURES = 4;
const COMFORTABLE_MIN_LESSONS = 3;
const COMFORTABLE_MIN_MASTERY = 0.8;
const COMFORTABLE_MIN_STABILITY_DAYS = 7;
const COMFORTABLE_MAX_SIGNAL_SHARE = 0.25;
const ATTENTION_MIN_SIGNALS = 2;
const ATTENTION_MAX_MASTERY = 0.6;
const ATTENTION_MIN_SIGNAL_SHARE = 0.5;

export function constructionDifficultySignals(construction: GrammarConstruction): number {
  return construction.counted_help_sessions + construction.inferred_difficulty_signals;
}

/**
 * Keep labels deliberately conservative: a single peek is not a problem, and "comfortable"
 * requires clean evidence across several texts and enough modeled stability to be meaningful.
 */
export function constructionStatus(construction: GrammarConstruction): ConstructionStatus {
  const signals = constructionDifficultySignals(construction);
  const evidence = construction.qualified_exposures + signals;
  if (evidence <= 1) return "seen";

  const signalShare = signals / evidence;
  if (
    signals >= ATTENTION_MIN_SIGNALS &&
    signalShare >= ATTENTION_MIN_SIGNAL_SHARE &&
    construction.mastery < ATTENTION_MAX_MASTERY
  ) {
    return "needs_attention";
  }

  if (
    construction.qualified_exposures >= COMFORTABLE_MIN_QUALIFIED_EXPOSURES &&
    construction.exposed_lesson_count >= COMFORTABLE_MIN_LESSONS &&
    construction.mastery >= COMFORTABLE_MIN_MASTERY &&
    construction.stability_days >= COMFORTABLE_MIN_STABILITY_DAYS &&
    signalShare <= COMFORTABLE_MAX_SIGNAL_SHARE
  ) {
    return "comfortable";
  }

  return "developing";
}

export function formatGrammarStatus(construction: GrammarConstruction): string {
  return CONSTRUCTION_STATUS_DETAILS[constructionStatus(construction)].label;
}

export function selectGrammar(
  constructions: readonly GrammarConstruction[],
  query: string,
  sort: GrammarSort,
  now = new Date(),
): GrammarConstruction[] {
  const needle = normalizeSearchText(query.trim());
  const selected = needle
    ? constructions.filter((construction) =>
        searchableValues(construction).some((value) =>
          matchesSearch(value, needle, { compactSymbols: true }),
        ),
      )
    : [...constructions];
  return selected.sort((left, right) => compareConstructions(left, right, sort, now));
}

export function parseGrammar(value: unknown): GrammarState {
  const response = record(value, "response");
  if (!Array.isArray(response.constructions)) {
    throw new Error("Invalid grammar: constructions must be a list");
  }
  const constructions = response.constructions.map(parseConstruction);
  if (
    new Set(constructions.map((construction) => construction.key)).size !== constructions.length
  ) {
    throw new Error("Invalid grammar: construction keys must be unique");
  }
  return {
    learning_language: text(response.learning_language, "learning_language"),
    translation_language: text(response.translation_language, "translation_language"),
    constructions,
  };
}

function parseConstruction(value: unknown, index: number): GrammarConstruction {
  const path = `constructions[${index}]`;
  const construction = record(value, path);
  const occurrenceCount = integer(construction.occurrence_count, `${path}.occurrence_count`, 1);
  const exposedLessonCount = integer(
    construction.exposed_lesson_count,
    `${path}.exposed_lesson_count`,
    1,
  );
  if (exposedLessonCount > occurrenceCount) {
    throw new Error(`Invalid grammar: ${path} cannot occur in fewer instances than lessons`);
  }
  const rawHelpCount = integer(construction.raw_help_count, `${path}.raw_help_count`);
  const countedHelpSessions = integer(
    construction.counted_help_sessions,
    `${path}.counted_help_sessions`,
  );
  if (countedHelpSessions > rawHelpCount) {
    throw new Error(`Invalid grammar: ${path} counted help exceeds raw help`);
  }
  if (!Array.isArray(construction.examples)) {
    throw new Error(`Invalid grammar: ${path}.examples must be a list`);
  }

  return {
    key: text(construction.key, `${path}.key`),
    label: text(construction.label, `${path}.label`),
    description: text(construction.description, `${path}.description`),
    category: text(construction.category, `${path}.category`),
    difficulty: boundedNumber(construction.difficulty, `${path}.difficulty`),
    occurrence_count: occurrenceCount,
    exposed_lesson_count: exposedLessonCount,
    raw_help_count: rawHelpCount,
    counted_help_sessions: countedHelpSessions,
    inferred_difficulty_signals: integer(
      construction.inferred_difficulty_signals,
      `${path}.inferred_difficulty_signals`,
    ),
    qualified_exposures: integer(construction.qualified_exposures, `${path}.qualified_exposures`),
    mastery: boundedNumber(construction.mastery, `${path}.mastery`),
    stability_days: nonNegativeNumber(construction.stability_days, `${path}.stability_days`),
    first_exposed_at: timestamp(construction.first_exposed_at, `${path}.first_exposed_at`),
    last_exposed_at: timestamp(construction.last_exposed_at, `${path}.last_exposed_at`),
    last_helped_at: timestamp(construction.last_helped_at, `${path}.last_helped_at`, true),
    next_due_at: timestamp(construction.next_due_at, `${path}.next_due_at`, true),
    examples: construction.examples.map((example, exampleIndex) =>
      parseExample(example, path, exampleIndex),
    ),
  };
}

function parseExample(value: unknown, constructionPath: string, index: number): GrammarExample {
  const path = `${constructionPath}.examples[${index}]`;
  const example = record(value, path);
  return {
    lesson_id: integer(example.lesson_id, `${path}.lesson_id`, 1),
    title: text(example.title, `${path}.title`),
    sentence_key: text(example.sentence_key, `${path}.sentence_key`),
    text: text(example.text, `${path}.text`),
    translation: optionalText(example.translation, `${path}.translation`),
    note: optionalText(example.note, `${path}.note`),
  };
}

function searchableValues(construction: GrammarConstruction): readonly string[] {
  return [
    construction.key,
    construction.label,
    construction.description,
    construction.category,
    ...construction.examples.flatMap((example) => [
      example.title,
      example.text,
      example.translation ?? "",
      example.note ?? "",
    ]),
  ];
}

function compareConstructions(
  left: GrammarConstruction,
  right: GrammarConstruction,
  sort: GrammarSort,
  now: Date,
): number {
  let difference = 0;
  if (sort === "attention") {
    difference =
      Number(isDue(right, now)) - Number(isDue(left, now)) ||
      statusPriority(constructionStatus(left)) - statusPriority(constructionStatus(right)) ||
      constructionDifficultySignals(right) - constructionDifficultySignals(left) ||
      left.mastery - right.mastery;
  } else if (sort === "encounters") {
    difference =
      right.occurrence_count - left.occurrence_count ||
      right.exposed_lesson_count - left.exposed_lesson_count;
  } else if (sort === "clean") {
    difference =
      right.qualified_exposures - left.qualified_exposures ||
      constructionDifficultySignals(left) - constructionDifficultySignals(right);
  } else if (sort === "mastery") {
    difference =
      left.mastery - right.mastery ||
      constructionDifficultySignals(right) - constructionDifficultySignals(left);
  } else if (sort === "category") {
    difference =
      left.category.localeCompare(right.category) || left.label.localeCompare(right.label);
  }
  return difference || left.label.localeCompare(right.label) || left.key.localeCompare(right.key);
}

function statusPriority(status: ConstructionStatus): number {
  if (status === "needs_attention") return 0;
  if (status === "seen") return 1;
  if (status === "developing") return 2;
  return 3;
}
