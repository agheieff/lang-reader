import { createJsonContract } from "./json-contract.js";
import { parseTermBands, type TermBand } from "./term-band.js";

export type JsonRecord = Record<string, unknown>;
export { TERM_BANDS, type TermBand } from "./term-band.js";

export interface Term {
  key: string;
  lemma: string;
  pos: string;
  gloss: string;
  pronunciation?: string | undefined;
  frequency_rank?: number | undefined;
}

export interface Run {
  text: string;
  term?: Term | undefined;
  pronunciation?: string | undefined;
}

export interface GrammarOccurrence {
  key: string;
  construction_key: string;
  run_start: number;
  run_end: number;
  note?: string | undefined;
}

export interface GrammarCatalogEntry {
  key: string;
  label: string;
  description: string;
  category: string;
  difficulty: number;
  generation_hint?: string | undefined;
}

export interface Sentence {
  key: string;
  translation?: string | undefined;
  runs: Run[];
  grammar: GrammarOccurrence[];
}

export interface Block {
  key: string;
  sentences: Sentence[];
}

export interface Lesson {
  schema_version: 1;
  key: string;
  title: string;
  title_sentence?: Sentence | undefined;
  learning_language: string;
  translation_language: string;
  topic?: string | undefined;
  level: string;
  difficulty: number;
  blocks: Block[];
  target_term_keys: string[];
  metadata: JsonRecord;
}

export interface Profile {
  learning_language: string;
  translation_language: string;
  level: string;
  difficulty: number;
  interests: string[];
  preferences: JsonRecord;
}

export interface WordSurfaceForm {
  text: string;
  occurrences: number;
}

export interface RelatedWordSense {
  key: string;
  pos: string;
  gloss: string;
  pronunciation?: string | undefined;
}

export interface WordView {
  key: string;
  lemma: string;
  pos: string;
  gloss: string;
  pronunciation?: string | undefined;
  frequency_rank?: number | undefined;
  surface_forms: WordSurfaceForm[];
  related_senses: RelatedWordSense[];
  occurrence_count: number;
  exposed_lesson_count: number;
  raw_reveal_count: number;
  counted_reveal_sessions: number;
  qualified_exposures: number;
  mastery: number;
  stability_days: number;
  reveal_failures: number;
  first_exposed_at: string;
  last_exposed_at: string;
  last_revealed_at?: string | undefined;
  next_due_at?: string | undefined;
}

export interface WordsPayload {
  learning_language: string;
  translation_language: string;
  words: WordView[];
}

export interface ReaderProgress {
  session_id?: string | undefined;
  started: boolean;
  completed: boolean;
  rating: -1 | 1 | null;
  revealed_term_keys: string[];
  revealed_sentence_keys: string[];
  revealed_grammar_occurrence_keys: string[];
  full_translation_revealed: boolean;
}

export type GenerationStatus = "pending" | "running" | "failed";

export interface ReaderPayload {
  profile: Profile;
  lesson_id: number | null;
  lesson: Lesson | null;
  term_bands: Record<string, TermBand>;
  grammar_catalog: GrammarCatalogEntry[];
  comfortable_grammar_construction_keys: string[];
  progress: ReaderProgress;
  generation_status: GenerationStatus | null;
}

export const EVENT_TYPES = [
  "lesson.started",
  "term.revealed",
  "translation.revealed",
  "lesson.completed",
  "lesson.rated",
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export interface LearningEvent {
  event_id: string;
  session_id: string;
  lesson_id: number;
  type: EventType;
  occurred_at: string;
  payload: JsonRecord;
}

const {
  record: asRecord,
  text: asText,
  optionalText: asOptionalText,
  boundedNumber: asDifficulty,
  integer: asInteger,
  nonNegativeNumber: asNonNegativeNumber,
  timestamp: asDateText,
} = createJsonContract("Invalid reader", {
  allowBlankText: true,
  textRequirement: "must be text",
});

function asKey(value: unknown, name: string): string {
  const key = asText(value, name);
  if (!key.trim()) throw new Error(`Invalid reader: ${name} must not be blank`);
  return key;
}

function asStringList(value: unknown, name: string): string[] {
  if (!Array.isArray(value)) throw new Error(`Invalid reader: ${name} must be a list`);
  return value.map((item, index) => asText(item, `${name}[${index}]`));
}

function asNonEmptyList(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`Invalid reader: ${name} must be a non-empty list`);
  }
  return value;
}

function parseTerm(value: unknown): Term | undefined {
  if (value === undefined || value === null) return undefined;
  const term = asRecord(value, "term");
  const rawRank = term.frequency_rank;
  let frequencyRank: number | undefined;
  if (rawRank !== undefined && rawRank !== null) {
    if (!Number.isInteger(rawRank) || (rawRank as number) < 1) {
      throw new Error("Invalid reader: term.frequency_rank must be a positive integer");
    }
    frequencyRank = rawRank as number;
  }
  return {
    key: asKey(term.key, "term.key"),
    lemma: asKey(term.lemma, "term.lemma"),
    pos: asKey(term.pos, "term.pos"),
    gloss: asKey(term.gloss, "term.gloss"),
    pronunciation: asOptionalText(term.pronunciation, "term.pronunciation"),
    frequency_rank: frequencyRank,
  };
}

function parseRun(value: unknown): Run {
  const run = asRecord(value, "run");
  const text = asText(run.text, "run.text");
  const term = parseTerm(run.term);
  const pronunciation =
    run.pronunciation === undefined || run.pronunciation === null
      ? undefined
      : asKey(run.pronunciation, "run.pronunciation");
  if (term && !text.trim()) throw new Error("Invalid reader: a term run must have visible text");
  return { text, term, pronunciation };
}

function parseGrammarOccurrence(
  value: unknown,
  index: number,
  runCount: number,
): GrammarOccurrence {
  const path = `sentence.grammar[${index}]`;
  const occurrence = asRecord(value, path);
  const runStart = asInteger(occurrence.run_start, `${path}.run_start`);
  const runEnd = asInteger(occurrence.run_end, `${path}.run_end`, 1);
  if (runEnd <= runStart || runEnd > runCount) {
    throw new Error(`Invalid reader: ${path} must cover a non-empty range of sentence runs`);
  }
  return {
    key: asKey(occurrence.key, `${path}.key`),
    construction_key: asKey(occurrence.construction_key, `${path}.construction_key`),
    run_start: runStart,
    run_end: runEnd,
    note:
      occurrence.note === undefined || occurrence.note === null
        ? undefined
        : asKey(occurrence.note, `${path}.note`),
  };
}

function parseSentence(value: unknown): Sentence {
  const sentence = asRecord(value, "sentence");
  const runs = asNonEmptyList(sentence.runs, "sentence.runs").map(parseRun);
  const grammar = Array.isArray(sentence.grammar)
    ? sentence.grammar.map((occurrence, index) =>
        parseGrammarOccurrence(occurrence, index, runs.length),
      )
    : sentence.grammar === undefined
      ? []
      : (() => {
          throw new Error("Invalid reader: sentence.grammar must be a list");
        })();
  if (new Set(grammar.map((occurrence) => occurrence.key)).size !== grammar.length) {
    throw new Error("Invalid reader: grammar occurrence keys must be unique within a sentence");
  }
  return {
    key: asKey(sentence.key, "sentence.key"),
    translation: asOptionalText(sentence.translation, "sentence.translation"),
    runs,
    grammar,
  };
}

function parseBlock(value: unknown): Block {
  const block = asRecord(value, "block");
  return {
    key: asKey(block.key, "block.key"),
    sentences: asNonEmptyList(block.sentences, "block.sentences").map(parseSentence),
  };
}

export function parseLesson(value: unknown): Lesson {
  const lesson = asRecord(value, "lesson");
  if (lesson.schema_version !== 1) {
    throw new Error("Invalid reader: unsupported lesson.schema_version");
  }
  const parsed: Lesson = {
    schema_version: 1,
    key: asKey(lesson.key, "lesson.key"),
    title: asKey(lesson.title, "lesson.title"),
    title_sentence:
      lesson.title_sentence === undefined || lesson.title_sentence === null
        ? undefined
        : parseSentence(lesson.title_sentence),
    learning_language: asKey(lesson.learning_language, "lesson.learning_language"),
    translation_language: asKey(lesson.translation_language, "lesson.translation_language"),
    topic: asOptionalText(lesson.topic, "lesson.topic"),
    level: asKey(lesson.level, "lesson.level"),
    difficulty: asDifficulty(lesson.difficulty, "lesson.difficulty"),
    blocks: asNonEmptyList(lesson.blocks, "lesson.blocks").map(parseBlock),
    target_term_keys: asStringList(lesson.target_term_keys, "lesson.target_term_keys"),
    metadata: asRecord(lesson.metadata, "lesson.metadata"),
  };
  const sentences = [
    ...(parsed.title_sentence ? [parsed.title_sentence] : []),
    ...parsed.blocks.flatMap((block) => block.sentences),
  ];
  const grammarKeys = sentences.flatMap((sentence) =>
    sentence.grammar.map((occurrence) => occurrence.key),
  );
  if (new Set(grammarKeys).size !== grammarKeys.length) {
    throw new Error("Invalid reader: grammar occurrence keys must be unique across a lesson");
  }
  return parsed;
}

function parseGrammarCatalogEntry(value: unknown, index: number): GrammarCatalogEntry {
  const path = `grammar_catalog[${index}]`;
  const entry = asRecord(value, path);
  return {
    key: asKey(entry.key, `${path}.key`),
    label: asKey(entry.label, `${path}.label`),
    description: asKey(entry.description, `${path}.description`),
    category: asKey(entry.category, `${path}.category`),
    difficulty: asDifficulty(entry.difficulty, `${path}.difficulty`),
    generation_hint: asOptionalText(entry.generation_hint, `${path}.generation_hint`),
  };
}

export function parseGrammarCatalog(value: unknown): GrammarCatalogEntry[] {
  if (!Array.isArray(value)) {
    throw new Error("Invalid reader: grammar_catalog must be a list");
  }
  const entries = value.map(parseGrammarCatalogEntry);
  if (new Set(entries.map((entry) => entry.key)).size !== entries.length) {
    throw new Error("Invalid reader: grammar catalog keys must be unique");
  }
  return entries;
}

export function parseComfortableGrammarConstructionKeys(value: unknown): string[] {
  if (value === undefined) return [];
  const keys = asStringList(value, "comfortable_grammar_construction_keys").map((key, index) =>
    asKey(key, `comfortable_grammar_construction_keys[${index}]`),
  );
  if (new Set(keys).size !== keys.length) {
    throw new Error(
      "Invalid reader: comfortable_grammar_construction_keys must contain unique keys",
    );
  }
  return keys;
}

function parseProfile(value: unknown): Profile {
  const profile = asRecord(value, "profile");
  return {
    learning_language: asKey(profile.learning_language, "profile.learning_language"),
    translation_language: asKey(profile.translation_language, "profile.translation_language"),
    level: asKey(profile.level, "profile.level"),
    difficulty: asDifficulty(profile.difficulty, "profile.difficulty"),
    interests: asStringList(profile.interests, "profile.interests"),
    preferences: asRecord(profile.preferences, "profile.preferences"),
  };
}

function parseProgress(value: unknown): ReaderProgress {
  const progress = asRecord(value, "progress");
  const rating = progress.rating;
  if (rating !== null && rating !== -1 && rating !== 1) {
    throw new Error("Invalid reader: progress.rating must be -1, 1, or null");
  }
  if (typeof progress.started !== "boolean" || typeof progress.completed !== "boolean") {
    throw new Error("Invalid reader: progress status fields must be boolean");
  }
  if (typeof progress.full_translation_revealed !== "boolean") {
    throw new Error("Invalid reader: progress.full_translation_revealed must be boolean");
  }
  return {
    session_id: asOptionalText(progress.session_id, "progress.session_id"),
    started: progress.started,
    completed: progress.completed,
    rating,
    revealed_term_keys: asStringList(progress.revealed_term_keys, "progress.revealed_term_keys"),
    revealed_sentence_keys: asStringList(
      progress.revealed_sentence_keys,
      "progress.revealed_sentence_keys",
    ),
    revealed_grammar_occurrence_keys:
      progress.revealed_grammar_occurrence_keys === undefined
        ? []
        : asStringList(
            progress.revealed_grammar_occurrence_keys,
            "progress.revealed_grammar_occurrence_keys",
          ),
    full_translation_revealed: progress.full_translation_revealed,
  };
}

function parseGenerationStatus(value: unknown): GenerationStatus | null {
  if (value === undefined || value === null) return null;
  if (value !== "pending" && value !== "running" && value !== "failed") {
    throw new Error("Invalid reader: generation_status is not supported");
  }
  return value;
}

export function parseReader(value: unknown): ReaderPayload {
  const reader = asRecord(value, "response");
  const rawLessonId = reader.lesson_id;
  if (rawLessonId !== null && (!Number.isInteger(rawLessonId) || (rawLessonId as number) < 1)) {
    throw new Error("Invalid reader: lesson_id must be a positive integer or null");
  }
  const lessonId = rawLessonId as number | null;
  const lesson = reader.lesson === null ? null : parseLesson(reader.lesson);
  if ((lessonId === null) !== (lesson === null)) {
    throw new Error("Invalid reader: lesson_id and lesson must both be present or null");
  }
  const grammarCatalog =
    reader.grammar_catalog === undefined ? [] : parseGrammarCatalog(reader.grammar_catalog);
  return {
    profile: parseProfile(reader.profile),
    lesson_id: lessonId,
    lesson,
    term_bands: parseTermBands(reader.term_bands),
    grammar_catalog: grammarCatalog,
    comfortable_grammar_construction_keys: parseComfortableGrammarConstructionKeys(
      reader.comfortable_grammar_construction_keys,
    ),
    progress: parseProgress(reader.progress),
    generation_status: parseGenerationStatus(reader.generation_status),
  };
}

function parseWordSurface(value: unknown, index: number): WordSurfaceForm {
  const surface = asRecord(value, `words.surface_forms[${index}]`);
  return {
    text: asKey(surface.text, `words.surface_forms[${index}].text`),
    occurrences: asInteger(surface.occurrences, `words.surface_forms[${index}].occurrences`, 1),
  };
}

function parseRelatedWordSense(value: unknown, wordIndex: number, senseIndex: number) {
  const path = `words[${wordIndex}].related_senses[${senseIndex}]`;
  const sense = asRecord(value, path);
  return {
    key: asKey(sense.key, `${path}.key`),
    pos: asKey(sense.pos, `${path}.pos`),
    gloss: asKey(sense.gloss, `${path}.gloss`),
    pronunciation: asOptionalText(sense.pronunciation, `${path}.pronunciation`),
  };
}

function parseWord(value: unknown, index: number): WordView {
  const raw = asRecord(value, `words[${index}]`);
  const term = parseTerm(raw);
  if (!term) throw new Error(`Invalid reader: words[${index}] must define a term`);
  const surfaces = asNonEmptyList(raw.surface_forms, `words[${index}].surface_forms`).map(
    parseWordSurface,
  );
  const occurrenceCount = asInteger(raw.occurrence_count, `words[${index}].occurrence_count`, 1);
  if (surfaces.reduce((total, surface) => total + surface.occurrences, 0) !== occurrenceCount) {
    throw new Error(`Invalid reader: words[${index}] surface occurrences do not match total`);
  }
  const rawRevealCount = asInteger(raw.raw_reveal_count, `words[${index}].raw_reveal_count`);
  const countedReveals = asInteger(
    raw.counted_reveal_sessions,
    `words[${index}].counted_reveal_sessions`,
  );
  return {
    ...term,
    surface_forms: surfaces,
    related_senses: Array.isArray(raw.related_senses)
      ? raw.related_senses.map((sense, senseIndex) =>
          parseRelatedWordSense(sense, index, senseIndex),
        )
      : (() => {
          throw new Error(`Invalid reader: words[${index}].related_senses must be a list`);
        })(),
    occurrence_count: occurrenceCount,
    exposed_lesson_count: asInteger(
      raw.exposed_lesson_count,
      `words[${index}].exposed_lesson_count`,
      1,
    ),
    raw_reveal_count: rawRevealCount,
    counted_reveal_sessions: countedReveals,
    qualified_exposures: asInteger(raw.qualified_exposures, `words[${index}].qualified_exposures`),
    mastery: asDifficulty(raw.mastery, `words[${index}].mastery`),
    stability_days: asNonNegativeNumber(raw.stability_days, `words[${index}].stability_days`),
    reveal_failures: asNonNegativeNumber(raw.reveal_failures, `words[${index}].reveal_failures`),
    first_exposed_at: asDateText(raw.first_exposed_at, `words[${index}].first_exposed_at`),
    last_exposed_at: asDateText(raw.last_exposed_at, `words[${index}].last_exposed_at`),
    last_revealed_at: asDateText(raw.last_revealed_at, `words[${index}].last_revealed_at`, true),
    next_due_at: asDateText(raw.next_due_at, `words[${index}].next_due_at`, true),
  };
}

export function parseWords(value: unknown): WordsPayload {
  const response = asRecord(value, "words response");
  const words = Array.isArray(response.words)
    ? response.words.map(parseWord)
    : (() => {
        throw new Error("Invalid reader: words must be a list");
      })();
  if (new Set(words.map((word) => word.key)).size !== words.length) {
    throw new Error("Invalid reader: word keys must be unique");
  }
  return {
    learning_language: asKey(response.learning_language, "words.learning_language"),
    translation_language: asKey(response.translation_language, "words.translation_language"),
    words,
  };
}
