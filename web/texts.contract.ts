import {
  type GrammarCatalogEntry,
  type Lesson,
  parseComfortableGrammarConstructionKeys,
  parseGrammarCatalog,
  parseLesson,
} from "./contracts.js";
import { createJsonContract } from "./json-contract.js";
import { parseTermBands, type TermBand } from "./term-band.js";

const TEXT_STATUSES = ["queued", "in_progress", "skipped", "read"] as const;
export type TextStatus = (typeof TEXT_STATUSES)[number];

const TEXT_REQUEST_STATES = ["pending", "running", "completed", "failed"] as const;
export type TextRequestState = (typeof TEXT_REQUEST_STATES)[number];

const TEXT_PREPARATION_STATES = ["pending", "running"] as const;
export type TextPreparationState = (typeof TEXT_PREPARATION_STATES)[number];

const TEXT_PREPARATION_KINDS = ["queue_fill", "topic_request"] as const;
export type TextPreparationKind = (typeof TEXT_PREPARATION_KINDS)[number];

export interface TextLibraryItem {
  id: number;
  key: string;
  title: string;
  topic?: string | undefined;
  level: string;
  difficulty: number;
  imported_at: string;
  status: TextStatus;
  queue_position: number | null;
  opened_at?: string | undefined;
  skipped_at?: string | undefined;
  last_completed_at?: string | undefined;
  session_count: number;
  completion_count: number;
  rating: -1 | 1 | null;
  lexical_token_count: number;
  known_share: number | null;
}

export interface TextRequestItem {
  task_id: number;
  request_id: string;
  topic: string;
  state: TextRequestState;
  created_at: string;
  updated_at: string;
  error?: string | undefined;
  lesson_id: number | null;
}

export interface TextPreparationItem {
  task_id: number;
  state: TextPreparationState;
  request_kind: TextPreparationKind;
  requested_topic: string | null;
  waiting_for_capacity: boolean;
  created_at: string;
  updated_at: string;
}

export interface TextLibraryPayload {
  learning_language: string;
  translation_language: string;
  texts: TextLibraryItem[];
  preparations: TextPreparationItem[];
  requests: TextRequestItem[];
}

export interface TextPreviewPayload {
  lesson_id: number;
  lesson: Lesson;
  term_bands: Record<string, TermBand>;
  grammar_catalog: GrammarCatalogEntry[];
  comfortable_grammar_construction_keys: string[];
}

const {
  record,
  text,
  optionalText,
  integer,
  nullableInteger,
  boundedNumber: difficulty,
  nullableNumber,
  boolean,
  timestamp,
  oneOf,
} = createJsonContract("Invalid texts");

function parseTextItem(value: unknown, index: number): TextLibraryItem {
  const path = `texts[${index}]`;
  const item = record(value, path);
  const rating = item.rating;
  if (rating !== null && rating !== -1 && rating !== 1) {
    throw new Error(`Invalid texts: ${path}.rating must be -1, 1, or null`);
  }
  return {
    id: integer(item.id, `${path}.id`, 1),
    key: text(item.key, `${path}.key`),
    title: text(item.title, `${path}.title`),
    topic: optionalText(item.topic, `${path}.topic`),
    level: text(item.level, `${path}.level`),
    difficulty: difficulty(item.difficulty, `${path}.difficulty`),
    imported_at: timestamp(item.imported_at, `${path}.imported_at`),
    status: oneOf(item.status, TEXT_STATUSES, `${path}.status`),
    queue_position: nullableInteger(item.queue_position, `${path}.queue_position`, 1),
    opened_at: timestamp(item.opened_at, `${path}.opened_at`, true),
    skipped_at: timestamp(item.skipped_at, `${path}.skipped_at`, true),
    last_completed_at: timestamp(item.last_completed_at, `${path}.last_completed_at`, true),
    session_count: integer(item.session_count, `${path}.session_count`),
    completion_count: integer(item.completion_count, `${path}.completion_count`),
    rating,
    lexical_token_count: integer(item.lexical_token_count, `${path}.lexical_token_count`),
    known_share:
      item.known_share === undefined
        ? null
        : nullableNumber(item.known_share, `${path}.known_share`, 0, 1),
  };
}

export function parseTextRequest(value: unknown, index = 0): TextRequestItem {
  const path = `requests[${index}]`;
  const item = record(value, path);
  return {
    task_id: integer(item.task_id, `${path}.task_id`, 1),
    request_id: text(item.request_id, `${path}.request_id`),
    topic: text(item.topic, `${path}.topic`),
    state: oneOf(item.state, TEXT_REQUEST_STATES, `${path}.state`),
    created_at: timestamp(item.created_at, `${path}.created_at`),
    updated_at: timestamp(item.updated_at, `${path}.updated_at`),
    error: optionalText(item.error, `${path}.error`),
    lesson_id: nullableInteger(item.lesson_id, `${path}.lesson_id`, 1),
  };
}

export function parseTextPreparation(value: unknown, index = 0): TextPreparationItem {
  const path = `preparations[${index}]`;
  const item = record(value, path);
  const requestKind = oneOf(item.request_kind, TEXT_PREPARATION_KINDS, `${path}.request_kind`);
  const requestedTopic =
    item.requested_topic === null ? null : text(item.requested_topic, `${path}.requested_topic`);
  if (requestKind === "topic_request" && requestedTopic === null) {
    throw new Error(`Invalid texts: ${path}.requested_topic is required for a topic request`);
  }
  if (requestKind === "queue_fill" && requestedTopic !== null) {
    throw new Error(`Invalid texts: ${path}.requested_topic must be null for a queue fill`);
  }
  return {
    task_id: integer(item.task_id, `${path}.task_id`, 1),
    state: oneOf(item.state, TEXT_PREPARATION_STATES, `${path}.state`),
    request_kind: requestKind,
    requested_topic: requestedTopic,
    waiting_for_capacity: boolean(item.waiting_for_capacity, `${path}.waiting_for_capacity`),
    created_at: timestamp(item.created_at, `${path}.created_at`),
    updated_at: timestamp(item.updated_at, `${path}.updated_at`),
  };
}

export function parseTextLibrary(value: unknown): TextLibraryPayload {
  const response = record(value, "response");
  if (
    !Array.isArray(response.texts) ||
    !Array.isArray(response.preparations) ||
    !Array.isArray(response.requests)
  ) {
    throw new Error("Invalid texts: texts, preparations, and requests must be lists");
  }
  const texts = response.texts.map(parseTextItem);
  const preparations = response.preparations.map(parseTextPreparation);
  const requests = response.requests.map(parseTextRequest);
  if (new Set(texts.map((item) => item.id)).size !== texts.length) {
    throw new Error("Invalid texts: text IDs must be unique");
  }
  if (new Set(preparations.map((item) => item.task_id)).size !== preparations.length) {
    throw new Error("Invalid texts: preparation task IDs must be unique");
  }
  if (new Set(requests.map((item) => item.request_id)).size !== requests.length) {
    throw new Error("Invalid texts: request IDs must be unique");
  }
  return {
    learning_language: text(response.learning_language, "learning_language"),
    translation_language: text(response.translation_language, "translation_language"),
    texts,
    preparations,
    requests,
  };
}

export function parseTextPreview(value: unknown): TextPreviewPayload {
  const response = record(value, "preview response");
  return {
    lesson_id: integer(response.lesson_id, "lesson_id", 1),
    lesson: parseLesson(response.lesson),
    term_bands: parseTermBands(response.term_bands, {
      prefix: "Invalid texts",
      blankKeyRequirement: "must be non-blank text",
      unsupportedBandRequirement: "is not supported",
    }),
    grammar_catalog:
      response.grammar_catalog === undefined ? [] : parseGrammarCatalog(response.grammar_catalog),
    comfortable_grammar_construction_keys: parseComfortableGrammarConstructionKeys(
      response.comfortable_grammar_construction_keys,
    ),
  };
}
