import { createJsonContract } from "./json-contract.js";

export interface CharacterView {
  character: string;
  occurrence_count: number;
  exposed_lesson_count: number;
  raw_reveal_count: number;
  counted_reveal_sessions: number;
  distinct_word_contexts: number;
  qualified_exposures: number;
  inferred_failure_sessions: number;
  inferred_failure_mass: number;
  direct_successes: number;
  direct_failures: number;
  mastery: number;
  mastery_uncertainty: number;
  retrievability: number;
  stability_days: number;
  first_exposed_at: string;
  last_exposed_at: string;
  last_evidence_at?: string | undefined;
  next_due_at?: string | undefined;
}

export interface CharactersPayload {
  learning_language: string;
  translation_language: string;
  characters: CharacterView[];
}

const {
  record: asRecord,
  text: asText,
  integer: asInteger,
  timestamp: asTimestamp,
  nonNegativeNumber: asNonNegativeNumber,
  boundedNumber: asBoundedNumber,
} = createJsonContract("Invalid characters");

const hanCharacter = /^\p{Script=Han}$/u;

function asHanCharacter(value: unknown, path: string): string {
  const character = asText(value, path);
  if ([...character].length !== 1 || !hanCharacter.test(character)) {
    throw new Error(`Invalid characters: ${path} must be one Han character`);
  }
  return character;
}

function parseCharacter(value: unknown, index: number): CharacterView {
  const path = `characters[${index}]`;
  const raw = asRecord(value, path);
  const occurrenceCount = asInteger(raw.occurrence_count, `${path}.occurrence_count`, 1);
  const exposedLessonCount = asInteger(raw.exposed_lesson_count, `${path}.exposed_lesson_count`, 1);
  const rawRevealCount = asInteger(raw.raw_reveal_count, `${path}.raw_reveal_count`);
  const countedRevealSessions = asInteger(
    raw.counted_reveal_sessions,
    `${path}.counted_reveal_sessions`,
  );
  const distinctWordContexts = asInteger(
    raw.distinct_word_contexts,
    `${path}.distinct_word_contexts`,
  );
  const qualifiedExposures = asInteger(raw.qualified_exposures, `${path}.qualified_exposures`);
  const inferredFailureSessions = asInteger(
    raw.inferred_failure_sessions,
    `${path}.inferred_failure_sessions`,
  );
  const inferredFailureMass = asNonNegativeNumber(
    raw.inferred_failure_mass,
    `${path}.inferred_failure_mass`,
  );
  const directSuccesses = asInteger(raw.direct_successes, `${path}.direct_successes`);
  const directFailures = asInteger(raw.direct_failures, `${path}.direct_failures`);
  const mastery = asBoundedNumber(raw.mastery, `${path}.mastery`);
  const masteryUncertainty = asBoundedNumber(
    raw.mastery_uncertainty,
    `${path}.mastery_uncertainty`,
  );
  const retrievability = asBoundedNumber(raw.retrievability, `${path}.retrievability`);
  const stabilityDays = asNonNegativeNumber(raw.stability_days, `${path}.stability_days`);
  const firstExposedAt = asTimestamp(raw.first_exposed_at, `${path}.first_exposed_at`);
  const lastExposedAt = asTimestamp(raw.last_exposed_at, `${path}.last_exposed_at`);
  const lastEvidenceAt = asTimestamp(raw.last_evidence_at, `${path}.last_evidence_at`, true);
  const nextDueAt = asTimestamp(raw.next_due_at, `${path}.next_due_at`, true);

  if (exposedLessonCount > occurrenceCount) {
    throw new Error(`Invalid characters: ${path} has more lessons than appearances`);
  }
  if (distinctWordContexts > occurrenceCount) {
    throw new Error(`Invalid characters: ${path} has more word contexts than appearances`);
  }
  if (countedRevealSessions > rawRevealCount) {
    throw new Error(`Invalid characters: ${path} has more counted checks than raw checks`);
  }
  if (inferredFailureSessions > countedRevealSessions) {
    throw new Error(`Invalid characters: ${path} has more difficulty sessions than checks`);
  }
  if (Date.parse(firstExposedAt) > Date.parse(lastExposedAt)) {
    throw new Error(`Invalid characters: ${path} was last seen before it was first seen`);
  }
  if (lastEvidenceAt && Date.parse(lastEvidenceAt) < Date.parse(firstExposedAt)) {
    throw new Error(`Invalid characters: ${path} has evidence before its first exposure`);
  }

  return {
    character: asHanCharacter(raw.character, `${path}.character`),
    occurrence_count: occurrenceCount,
    exposed_lesson_count: exposedLessonCount,
    raw_reveal_count: rawRevealCount,
    counted_reveal_sessions: countedRevealSessions,
    distinct_word_contexts: distinctWordContexts,
    qualified_exposures: qualifiedExposures,
    inferred_failure_sessions: inferredFailureSessions,
    inferred_failure_mass: inferredFailureMass,
    direct_successes: directSuccesses,
    direct_failures: directFailures,
    mastery,
    mastery_uncertainty: masteryUncertainty,
    retrievability,
    stability_days: stabilityDays,
    first_exposed_at: firstExposedAt,
    last_exposed_at: lastExposedAt,
    last_evidence_at: lastEvidenceAt,
    next_due_at: nextDueAt,
  };
}

export function parseCharacters(value: unknown): CharactersPayload {
  const response = asRecord(value, "response");
  if (!Array.isArray(response.characters)) {
    throw new Error("Invalid characters: characters must be a list");
  }
  const characters = response.characters.map(parseCharacter);
  if (new Set(characters.map((entry) => entry.character)).size !== characters.length) {
    throw new Error("Invalid characters: characters must be unique");
  }
  return {
    learning_language: asText(response.learning_language, "learning_language"),
    translation_language: asText(response.translation_language, "translation_language"),
    characters,
  };
}
