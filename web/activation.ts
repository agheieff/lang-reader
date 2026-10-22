export const STARTING_POINTS = [
  "unsure",
  "complete_beginner",
  "basic_phrases",
  "simple_texts",
  "general_reading",
  "complex_texts",
  "advanced_reading",
  "near_native",
] as const;

export type StartingPoint = (typeof STARTING_POINTS)[number];
export const QUESTIONNAIRE_CONFIDENCES = ["low", "medium", "high"] as const;
export type QuestionnaireConfidence = (typeof QUESTIONNAIRE_CONFIDENCES)[number];

export interface ProfileActivation {
  active: boolean;
  learning_language: string;
  activated_at: string | null;
  questionnaire_completed: boolean;
  starting_point: StartingPoint | null;
  confidence: QuestionnaireConfidence | null;
}

export interface ProfileActivationUpdate {
  starting_point: StartingPoint;
  confidence: QuestionnaireConfidence;
  interests: string[];
  text_length?: number | undefined;
}

type UnknownRecord = Record<string, unknown>;

function record(value: unknown): UnknownRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Invalid activation response");
  }
  return value as UnknownRecord;
}

export function parseProfileActivation(value: unknown): ProfileActivation {
  const payload = record(value);
  if (typeof payload.active !== "boolean" || typeof payload.questionnaire_completed !== "boolean") {
    throw new Error("Invalid activation response status");
  }
  if (typeof payload.learning_language !== "string" || !payload.learning_language.trim()) {
    throw new Error("Invalid activation response language");
  }
  const startingPoint = payload.starting_point;
  if (startingPoint !== null && !STARTING_POINTS.includes(startingPoint as StartingPoint)) {
    throw new Error("Invalid activation starting point");
  }
  const confidence = payload.confidence;
  if (
    confidence !== null &&
    !QUESTIONNAIRE_CONFIDENCES.includes(confidence as QuestionnaireConfidence)
  ) {
    throw new Error("Invalid activation confidence");
  }
  const activatedAt = payload.activated_at;
  if (
    activatedAt !== null &&
    (typeof activatedAt !== "string" || !Number.isFinite(Date.parse(activatedAt)))
  ) {
    throw new Error("Invalid activation timestamp");
  }
  return {
    active: payload.active,
    learning_language: payload.learning_language,
    activated_at: activatedAt as string | null,
    questionnaire_completed: payload.questionnaire_completed,
    starting_point: startingPoint as StartingPoint | null,
    confidence: confidence as QuestionnaireConfidence | null,
  };
}

export function parseInterestInput(value: string): string[] {
  return [
    ...new Set(
      value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ].slice(0, 12);
}
