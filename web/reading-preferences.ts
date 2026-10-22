import { createJsonContract } from "./json-contract.js";

export interface ReadingPreferences {
  text: string;
  revision_id: number | null;
  source: "default" | "user" | "agent";
  updated_at: string | null;
  last_agent_reason: string | null;
  pending_messages: { text: string; created_at: string }[];
}

const { record, text, nullableInteger, nullableText, timestamp, oneOf } = createJsonContract(
  "Invalid reading preferences",
  { allowBlankText: true },
);

export function parseReadingPreferences(value: unknown): ReadingPreferences {
  const item = record(value, "preferences");
  const pending = item.pending_messages;
  if (!Array.isArray(pending)) throw new Error("Invalid reading preferences: pending_messages");
  return {
    text: text(item.text, "text"),
    revision_id: nullableInteger(item.revision_id, "revision_id", 1),
    source: oneOf(item.source, ["default", "user", "agent"] as const, "source"),
    updated_at: timestamp(item.updated_at, "updated_at", true) ?? null,
    last_agent_reason: nullableText(item.last_agent_reason, "last_agent_reason"),
    pending_messages: pending.map((message, index) => {
      const entry = record(message, `pending_messages[${index}]`);
      return {
        text: text(entry.text, `pending_messages[${index}].text`),
        created_at: timestamp(entry.created_at, `pending_messages[${index}].created_at`),
      };
    }),
  };
}

export function contentExperiment(
  metadata: Record<string, unknown>,
): { kind: string; question: string } | null {
  const plan = metadata.content_plan;
  const move =
    typeof plan === "object" && plan !== null && !Array.isArray(plan)
      ? (plan as Record<string, unknown>).move
      : undefined;
  const question = metadata.content_hypothesis;
  if ((move !== "variation" && move !== "new") || typeof question !== "string" || !question) {
    return null;
  }
  return { kind: move === "new" ? "Testing a new subject" : "Testing a variation", question };
}
