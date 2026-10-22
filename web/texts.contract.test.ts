import { describe, expect, it } from "vitest";

import { parseTextLibrary, parseTextPreview, parseTextRequest } from "./texts.contract.js";

const text = {
  id: 4,
  key: "park-morning",
  title: "清早的公园",
  topic: "daily life",
  level: "A2",
  difficulty: 0.22,
  imported_at: "2026-07-10T12:00:00Z",
  status: "in_progress",
  queue_position: 1,
  opened_at: "2026-07-10T12:01:00Z",
  last_completed_at: null,
  session_count: 1,
  completion_count: 0,
  rating: null,
  lexical_token_count: 284,
};

const request = {
  task_id: 8,
  request_id: "request-1",
  topic: "a train journey",
  state: "running",
  created_at: "2026-07-10T12:00:00Z",
  updated_at: "2026-07-10T12:02:00Z",
  error: null,
  lesson_id: null,
};

const preparation = {
  task_id: 9,
  state: "running",
  request_kind: "queue_fill",
  requested_topic: null,
  waiting_for_capacity: false,
  created_at: "2026-07-10T12:03:00Z",
  updated_at: "2026-07-10T12:04:00Z",
};

describe("parseTextLibrary", () => {
  it("parses queue state, active preparations, reading history, and topic requests", () => {
    const parsed = parseTextLibrary({
      learning_language: "zh-Hans",
      translation_language: "en",
      texts: [text],
      preparations: [preparation],
      requests: [request],
    });

    expect(parsed.texts[0]).toMatchObject({
      id: 4,
      status: "in_progress",
      queue_position: 1,
      opened_at: "2026-07-10T12:01:00Z",
    });
    expect(parsed.preparations[0]).toMatchObject({
      task_id: 9,
      state: "running",
      request_kind: "queue_fill",
      requested_topic: null,
      waiting_for_capacity: false,
    });
    expect(parsed.requests[0]).toMatchObject({ state: "running", error: undefined });
  });

  it("requires the capacity flag on a preparation", () => {
    const { waiting_for_capacity: _omitted, ...withoutFlag } = preparation;
    expect(() =>
      parseTextLibrary({
        learning_language: "zh-Hans",
        translation_language: "en",
        texts: [],
        preparations: [withoutFlag],
        requests: [],
      }),
    ).toThrow("waiting_for_capacity must be a boolean");
  });

  it("keeps skipped texts recoverable with their disposition timestamp", () => {
    const parsed = parseTextLibrary({
      learning_language: "zh-Hans",
      translation_language: "en",
      texts: [
        {
          ...text,
          status: "skipped",
          queue_position: null,
          skipped_at: "2026-07-10T12:03:00Z",
        },
      ],
      preparations: [],
      requests: [],
    });

    expect(parsed.texts[0]).toMatchObject({
      status: "skipped",
      queue_position: null,
      skipped_at: "2026-07-10T12:03:00Z",
    });
  });

  it("rejects malformed states, dates, and duplicate identities", () => {
    expect(() =>
      parseTextLibrary({
        learning_language: "zh-Hans",
        translation_language: "en",
        texts: [{ ...text, status: "lost" }],
        preparations: [],
        requests: [],
      }),
    ).toThrow(/status/);
    expect(() =>
      parseTextLibrary({
        learning_language: "zh-Hans",
        translation_language: "en",
        texts: [{ ...text, imported_at: "yesterday-ish" }],
        preparations: [],
        requests: [],
      }),
    ).toThrow(/timestamp/);
    expect(() =>
      parseTextLibrary({
        learning_language: "zh-Hans",
        translation_language: "en",
        texts: [{ ...text, status: "skipped", queue_position: null, skipped_at: "later" }],
        preparations: [],
        requests: [],
      }),
    ).toThrow(/timestamp/);
    expect(() =>
      parseTextLibrary({
        learning_language: "zh-Hans",
        translation_language: "en",
        texts: [text, text],
        preparations: [],
        requests: [],
      }),
    ).toThrow(/unique/);
  });

  it("validates preparation states, topics, and unique task IDs", () => {
    const payload = {
      learning_language: "zh-Hans",
      translation_language: "en",
      texts: [],
      preparations: [preparation],
      requests: [],
    };
    expect(() =>
      parseTextLibrary({
        ...payload,
        preparations: [{ ...preparation, state: "completed" }],
      }),
    ).toThrow(/state/);
    expect(() =>
      parseTextLibrary({
        ...payload,
        preparations: [
          {
            ...preparation,
            request_kind: "topic_request",
            requested_topic: null,
          },
        ],
      }),
    ).toThrow(/requested_topic/);
    expect(() =>
      parseTextLibrary({
        ...payload,
        preparations: [preparation, preparation],
      }),
    ).toThrow(/unique/);
  });
});

describe("text request and preview contracts", () => {
  it("parses a request returned by the create endpoint", () => {
    expect(parseTextRequest({ ...request, state: "completed", lesson_id: 9 })).toMatchObject({
      state: "completed",
      lesson_id: 9,
    });
  });

  it("parses a read-only lesson without requiring progress or a session", () => {
    const response = {
      lesson_id: 4,
      lesson: {
        schema_version: 1,
        key: "park-morning",
        title: "清早的公园",
        learning_language: "zh-Hans",
        translation_language: "en",
        topic: "daily life",
        level: "A2",
        difficulty: 0.22,
        blocks: [{ key: "b1", sentences: [{ key: "s1", runs: [{ text: "早上。" }] }] }],
        target_term_keys: [],
        metadata: {},
      },
      term_bands: { morning: "expected" },
    };
    const parsed = parseTextPreview({
      ...response,
      comfortable_grammar_construction_keys: ["zh:measure-word-phrase"],
    });

    expect(parsed.lesson_id).toBe(4);
    expect(parsed.term_bands).toEqual({ morning: "expected" });
    expect(parsed.comfortable_grammar_construction_keys).toEqual(["zh:measure-word-phrase"]);
    expect(parsed).not.toHaveProperty("progress");
    expect(parseTextPreview(response).comfortable_grammar_construction_keys).toEqual([]);
  });
});
