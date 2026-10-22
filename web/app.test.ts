import { describe, expect, it } from "vitest";

import { EVENT_TYPES, type LearningEvent, parseReader } from "./contracts.js";
import {
  DurableEventStore,
  EVENT_QUEUE_STORAGE_KEY,
  eventQueueStorageKey,
  flushStoredEvents,
  LESSON_SESSION_STORAGE_KEY,
  LessonSessionStore,
  lessonSessionStorageKey,
  migrateLegacyPersistence,
} from "./contracts.persistence.js";

class MemoryStorage {
  private readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

function event(eventId: string, type: LearningEvent["type"] = "term.revealed"): LearningEvent {
  return {
    event_id: eventId,
    session_id: "session-1",
    lesson_id: 7,
    type,
    occurred_at: "2026-07-10T08:00:00.000Z",
    payload: type === "term.revealed" ? { term_key: "es:tren:NOUN" } : {},
  };
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve = (): void => undefined;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const profile = {
  learning_language: "es-ES",
  translation_language: "en",
  level: "A1",
  difficulty: 0.15,
  interests: ["science"],
  preferences: {},
};

const progress = {
  session_id: null,
  started: false,
  completed: false,
  rating: null,
  revealed_term_keys: [],
  revealed_sentence_keys: [],
  full_translation_revealed: false,
};

const termBands = { "es:manana:NOUN": "uncertain" };

const lesson = {
  schema_version: 1,
  key: "unicode",
  title: "Unicode",
  learning_language: "es-ES",
  translation_language: "en",
  topic: "science",
  level: "A1",
  difficulty: 0.15,
  blocks: [
    {
      key: "block-1",
      sentences: [
        {
          key: "sentence-1",
          translation: "Tomorrow 🧠",
          runs: [
            { text: "<script>alert('x')</script> 🌍 " },
            {
              text: "mañana🧠",
              term: {
                key: "es:manana:NOUN",
                lemma: "mañana",
                pos: "NOUN",
                gloss: "<b>tomorrow</b>",
                frequency_rank: 5,
              },
            },
          ],
        },
      ],
    },
  ],
  target_term_keys: ["es:manana:NOUN"],
  metadata: { series: "test" },
};

describe("reader contract", () => {
  it("accepts the empty state", () => {
    const parsed = parseReader({
      profile,
      lesson_id: null,
      lesson: null,
      term_bands: {},
      progress,
    });

    expect(parsed.lesson).toBeNull();
    expect(parsed.profile.learning_language).toBe("es-ES");
  });

  it("preserves Unicode and markup-looking text as data", () => {
    const parsed = parseReader({
      profile,
      lesson_id: 1,
      lesson,
      term_bands: termBands,
      progress,
    });
    const runs = parsed.lesson?.blocks[0]?.sentences[0]?.runs;

    expect(runs?.[0]?.text).toBe("<script>alert('x')</script> 🌍 ");
    expect(runs?.[1]?.text).toBe("mañana🧠");
    expect(runs?.[1]?.term?.gloss).toBe("<b>tomorrow</b>");
  });

  it("rejects partial and out-of-range states", () => {
    expect(() =>
      parseReader({ profile, lesson_id: 1, lesson: null, term_bands: {}, progress }),
    ).toThrow("lesson_id and lesson");
    expect(() =>
      parseReader({
        profile: { ...profile, difficulty: 1.01 },
        lesson_id: null,
        lesson: null,
        term_bands: {},
        progress,
      }),
    ).toThrow("between 0 and 1");
  });
});

describe("durable event storage", () => {
  it("restores every event type from the shared runtime source of truth", () => {
    const storage = new MemoryStorage();
    storage.setItem(
      EVENT_QUEUE_STORAGE_KEY,
      JSON.stringify(EVENT_TYPES.map((type, index) => event(`event-${index}`, type))),
    );

    expect(
      new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY).snapshot().map((item) => item.type),
    ).toEqual(EVENT_TYPES);
  });

  it("keeps in-flight events until the request succeeds", async () => {
    const storage = new MemoryStorage();
    const store = new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY);
    store.append(event("first"));
    const request = deferred();

    const flush = flushStoredEvents(store, () => request.promise);
    expect(storage.getItem(EVENT_QUEUE_STORAGE_KEY)).toContain("first");

    request.resolve();
    await flush;
    expect(storage.getItem(EVENT_QUEUE_STORAGE_KEY)).toBeNull();
  });

  it("restores failed events and does not drop events added during a request", async () => {
    const storage = new MemoryStorage();
    const store = new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY);
    store.append(event("first"));

    await expect(
      flushStoredEvents(store, () => Promise.reject(new Error("offline"))),
    ).rejects.toThrow("offline");
    expect(
      new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY)
        .snapshot()
        .map((item) => item.event_id),
    ).toEqual(["first"]);

    const request = deferred();
    const flush = flushStoredEvents(store, () => request.promise);
    store.append(event("second"));
    request.resolve();
    await flush;

    expect(
      new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY)
        .snapshot()
        .map((item) => item.event_id),
    ).toEqual(["second"]);
  });

  it("clears persisted and in-memory events for a reset profile", () => {
    const storage = new MemoryStorage();
    const store = new DurableEventStore(storage, EVENT_QUEUE_STORAGE_KEY);
    store.append(event("old"));

    store.clear();

    expect(store.snapshot()).toEqual([]);
    expect(storage.getItem(EVENT_QUEUE_STORAGE_KEY)).toBeNull();
  });
});

describe("lesson session storage", () => {
  it("reuses a lesson session until its matching completion succeeds", () => {
    const storage = new MemoryStorage();
    const sessions = new LessonSessionStore(storage, LESSON_SESSION_STORAGE_KEY);
    const first = sessions.getOrCreate(7, undefined, () => "session-1");

    expect(
      new LessonSessionStore(storage, LESSON_SESSION_STORAGE_KEY).getOrCreate(
        7,
        undefined,
        () => "unused",
      ),
    ).toBe(first);
    sessions.complete(7, "another-session");
    expect(sessions.getOrCreate(7, undefined, () => "unused")).toBe(first);

    sessions.complete(7, first);
    expect(sessions.getOrCreate(7, undefined, () => "session-2")).toBe("session-2");
  });

  it("clears persisted and in-memory sessions for a reset profile", () => {
    const storage = new MemoryStorage();
    const sessions = new LessonSessionStore(storage, LESSON_SESSION_STORAGE_KEY);
    sessions.getOrCreate(7, undefined, () => "old-session");

    sessions.clear();

    expect(sessions.getOrCreate(7, undefined, () => "new-session")).toBe("new-session");
  });
});

describe("profile-scoped persistence", () => {
  it("uses separate keys and migrates the legacy Spanish state", () => {
    const storage = new MemoryStorage();
    storage.setItem(EVENT_QUEUE_STORAGE_KEY, JSON.stringify([event("legacy")]));
    storage.setItem(LESSON_SESSION_STORAGE_KEY, JSON.stringify({ "7": "session-1" }));

    migrateLegacyPersistence(storage, "es-es");

    expect(storage.getItem(EVENT_QUEUE_STORAGE_KEY)).toBeNull();
    expect(storage.getItem(LESSON_SESSION_STORAGE_KEY)).toBeNull();
    expect(new DurableEventStore(storage, eventQueueStorageKey("es-es")).snapshot()).toHaveLength(
      1,
    );
    expect(
      new LessonSessionStore(storage, lessonSessionStorageKey("es-es")).getOrCreate(
        7,
        undefined,
        () => "unused",
      ),
    ).toBe("session-1");
    expect(eventQueueStorageKey("es-es")).not.toBe(eventQueueStorageKey("zh-hant"));
  });

  it("does not assign legacy state to another profile", () => {
    const storage = new MemoryStorage();
    storage.setItem(EVENT_QUEUE_STORAGE_KEY, JSON.stringify([event("legacy")]));

    migrateLegacyPersistence(storage, "zh-hant");

    expect(storage.getItem(EVENT_QUEUE_STORAGE_KEY)).not.toBeNull();
    expect(storage.getItem(eventQueueStorageKey("zh-hant"))).toBeNull();
  });
});
