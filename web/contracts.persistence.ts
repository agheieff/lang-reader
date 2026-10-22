import { EVENT_TYPES, type EventType, type JsonRecord, type LearningEvent } from "./contracts.js";

import { STORAGE_PREFIX } from "./storage-prefix.js";

export const EVENT_QUEUE_STORAGE_KEY = `${STORAGE_PREFIX}event-queue:v1`;
export const LESSON_SESSION_STORAGE_KEY = `${STORAGE_PREFIX}lesson-sessions:v1`;

export function eventQueueStorageKey(profileId: string): string {
  return `${STORAGE_PREFIX}${profileId}:event-queue:v2`;
}

export function lessonSessionStorageKey(profileId: string): string {
  return `${STORAGE_PREFIX}${profileId}:lesson-sessions:v2`;
}

export function activeTimeStorageKey(profileId: string): string {
  return `${STORAGE_PREFIX}${profileId}:active-time:v1`;
}

export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const eventTypes = new Set<EventType>(EVENT_TYPES);

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function storedEvent(value: unknown, index: number): LearningEvent {
  if (!isRecord(value)) throw new Error(`Invalid stored event at index ${index}`);
  if (
    typeof value.event_id !== "string" ||
    typeof value.session_id !== "string" ||
    !Number.isInteger(value.lesson_id) ||
    (value.lesson_id as number) < 1 ||
    typeof value.type !== "string" ||
    !eventTypes.has(value.type as EventType) ||
    typeof value.occurred_at !== "string" ||
    !isRecord(value.payload)
  ) {
    throw new Error(`Invalid stored event at index ${index}`);
  }
  return {
    event_id: value.event_id,
    session_id: value.session_id,
    lesson_id: value.lesson_id as number,
    type: value.type as EventType,
    occurred_at: value.occurred_at,
    payload: value.payload,
  };
}

function readEvents(storage: KeyValueStorage, key: string): LearningEvent[] {
  const raw = storage.getItem(key);
  if (raw === null) return [];
  const value: unknown = JSON.parse(raw);
  if (!Array.isArray(value)) throw new Error("Invalid stored event queue");
  return value.map(storedEvent);
}

export function migrateLegacyPersistence(
  storage: KeyValueStorage,
  profileId: string,
  legacyProfileId = "es-es",
): void {
  if (profileId !== legacyProfileId) return;
  const queueKey = eventQueueStorageKey(profileId);
  const legacyEvents = readEvents(storage, EVENT_QUEUE_STORAGE_KEY);
  if (legacyEvents.length > 0) {
    const merged = new Map(
      [...readEvents(storage, queueKey), ...legacyEvents].map((event) => [event.event_id, event]),
    );
    storage.setItem(queueKey, JSON.stringify([...merged.values()]));
  }
  storage.removeItem(EVENT_QUEUE_STORAGE_KEY);

  const sessionKey = lessonSessionStorageKey(profileId);
  const legacySessions = readSessions(storage, LESSON_SESSION_STORAGE_KEY);
  const sessions = { ...legacySessions, ...readSessions(storage, sessionKey) };
  if (Object.keys(sessions).length > 0) storage.setItem(sessionKey, JSON.stringify(sessions));
  storage.removeItem(LESSON_SESSION_STORAGE_KEY);
}

export class DurableEventStore {
  private events: LearningEvent[];

  constructor(
    private readonly storage: KeyValueStorage,
    private readonly key: string,
  ) {
    this.events = readEvents(storage, key);
  }

  snapshot(): LearningEvent[] {
    return [...this.events];
  }

  append(event: LearningEvent): void {
    this.replace([...this.events, event]);
  }

  acknowledge(events: readonly LearningEvent[]): void {
    const acknowledged = new Set(events.map((event) => event.event_id));
    this.replace(this.events.filter((event) => !acknowledged.has(event.event_id)));
  }

  clear(): void {
    this.replace([]);
  }

  private replace(events: LearningEvent[]): void {
    if (events.length === 0) this.storage.removeItem(this.key);
    else this.storage.setItem(this.key, JSON.stringify(events));
    this.events = events;
  }
}

export async function flushStoredEvents(
  store: DurableEventStore,
  send: (events: readonly LearningEvent[]) => Promise<void>,
): Promise<LearningEvent[]> {
  const batch = store.snapshot();
  if (batch.length === 0) return [];
  await send(batch);
  store.acknowledge(batch);
  return batch;
}

function readSessions(storage: KeyValueStorage, key: string): Record<string, string> {
  const raw = storage.getItem(key);
  if (raw === null) return {};
  const value: unknown = JSON.parse(raw);
  if (!isRecord(value)) throw new Error("Invalid stored lesson sessions");
  const sessions: Record<string, string> = {};
  for (const [lessonId, sessionId] of Object.entries(value)) {
    if (!/^\d+$/.test(lessonId) || typeof sessionId !== "string" || !sessionId.trim()) {
      throw new Error("Invalid stored lesson sessions");
    }
    sessions[lessonId] = sessionId;
  }
  return sessions;
}

export class LessonSessionStore {
  private sessions: Record<string, string>;

  constructor(
    private readonly storage: KeyValueStorage,
    private readonly key: string,
  ) {
    this.sessions = readSessions(storage, key);
  }

  getOrCreate(lessonId: number, preferred: string | undefined, create: () => string): string {
    const key = String(lessonId);
    if (preferred) {
      if (this.sessions[key] !== preferred) this.replace({ ...this.sessions, [key]: preferred });
      return preferred;
    }
    const existing = this.sessions[key];
    if (existing) return existing;
    const sessionId = create();
    if (!sessionId.trim()) throw new Error("A lesson session ID must not be blank");
    this.replace({ ...this.sessions, [key]: sessionId });
    return sessionId;
  }

  complete(lessonId: number, sessionId: string): void {
    const key = String(lessonId);
    if (this.sessions[key] !== sessionId) return;
    const next = { ...this.sessions };
    delete next[key];
    this.replace(next);
  }

  clear(): void {
    this.replace({});
  }

  private replace(sessions: Record<string, string>): void {
    if (Object.keys(sessions).length === 0) this.storage.removeItem(this.key);
    else this.storage.setItem(this.key, JSON.stringify(sessions));
    this.sessions = sessions;
  }
}

interface StoredActiveTime {
  session_id: string;
  active_ms: number;
}

function readActiveTimes(storage: KeyValueStorage, key: string): Record<string, StoredActiveTime> {
  const raw = storage.getItem(key);
  if (raw === null) return {};
  const value: unknown = JSON.parse(raw);
  if (!isRecord(value)) throw new Error("Invalid stored active reading times");
  const activeTimes: Record<string, StoredActiveTime> = {};
  for (const [lessonId, entry] of Object.entries(value)) {
    if (
      !/^\d+$/.test(lessonId) ||
      !isRecord(entry) ||
      typeof entry.session_id !== "string" ||
      !entry.session_id.trim() ||
      typeof entry.active_ms !== "number" ||
      !Number.isFinite(entry.active_ms) ||
      entry.active_ms < 0
    ) {
      throw new Error("Invalid stored active reading times");
    }
    activeTimes[lessonId] = {
      session_id: entry.session_id,
      active_ms: entry.active_ms,
    };
  }
  return activeTimes;
}

export class ActiveTimeStore {
  private activeTimes: Record<string, StoredActiveTime>;

  constructor(
    private readonly storage: KeyValueStorage,
    private readonly key: string,
  ) {
    this.activeTimes = readActiveTimes(storage, key);
  }

  resume(lessonId: number, sessionId: string): number {
    const entry = this.activeTimes[String(lessonId)];
    return entry?.session_id === sessionId ? entry.active_ms : 0;
  }

  save(lessonId: number, sessionId: string, activeMs: number): void {
    if (!sessionId.trim()) throw new Error("A lesson session ID must not be blank");
    if (!Number.isFinite(activeMs) || activeMs < 0) {
      throw new Error("Active reading time must be a finite, non-negative number");
    }
    this.replace({
      ...this.activeTimes,
      [String(lessonId)]: { session_id: sessionId, active_ms: activeMs },
    });
  }

  complete(lessonId: number, sessionId: string): void {
    const key = String(lessonId);
    if (this.activeTimes[key]?.session_id !== sessionId) return;
    const next = { ...this.activeTimes };
    delete next[key];
    this.replace(next);
  }

  clear(): void {
    this.replace({});
  }

  private replace(activeTimes: Record<string, StoredActiveTime>): void {
    if (Object.keys(activeTimes).length === 0) this.storage.removeItem(this.key);
    else this.storage.setItem(this.key, JSON.stringify(activeTimes));
    this.activeTimes = activeTimes;
  }
}
