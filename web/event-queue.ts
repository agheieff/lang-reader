/** Learning events: durable local queue, batched delivery, and a last-chance beacon on exit. */

import type { LearningEvent } from "./contracts.js";
import { type DurableEventStore, flushStoredEvents } from "./contracts.persistence.js";

export interface EventQueueOptions {
  endpoint: string;
  readOnly: boolean;
  onFlushError(error: unknown): void;
  onLessonCompleted(lessonId: number, sessionId: string): void;
}

export class EventQueue {
  private timer: number | null = null;
  private flushing: Promise<void> | null = null;

  constructor(
    private readonly store: DurableEventStore,
    private readonly options: EventQueueOptions,
  ) {}

  add(event: LearningEvent, immediate = false): void {
    if (this.options.readOnly) return;
    this.store.append(event);
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = window.setTimeout(
      () => {
        this.timer = null;
        void this.flush().catch(this.options.onFlushError);
      },
      immediate ? 0 : 450,
    );
  }

  flush(): Promise<void> {
    if (this.options.readOnly) return Promise.resolve();
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    if (!this.flushing) {
      this.flushing = this.drain().finally(() => {
        this.flushing = null;
      });
    }
    return this.flushing;
  }

  clear(): void {
    if (this.options.readOnly) return;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.store.clear();
  }

  sendOnExit(): void {
    if (this.options.readOnly) return;
    const pending = this.store.snapshot();
    if (pending.length === 0) return;
    const body = JSON.stringify({ events: pending });
    navigator.sendBeacon(this.options.endpoint, new Blob([body], { type: "application/json" }));
  }

  private async drain(): Promise<void> {
    while (this.store.snapshot().length > 0) {
      await flushStoredEvents(this.store, (batch) => this.send(batch));
    }
  }

  private async send(events: readonly LearningEvent[]): Promise<void> {
    const response = await fetch(this.options.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events }),
    });
    if (!response.ok) throw new Error(`Event sync failed (${response.status})`);
    for (const event of events) {
      if (event.type === "lesson.completed") {
        this.options.onLessonCompleted(event.lesson_id, event.session_id);
      }
    }
  }
}
