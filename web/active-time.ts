import type { ActiveTimeStore } from "./contracts.persistence.js";

const DEFAULT_READING_ACTIVITY_WINDOW_MS = 5_000;
// Touch reading has no continuous pointer movement: taps and scrolls are sparse, so each one keeps
// the session active for longer than a mouse movement does.
export const TOUCH_READING_ACTIVITY_WINDOW_MS = 15_000;

export interface ActiveTimerOptions {
  clock: () => number;
  isHidden: () => boolean;
  render: (activeMs: number) => void;
  activityWindowMs?: number;
}

export class ActiveTimer {
  private activeMs = 0;
  private lessonId: number | null = null;
  private sessionId = "";
  private lastTick: number;
  private activeUntil: number;
  private hidden: boolean;
  private paused = false;
  private readonly activityWindowMs: number;

  constructor(
    private readonly store: ActiveTimeStore,
    private readonly options: ActiveTimerOptions,
  ) {
    this.lastTick = options.clock();
    this.activeUntil = this.lastTick;
    this.hidden = options.isHidden();
    this.activityWindowMs = options.activityWindowMs ?? DEFAULT_READING_ACTIVITY_WINDOW_MS;
    if (!Number.isFinite(this.activityWindowMs) || this.activityWindowMs <= 0) {
      throw new Error("The reading activity window must be finite and positive");
    }
    this.options.render(0);
  }

  start(lessonId: number, sessionId: string): void {
    if (!sessionId.trim()) throw new Error("A lesson session ID must not be blank");
    if (this.lessonId === lessonId && this.sessionId === sessionId) {
      this.resume();
      return;
    }
    if (this.lessonId !== null) this.advance();
    this.lessonId = lessonId;
    this.sessionId = sessionId;
    this.activeMs = this.store.resume(lessonId, sessionId);
    this.paused = false;
    this.lastTick = this.options.clock();
    this.activeUntil = this.lastTick;
    this.hidden = this.options.isHidden();
    this.options.render(this.activeMs);
  }

  noteMovement(windowMs: number = this.activityWindowMs): void {
    const now = this.options.clock();
    this.capture(now);
    if (this.lessonId !== null && !this.paused && !this.hidden) {
      // A short mouse window never truncates a longer touch window that is still open.
      this.activeUntil = Math.max(this.activeUntil, now + windowMs);
    }
  }

  visibilityChanged(): void {
    this.advance();
    if (this.hidden) this.activeUntil = this.lastTick;
  }

  tick(): void {
    this.advance();
  }

  pagehide(): void {
    this.advance();
    this.activeUntil = this.lastTick;
  }

  save(): void {
    if (this.lessonId === null) return;
    this.store.save(this.lessonId, this.sessionId, this.activeMs);
  }

  pause(): void {
    if (this.lessonId === null || this.paused) return;
    this.advance();
    this.paused = true;
  }

  resume(): void {
    if (this.lessonId === null || !this.paused) return;
    const now = this.options.clock();
    this.paused = false;
    this.lastTick = now;
    this.activeUntil = now;
    this.hidden = this.options.isHidden();
  }

  reset(): void {
    if (this.lessonId === null) return;
    const now = this.options.clock();
    this.activeMs = 0;
    this.lastTick = now;
    this.activeUntil = now;
    this.save();
    this.options.render(0);
  }

  seconds(): number {
    this.tick();
    return Math.max(0, Math.round(this.activeMs / 1_000));
  }

  complete(lessonId: number, sessionId: string): void {
    this.store.complete(lessonId, sessionId);
    if (this.lessonId !== lessonId || this.sessionId !== sessionId) return;
    this.lessonId = null;
    this.sessionId = "";
    this.activeMs = 0;
    this.paused = false;
    this.activeUntil = this.options.clock();
    this.options.render(0);
  }

  private advance(): void {
    const now = this.options.clock();
    this.capture(now);
    this.save();
    this.options.render(this.activeMs);
  }

  private capture(now: number): void {
    if (this.lessonId !== null && !this.paused && !this.hidden) {
      const countableUntil = Math.min(now, this.activeUntil);
      this.activeMs += Math.max(countableUntil - this.lastTick, 0);
    }
    this.lastTick = now;
    this.hidden = this.options.isHidden();
  }
}
