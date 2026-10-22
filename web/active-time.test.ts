import { describe, expect, it } from "vitest";

import { ActiveTimer, TOUCH_READING_ACTIVITY_WINDOW_MS } from "./active-time.js";
import {
  ActiveTimeStore,
  activeTimeStorageKey,
  type KeyValueStorage,
} from "./contracts.persistence.js";

class MemoryStorage implements KeyValueStorage {
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

describe("active reading time persistence", () => {
  it("resumes cumulative time for only the same profile, lesson, and session", () => {
    const storage = new MemoryStorage();
    const spanishKey = activeTimeStorageKey("es-es");
    new ActiveTimeStore(storage, spanishKey).save(7, "session-1", 12_400);

    const restored = new ActiveTimeStore(storage, spanishKey);
    expect(restored.resume(7, "session-1")).toBe(12_400);
    expect(restored.resume(7, "session-2")).toBe(0);
    expect(restored.resume(8, "session-1")).toBe(0);
    expect(
      new ActiveTimeStore(storage, activeTimeStorageKey("zh-hant")).resume(7, "session-1"),
    ).toBe(0);
  });

  it("clears stored active time for a reset profile", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    const store = new ActiveTimeStore(storage, key);
    store.save(7, "old-session", 12_400);

    store.clear();

    expect(store.resume(7, "old-session")).toBe(0);
    expect(storage.getItem(key)).toBeNull();
  });

  it("saves on ticks and clears only a matching confirmed completion", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    let rendered = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: (activeMs) => {
        rendered = activeMs;
      },
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 1_200;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(1_200);

    now = 0;
    const restored = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: (activeMs) => {
        rendered = activeMs;
      },
    });
    restored.start(7, "session-1");
    expect(rendered).toBe(1_200);
    restored.noteMovement();
    now = 800;
    restored.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(2_000);

    now = 1_400;
    restored.pagehide();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(2_600);

    restored.complete(7, "another-session");
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(2_600);
    restored.complete(7, "session-1");
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(0);
    expect(rendered).toBe(0);
  });

  it("resets and persists the active lesson session immediately", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const renders: number[] = [];
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: (activeMs) => {
        renders.push(activeMs);
      },
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 1_200;
    timer.tick();
    now = 2_000;
    timer.reset();

    expect(renders.at(-1)).toBe(0);
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(0);
    expect(JSON.parse(storage.getItem(key) ?? "{}")).toEqual({
      "7": {
        session_id: "session-1",
        active_ms: 0,
      },
    });

    now = 2_800;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(0);

    timer.noteMovement();
    now = 3_600;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(800);
  });

  it("pauses without charging the wait and resumes the same persisted session", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 1_200;
    timer.pause();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(1_200);

    now = 9_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(1_200);

    timer.resume();
    now = 9_800;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(1_200);

    timer.noteMovement();
    now = 10_600;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(2_000);
  });

  it("resumes the same session but waits for movement before counting again", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 500;
    timer.pause();
    now = 5_000;
    timer.start(7, "session-1");
    now = 5_400;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(500);

    timer.noteMovement();
    now = 5_800;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(900);
    expect(timer.seconds()).toBe(1);
  });

  it("counts at most five seconds after movement, then waits for the next movement", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 10_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(5_000);

    now = 20_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(5_000);

    timer.noteMovement();
    now = 27_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(10_000);
  });

  it("keeps touch reading active longer and never shortens it with a mouse window", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement(TOUCH_READING_ACTIVITY_WINDOW_MS);
    now = 2_000;
    timer.noteMovement();
    now = 30_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(15_000);
  });

  it("extends a still-active window without counting an idle gap", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 4_000;
    timer.noteMovement();
    now = 9_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(9_000);

    now = 20_000;
    timer.noteMovement();
    now = 21_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(10_000);
  });

  it("reports zero until the first movement", () => {
    const storage = new MemoryStorage();
    let now = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, activeTimeStorageKey("es-es")), {
      clock: () => now,
      isHidden: () => false,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    now = 30_000;

    expect(timer.seconds()).toBe(0);
  });

  it("requires fresh movement after returning to a visible tab", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let now = 0;
    let hidden = false;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => now,
      isHidden: () => hidden,
      render: () => undefined,
    });

    timer.start(7, "session-1");
    timer.noteMovement();
    now = 2_000;
    hidden = true;
    timer.visibilityChanged();
    now = 3_000;
    hidden = false;
    timer.visibilityChanged();
    now = 4_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(2_000);

    timer.noteMovement();
    now = 5_000;
    timer.tick();
    expect(new ActiveTimeStore(storage, key).resume(7, "session-1")).toBe(3_000);
  });

  it("does nothing when reset without an active lesson", () => {
    const storage = new MemoryStorage();
    const key = activeTimeStorageKey("es-es");
    let renderCount = 0;
    const timer = new ActiveTimer(new ActiveTimeStore(storage, key), {
      clock: () => 0,
      isHidden: () => false,
      render: () => {
        renderCount += 1;
      },
    });

    timer.reset();

    expect(storage.getItem(key)).toBeNull();
    expect(renderCount).toBe(1);
  });
});
