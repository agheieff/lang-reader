import { describe, expect, it } from "vitest";

import type { KeyValueStorage } from "./contracts.persistence.js";
import { ToneColorPreference, toneColorStorageKey } from "./tone-color.js";

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

describe("Hanzi tone-color preference", () => {
  it("is profile-scoped and off by default", () => {
    const storage = new MemoryStorage();
    const chinese = new ToneColorPreference(storage, toneColorStorageKey("zh-hans"));
    const spanish = new ToneColorPreference(storage, toneColorStorageKey("es-es"));

    chinese.set(true);

    expect(chinese.enabled()).toBe(true);
    expect(spanish.enabled()).toBe(false);
  });

  it("removes the stored preference when disabled", () => {
    const storage = new MemoryStorage();
    const preference = new ToneColorPreference(storage, toneColorStorageKey("zh-hans"));

    preference.set(true);
    preference.set(false);

    expect(preference.enabled()).toBe(false);
  });
});
