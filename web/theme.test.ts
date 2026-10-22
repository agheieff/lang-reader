import { describe, expect, it } from "vitest";

import {
  applyTheme,
  parseThemeMode,
  persistThemePreference,
  readThemePreference,
  THEME_STORAGE_KEY,
  type ThemePreferenceStorage,
} from "./theme.js";

class MemoryStorage implements ThemePreferenceStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

describe("theme preference", () => {
  it("accepts only the two supported modes", () => {
    expect(parseThemeMode("light")).toBe("light");
    expect(parseThemeMode("dark")).toBe("dark");
    expect(parseThemeMode("system")).toBeNull();
    expect(parseThemeMode(null)).toBeNull();
  });

  it("uses a supplied fallback for missing or invalid stored values", () => {
    const storage = new MemoryStorage();
    expect(readThemePreference(storage)).toBe("light");
    expect(readThemePreference(storage, "dark")).toBe("dark");

    storage.values.set(THEME_STORAGE_KEY, "sepia");
    expect(readThemePreference(storage)).toBe("light");
  });

  it("persists and applies an explicit preference", () => {
    const storage = new MemoryStorage();
    const root = { dataset: {} as DOMStringMap };

    expect(persistThemePreference(storage, "dark")).toBe("dark");
    expect(readThemePreference(storage)).toBe("dark");
    expect(applyTheme(root, "dark")).toBe("dark");
    expect(root.dataset.theme).toBe("dark");
  });
});
