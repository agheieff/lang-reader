import { STORAGE_PREFIX } from "./storage-prefix.js";

export const THEME_STORAGE_KEY = `${STORAGE_PREFIX}theme:v1`;
export const THEME_MODES = ["light", "dark"] as const;

export type ThemeMode = (typeof THEME_MODES)[number];

export interface ThemePreferenceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface ThemeRoot {
  dataset: DOMStringMap;
}

export function parseThemeMode(value: unknown): ThemeMode | null {
  return typeof value === "string" && THEME_MODES.includes(value as ThemeMode)
    ? (value as ThemeMode)
    : null;
}

export function readThemePreference(
  storage: ThemePreferenceStorage,
  fallback: ThemeMode = "light",
): ThemeMode {
  return parseThemeMode(storage.getItem(THEME_STORAGE_KEY)) ?? fallback;
}

export function persistThemePreference(
  storage: ThemePreferenceStorage,
  theme: ThemeMode,
): ThemeMode {
  storage.setItem(THEME_STORAGE_KEY, theme);
  return theme;
}

export function applyTheme(root: ThemeRoot, theme: ThemeMode): ThemeMode {
  root.dataset.theme = theme;
  return theme;
}
