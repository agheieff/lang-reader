import type { KeyValueStorage } from "./contracts.persistence.js";
import { STORAGE_PREFIX } from "./storage-prefix.js";

export function toneColorStorageKey(profileId: string): string {
  return `${STORAGE_PREFIX}${profileId}:hanzi-tone-colors:v1`;
}

export class ToneColorPreference {
  constructor(
    private readonly storage: KeyValueStorage,
    private readonly key: string,
  ) {}

  enabled(): boolean {
    return this.storage.getItem(this.key) === "on";
  }

  set(enabled: boolean): void {
    if (enabled) this.storage.setItem(this.key, "on");
    else this.storage.removeItem(this.key);
  }
}
