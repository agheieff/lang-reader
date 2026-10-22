import { describe, expect, it } from "vitest";

import {
  setTextCardActionPending,
  textDispositionPath,
  textDispositionPayload,
} from "./text-disposition.js";

function attributeTarget(): {
  attributes: Map<string, string>;
  setAttribute(name: string, value: string): void;
  removeAttribute(name: string): void;
} {
  const attributes = new Map<string, string>();
  return {
    attributes,
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
  };
}

describe("text disposition requests", () => {
  it("builds profile-scoped skip and restore requests", () => {
    expect(textDispositionPath("/api/profiles/zh-hans", 7, "skip")).toBe(
      "/api/profiles/zh-hans/texts/7/skip",
    );
    expect(textDispositionPath("/api/profiles/zh-hans", 7, "restore")).toBe(
      "/api/profiles/zh-hans/texts/7/restore",
    );
    expect(textDispositionPayload("action-1")).toEqual({ action_id: "action-1" });
  });

  it("rejects invalid request identities", () => {
    expect(() => textDispositionPath("/api/profiles/es-es", 0, "skip")).toThrow(/positive/);
    expect(() => textDispositionPayload("  ")).toThrow(/blank/);
  });

  it("applies and restores one pending state across a text card action", () => {
    const cardAttributes = attributeTarget();
    const card = {
      ...cardAttributes,
      inert: false,
    } as unknown as HTMLElement;
    const buttonAttributes = attributeTarget();
    const button = {
      ...buttonAttributes,
      closest: () => card,
      disabled: false,
      textContent: "Skip",
    } as unknown as HTMLButtonElement;

    expect(setTextCardActionPending(button, true, "Skipping…")).toBe(card);
    expect(button.disabled).toBe(true);
    expect(button.textContent).toBe("Skipping…");
    expect(buttonAttributes.attributes.get("aria-busy")).toBe("true");
    expect(card.inert).toBe(true);
    expect(cardAttributes.attributes.get("aria-busy")).toBe("true");

    setTextCardActionPending(button, false, "Skip", card);
    expect(button.disabled).toBe(false);
    expect(button.textContent).toBe("Skip");
    expect(buttonAttributes.attributes.has("aria-busy")).toBe(false);
    expect(card.inert).toBe(false);
    expect(cardAttributes.attributes.has("aria-busy")).toBe(false);
  });
});
