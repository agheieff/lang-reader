import { describe, expect, it } from "vitest";

import { baseLanguage, isChineseLanguage } from "./language-tag.js";

describe("language tags", () => {
  it("extracts the case-insensitive base language from BCP-47-style tags", () => {
    expect(baseLanguage("zh-Hant-TW")).toBe("zh");
    expect(baseLanguage("ES-419")).toBe("es");
    expect(baseLanguage("de")).toBe("de");
  });

  it("recognizes only the exact Chinese base language", () => {
    expect(isChineseLanguage("zh")).toBe(true);
    expect(isChineseLanguage("ZH-Hans")).toBe(true);
    expect(isChineseLanguage("zh-Hant-TW")).toBe(true);
    expect(isChineseLanguage("zhongwen")).toBe(false);
    expect(isChineseLanguage("zho")).toBe(false);
    expect(isChineseLanguage("zh_CN")).toBe(false);
  });
});
