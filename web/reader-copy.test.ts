import { describe, expect, it } from "vitest";

import { formatReaderCopy, type ReaderCopyPiece } from "./reader-copy.js";

function source(paragraph: number, sentence: number, text: string): ReaderCopyPiece {
  return { kind: "source", paragraph, sentence, text };
}

function control(paragraph: number, sentence: number, text: string): ReaderCopyPiece {
  return { kind: "control", paragraph, sentence, text };
}

describe("clean reader copy", () => {
  it("removes reader controls and joins Chinese sentences without line breaks", () => {
    expect(
      formatReaderCopy(
        [
          source(0, 0, "明天，办公室将使用一台新标签机。"),
          control(0, 0, "G"),
          control(0, 0, "A·文"),
          source(0, 1, "它的任务是包裹分类和打印地址。"),
          control(0, 1, "G"),
          control(0, 1, "2"),
          control(0, 1, "A·文"),
        ],
        "zh-Hans",
      ),
    ).toBe("明天，办公室将使用一台新标签机。它的任务是包裹分类和打印地址。");
  });

  it("preserves paragraph boundaries instead of adding sentence boundaries", () => {
    expect(
      formatReaderCopy(
        [
          source(0, 0, "A useful title"),
          source(1, 0, "The first sentence."),
          source(1, 1, "The second sentence."),
          source(2, 0, "A new paragraph."),
        ],
        "en",
      ),
    ).toBe("A useful title\n\nThe first sentence. The second sentence.\n\nA new paragraph.");
  });

  it("does not duplicate authored whitespace between sentences", () => {
    expect(
      formatReaderCopy([source(0, 0, "Primera frase. "), source(0, 1, "Segunda frase.")], "es-ES"),
    ).toBe("Primera frase. Segunda frase.");
  });

  it("keeps visually separated dialogue turns as clean paragraph text", () => {
    expect(
      formatReaderCopy(
        [
          source(1, 0, "Mara asked, “Should we wait?”"),
          control(1, 0, "A·文"),
          source(1, 1, "“Only ten minutes,” Alex said."),
        ],
        "en",
      ),
    ).toBe("Mara asked, “Should we wait?” “Only ten minutes,” Alex said.");
  });
});
