import { describe, expect, it } from "vitest";

import { hanziToneParts, pinyinParts, renderHanziTones, renderPinyin } from "./pinyin.js";

function syllables(pronunciation: string, lemma = ""): readonly [string, number][] {
  return pinyinParts(pronunciation, lemma).flatMap((part) =>
    part.kind === "syllable" ? [[part.text, part.tone] as const] : [],
  );
}

function display(pronunciation: string, lemma = ""): string {
  return pinyinParts(pronunciation, lemma)
    .map((part) => part.text)
    .join("");
}

describe("pinyin display", () => {
  it("detects tones in joined diacritic pinyin", () => {
    expect(syllables("gōngyuán", "公园")).toEqual([
      ["gōng", 1],
      ["yuán", 2],
    ]);
    expect(display("gōngyuán", "公园")).toBe("gōng yuán");
  });

  it("normalizes existing spaces and joined neutral syllables consistently", () => {
    expect(display("hěn jiǔ", "很久")).toBe("hěn jiǔ");
    expect(display("hěnjiǔ", "很久")).toBe("hěn jiǔ");
    expect(syllables("háizimen", "孩子们")).toEqual([
      ["hái", 2],
      ["zi", 5],
      ["men", 5],
    ]);
  });

  it("recognizes compact erhua as a suffix on the preceding syllable", () => {
    expect(syllables("yíhuìr", "一会儿")).toEqual([
      ["yí", 2],
      ["huìr", 4],
    ]);
    expect(display("yíhuìr", "一会儿")).toBe("yí huìr");
    expect(display("yí huìr", "一会儿")).toBe("yí huìr");
    expect(syllables("na3r", "哪儿")).toEqual([["nǎr", 3]]);
  });

  it("detects numbered pinyin and replaces tone numbers with marks", () => {
    expect(syllables("gong1yuan2", "公园")).toEqual([
      ["gōng", 1],
      ["yuán", 2],
    ]);
    expect(syllables("nu:3", "女")).toEqual([["nǚ", 3]]);
    expect(syllables("de5", "的")).toEqual([["de", 5]]);
  });

  it("preserves explicit separators when the Han alignment is not reliable", () => {
    expect(display("qiányí-mòhuà")).toBe("qián yí-mò huà");
  });

  it("renders pronunciation as text nodes and tone-labelled spans", () => {
    interface FakeNode {
      className?: string;
      dataset?: Record<string, string>;
      textContent: string;
    }

    let children: readonly FakeNode[] = [];
    const ownerDocument = {
      createElement: (): FakeNode => ({ className: "", dataset: {}, textContent: "" }),
      createTextNode: (text: string): FakeNode => ({ textContent: text }),
    };
    const container = {
      ownerDocument,
      replaceChildren: (...nodes: readonly FakeNode[]): void => {
        children = nodes;
      },
    };

    renderPinyin(container as unknown as HTMLElement, "gong1yuan2", "公园");

    expect(children).toEqual([
      { className: "pinyin-syllable", dataset: { tone: "1" }, textContent: "gōng" },
      { textContent: " " },
      { className: "pinyin-syllable", dataset: { tone: "2" }, textContent: "yuán" },
    ]);
  });
});

describe("Hanzi tone alignment", () => {
  const toneValues = (surface: string, pronunciation: string): readonly number[] =>
    hanziToneParts(surface, pronunciation).flatMap((part) =>
      part.kind === "hanzi" ? [part.tone] : [],
    );

  it("aligns joined pinyin to Han characters", () => {
    expect(toneValues("公园", "gōngyuán")).toEqual([1, 2]);
  });

  it("aligns spaced pinyin and preserves non-Han text", () => {
    expect(hanziToneParts("去公园！", "qù gōng yuán")).toEqual([
      { kind: "hanzi", text: "去", tone: 4 },
      { kind: "hanzi", text: "公", tone: 1 },
      { kind: "hanzi", text: "园", tone: 2 },
      { kind: "text", text: "！" },
    ]);
  });

  it("aligns numbered pinyin", () => {
    expect(toneValues("你好", "ni3hao3")).toEqual([3, 3]);
  });

  it("extends a rhotacized syllable tone over its written 儿 suffix", () => {
    expect(toneValues("一会儿", "yíhuìr")).toEqual([2, 4, 4]);
    expect(toneValues("一会儿", "yí huìr")).toEqual([2, 4, 4]);
    expect(toneValues("哪儿", "na3r")).toEqual([3, 3]);
    expect(toneValues("儿", "ér")).toEqual([2]);
    expect(toneValues("女儿", "nǚ ér")).toEqual([3, 2]);
  });

  it("treats unmarked syllables as neutral tone", () => {
    expect(toneValues("孩子们", "háizimen")).toEqual([2, 5, 5]);
  });

  it("falls back to plain text for missing or mismatched pronunciation", () => {
    expect(hanziToneParts("公园", "gōng")).toEqual([{ kind: "text", text: "公园" }]);
    expect(hanziToneParts("公园", "")).toEqual([{ kind: "text", text: "公园" }]);
  });

  it("renders tone-labelled Han spans", () => {
    interface FakeNode {
      className?: string;
      dataset?: Record<string, string>;
      textContent: string;
    }

    let children: readonly FakeNode[] = [];
    const ownerDocument = {
      createElement: (): FakeNode => ({ className: "", dataset: {}, textContent: "" }),
      createTextNode: (text: string): FakeNode => ({ textContent: text }),
    };
    const container = {
      ownerDocument,
      replaceChildren: (...nodes: readonly FakeNode[]): void => {
        children = nodes;
      },
    };

    renderHanziTones(container as unknown as HTMLElement, "公园！", "gong1yuan2");

    expect(children).toEqual([
      { className: "hanzi-tone", dataset: { tone: "1" }, textContent: "公" },
      { className: "hanzi-tone", dataset: { tone: "2" }, textContent: "园" },
      { textContent: "！" },
    ]);
  });
});
