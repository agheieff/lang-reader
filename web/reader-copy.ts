import { isChineseLanguage } from "./language-tag.js";

export interface ReaderCopyPiece {
  readonly kind: "source" | "control";
  readonly paragraph: number;
  readonly sentence: number;
  readonly text: string;
}

function sentenceSeparator(learningLanguage: string): string {
  return isChineseLanguage(learningLanguage) ? "" : " ";
}

function appendSentence(current: string, next: string, separator: string): string {
  if (!current || !next || /\s$/u.test(current) || /^\s/u.test(next)) return `${current}${next}`;
  return `${current}${separator}${next}`;
}

export function formatReaderCopy(
  pieces: readonly ReaderCopyPiece[],
  learningLanguage: string,
): string {
  const paragraphs = new Map<number, Map<number, string>>();
  for (const piece of pieces) {
    if (piece.kind !== "source" || piece.text === "") continue;
    let sentences = paragraphs.get(piece.paragraph);
    if (!sentences) {
      sentences = new Map<number, string>();
      paragraphs.set(piece.paragraph, sentences);
    }
    sentences.set(piece.sentence, `${sentences.get(piece.sentence) ?? ""}${piece.text}`);
  }

  const separator = sentenceSeparator(learningLanguage);
  return [...paragraphs.values()]
    .map((sentences) =>
      [...sentences.values()]
        .reduce((text, sentence) => appendSentence(text, sentence, separator), "")
        .trim(),
    )
    .filter(Boolean)
    .join("\n\n");
}

function selectedSlices(node: Text, ranges: readonly Range[]): readonly string[] {
  const intervals: { start: number; end: number }[] = [];
  for (const range of ranges) {
    if (!range.intersectsNode(node)) continue;
    const start = range.startContainer === node ? range.startOffset : 0;
    const end = range.endContainer === node ? range.endOffset : node.data.length;
    if (start < end) intervals.push({ start, end });
  }
  intervals.sort((left, right) => left.start - right.start || left.end - right.end);

  const merged: { start: number; end: number }[] = [];
  for (const interval of intervals) {
    const previous = merged.at(-1);
    if (previous && interval.start <= previous.end) {
      previous.end = Math.max(previous.end, interval.end);
    } else {
      merged.push({ ...interval });
    }
  }
  return merged.map(({ start, end }) => node.data.slice(start, end));
}

function selectedReaderPieces(
  selection: Selection,
  title: HTMLElement,
  content: HTMLElement,
): ReaderCopyPiece[] {
  const ranges = Array.from({ length: selection.rangeCount }, (_, index) =>
    selection.getRangeAt(index),
  );
  const paragraphs = [title, ...content.querySelectorAll<HTMLElement>(".lesson-block")];
  const paragraphIndexes = new Map<Element, number>(
    paragraphs.map((paragraph, index) => [paragraph, index]),
  );
  const sentenceIndexes = new Map<Element, number>();
  for (const paragraph of paragraphs) {
    for (const [index, sentence] of [
      ...paragraph.querySelectorAll<HTMLElement>(".sentence-unit"),
    ].entries()) {
      sentenceIndexes.set(sentence, index);
    }
  }

  const pieces: ReaderCopyPiece[] = [];
  for (const root of [title, content]) {
    const walker = root.ownerDocument.createTreeWalker(root, 4);
    let current = walker.nextNode();
    while (current) {
      if (current instanceof Text) {
        const parent = current.parentElement;
        const paragraphElement = parent?.closest(".lesson-block") ?? root;
        const paragraph = paragraphIndexes.get(paragraphElement);
        const sentenceElement = parent?.closest(".sentence-unit");
        const sentence = sentenceElement ? sentenceIndexes.get(sentenceElement) : 0;
        if (paragraph !== undefined && sentence !== undefined) {
          const kind = parent?.closest("[data-reader-copy-ignore]") ? "control" : "source";
          for (const text of selectedSlices(current, ranges)) {
            pieces.push({ kind, paragraph, sentence, text });
          }
        }
      }
      current = walker.nextNode();
    }
  }
  return pieces;
}

export function copyReaderSelection(
  event: ClipboardEvent,
  selection: Selection | null,
  title: HTMLElement,
  content: HTMLElement,
  learningLanguage: string,
): boolean {
  if (!selection || selection.isCollapsed || !event.clipboardData) return false;
  const text = formatReaderCopy(selectedReaderPieces(selection, title, content), learningLanguage);
  if (!text) return false;
  event.clipboardData.setData("text/plain", text);
  event.preventDefault();
  return true;
}
