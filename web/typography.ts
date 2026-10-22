export interface PunctuationSplit {
  attached: string;
  remainder: string;
}

const closingPunctuation = /^[\p{Pd}\p{Pe}\p{Pf}\p{Po}%‰‱°℃℉]+/u;

export function splitClosingPunctuation(text: string): PunctuationSplit {
  const match = closingPunctuation.exec(text);
  const attached = match?.[0] ?? "";
  return { attached, remainder: text.slice(attached.length) };
}
