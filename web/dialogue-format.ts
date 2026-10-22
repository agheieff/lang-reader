const dialogueDash = /^(?:[\u2012-\u2015]\s*|-\s+)\S/u;
const openingQuote = /^[\s\u00a0]*["“„«「『]/u;
const introducedQuote = /(?:[:：,，]\s*)["“„«「『][^"”»」』\n]+["”»」』]/u;

/**
 * Detect visible direct-speech turns without relying on one language's reporting verbs.
 *
 * Generated dialogue uses standard quotation marks or dialogue dashes. Keeping this deliberately
 * narrow avoids turning ordinary sentences that merely mention a quoted word into separate lines.
 */
export function isDialogueSentence(text: string): boolean {
  const source = text.trimStart();
  return dialogueDash.test(source) || openingQuote.test(source) || introducedQuote.test(source);
}
