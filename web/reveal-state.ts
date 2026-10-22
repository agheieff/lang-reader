export class SentenceRevealState {
  private readonly revealed: Set<string>;

  constructor(previouslyRevealed: Iterable<string> = []) {
    this.revealed = new Set(previouslyRevealed);
  }

  has(sentenceKey: string): boolean {
    return this.revealed.has(sentenceKey);
  }

  reveal(sentenceKey: string): boolean {
    if (this.revealed.has(sentenceKey)) return false;
    this.revealed.add(sentenceKey);
    return true;
  }
}

// A long press opens sentence help on touch; a plain tap between words is too easy to hit.
export const SENTENCE_LONG_PRESS_MS = 450;
export const LONG_PRESS_MOVE_TOLERANCE_PX = 10;
// On touch, sentence help that is closed again this quickly was opened by mistake and is not
// recorded as learning evidence.
export const TOUCH_SENTENCE_HELP_GRACE_MS = 1_500;

export function shouldHandleSentenceClick(
  selectionMode: boolean,
  targetIsTerm: boolean,
  targetIsControl: boolean,
  touch = false,
): boolean {
  if (targetIsControl) return false;
  if (selectionMode) return true;
  return !touch && !targetIsTerm;
}
