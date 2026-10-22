import { isChineseLanguage } from "./language-tag.js";

export interface ReaderLevelPresentation {
  label: string;
  title?: string | undefined;
}

const HSK_BAND_COUNT = 9;

/**
 * Maps the shared 0-1 lesson difficulty scale into nine equal HSK display bands.
 * The upper endpoint belongs to HSK 9; every other band is [n / 9, (n + 1) / 9).
 */
export function hskBandForDifficulty(difficulty: number): number {
  if (!Number.isFinite(difficulty) || difficulty < 0 || difficulty > 1) {
    throw new RangeError("Lesson difficulty must be between 0 and 1");
  }
  return Math.min(HSK_BAND_COUNT, Math.floor(difficulty * HSK_BAND_COUNT) + 1);
}

export function readerLevelPresentation(
  learningLanguage: string,
  lessonLevel: string,
  difficulty: number,
): ReaderLevelPresentation {
  if (!isChineseLanguage(learningLanguage)) return { label: lessonLevel };

  const hskBand = hskBandForDifficulty(difficulty);
  return {
    label: `Approx. HSK ${hskBand}`,
    title: `Approximate HSK ${hskBand}, inferred from this lesson's difficulty`,
  };
}
