export const TERM_BANDS = ["familiar", "expected", "uncertain", "focus", "incidental"] as const;
export type TermBand = (typeof TERM_BANDS)[number];

export interface TermBandParseOptions {
  prefix?: string;
  blankKeyRequirement?: string;
  unsupportedBandRequirement?: string;
}

export function parseTermBands(
  value: unknown,
  options: TermBandParseOptions = {},
): Record<string, TermBand> {
  const prefix = options.prefix ?? "Invalid reader";
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${prefix}: term_bands must be an object`);
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, band]) => {
      if (!key.trim()) {
        throw new Error(
          `${prefix}: term_bands key ${options.blankKeyRequirement ?? "must not be blank"}`,
        );
      }
      if (typeof band !== "string" || !TERM_BANDS.includes(band as TermBand)) {
        throw new Error(
          `${prefix}: term_bands.${key} ${
            options.unsupportedBandRequirement ?? "has an unsupported band"
          }`,
        );
      }
      return [key, band as TermBand];
    }),
  );
}

interface TermBandElement {
  dataset: { termBand?: string | undefined };
  classList: { add(token: string): void };
}

export function attachTermBand(element: TermBandElement, band: TermBand | undefined): void {
  if (!band) return;
  element.dataset.termBand = band;
  element.classList.add(`term-band-${band}`);
}
