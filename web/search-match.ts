const punctuationAndSpacing = /[\p{P}\p{Z}\s]/gu;
const punctuationSymbolsAndSpacing = /[\p{P}\p{S}\p{Z}\s]/gu;

export interface SearchMatchOptions {
  compactSymbols?: boolean;
}

export function normalizeSearchText(value: string): string {
  return value
    .toLocaleLowerCase()
    .replaceAll("u:", "u")
    .normalize("NFKD")
    .replaceAll(/\p{M}/gu, "");
}

export function matchesSearch(
  value: string,
  normalizedNeedle: string,
  options: SearchMatchOptions = {},
): boolean {
  const normalized = normalizeSearchText(value);
  if (normalized.includes(normalizedNeedle)) return true;
  const pattern = options.compactSymbols ? punctuationSymbolsAndSpacing : punctuationAndSpacing;
  const compactNeedle = normalizedNeedle.replaceAll(pattern, "");
  return compactNeedle.length > 0 && normalized.replaceAll(pattern, "").includes(compactNeedle);
}
