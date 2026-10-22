export function baseLanguage(languageTag: string): string {
  const separator = languageTag.indexOf("-");
  const base = separator === -1 ? languageTag : languageTag.slice(0, separator);
  return base.toLowerCase();
}

export function isChineseLanguage(languageTag: string): boolean {
  return baseLanguage(languageTag) === "zh";
}
