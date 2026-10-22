import type { GrammarOccurrence } from "./contracts.js";

export function visibleGrammarOccurrences(
  occurrences: readonly GrammarOccurrence[],
  comfortableConstructionKeys: ReadonlySet<string>,
): GrammarOccurrence[] {
  return occurrences.filter(
    (occurrence) => !comfortableConstructionKeys.has(occurrence.construction_key),
  );
}
