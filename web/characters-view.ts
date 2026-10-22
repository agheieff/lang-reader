import { nextCharacterSort, parseCharacterSortSelection } from "./characters.js";
import { attachSortHeaderButtons } from "./sort-controls.js";

/** The characters view: Chinese character exposure table with search and sort. */

import type { CharactersPayload, CharacterView } from "./characters.contract.js";
import {
  CHARACTER_SORTS,
  type CharacterSort,
  type CharacterSortState,
  characterSortSelection,
  DEFAULT_CHARACTER_SORT,
  reverseCharacterSortResult,
  selectCharacters,
  summarizeCharacters,
} from "./characters.js";
import { byId, plural, summaryStat, textElement } from "./dom.js";
import { formatLibraryDate } from "./library-date.js";
import { type SortControlsConfig, updateSortControls } from "./sort-controls.js";
import {
  formatEvidence,
  formatStability,
  masteryMeter,
  metricCell,
  wordCell,
} from "./table-cells.js";
import { formatDue } from "./words.js";

let charactersPayload: CharactersPayload | null = null;

let characterSortState: CharacterSortState = DEFAULT_CHARACTER_SORT;

export function renderCharactersView(state: CharactersPayload): void {
  charactersPayload = state;
  const summary = summarizeCharacters(state.characters);
  byId("characters-summary").replaceChildren(
    summaryStat("characters-summary-stat", summary.unique_characters, "characters"),
    summaryStat("characters-summary-stat", summary.appearances, "appearances"),
    summaryStat("characters-summary-stat", summary.qualified_exposures, "clean reading signals"),
    summaryStat(
      "characters-summary-stat",
      summary.inferred_failure_mass,
      "weighted difficulty signals",
    ),
  );
  renderCharacterTable();
}

function renderCharacterTable(): void {
  const state = charactersPayload;
  if (!state) return;
  const selected = selectCharacters(
    state.characters,
    byId<HTMLInputElement>("characters-search").value,
    characterSortState.sort,
  );
  if (reverseCharacterSortResult(characterSortState)) selected.reverse();
  updateCharacterSortControls();
  byId<HTMLTableSectionElement>("characters-table-body").replaceChildren(
    ...selected.map((entry) => renderCharacterRow(entry, state.learning_language)),
  );
  byId("characters-empty").hidden = selected.length > 0;
}

const CHARACTER_SORT_CONTROLS: SortControlsConfig<CharacterSort> = {
  selectId: "characters-sort",
  buttonSelector: "[data-character-sort]",
  datasetKey: "characterSort",
  sorts: CHARACTER_SORTS,
};

function updateCharacterSortControls(): void {
  updateSortControls(
    CHARACTER_SORT_CONTROLS,
    characterSortState,
    characterSortSelection(characterSortState),
  );
}

function renderCharacterRow(entry: CharacterView, learningLanguage: string): HTMLTableRowElement {
  const row = document.createElement("tr");
  const identity = wordCell("Character");
  const glyph = textElement("p", "character-glyph", entry.character);
  glyph.lang = learningLanguage;
  identity.append(
    glyph,
    textElement("p", "character-date", `Last met ${formatLibraryDate(entry.last_exposed_at)}`),
  );

  const met = metricCell(
    "Met",
    entry.occurrence_count,
    `${entry.exposed_lesson_count} opened ${plural(entry.exposed_lesson_count, "text")}`,
  );
  met.title = "Literal appearances across opened titles and text";
  const contexts = metricCell(
    "Contexts",
    entry.distinct_word_contexts,
    `distinct reusable word ${plural(entry.distinct_word_contexts, "context")}`,
  );
  const checks = metricCell(
    "Checks",
    entry.raw_reveal_count,
    `${entry.counted_reveal_sessions} counted ${plural(entry.counted_reveal_sessions, "session")}`,
  );
  checks.title =
    "Popup openings for words or display spans containing this character; each check is only bounded evidence about the character.";

  const evidence = wordCell("Evidence");
  evidence.append(
    textElement("p", "word-count", entry.qualified_exposures.toLocaleString()),
    textElement(
      "p",
      "word-count-detail",
      `clean completed-reading ${plural(entry.qualified_exposures, "signal")}`,
    ),
    textElement(
      "p",
      "character-difficulty-evidence",
      entry.inferred_failure_mass > 0
        ? `${formatEvidence(entry.inferred_failure_mass)} weighted difficulty · ${
            entry.inferred_failure_sessions
          } ${plural(entry.inferred_failure_sessions, "session")}`
        : "No inferred difficulty",
    ),
  );
  if (entry.direct_successes > 0 || entry.direct_failures > 0) {
    evidence.append(
      textElement(
        "p",
        "character-direct-evidence",
        `${entry.direct_successes} direct ${plural(
          entry.direct_successes,
          "success",
        )} · ${entry.direct_failures} direct ${plural(entry.direct_failures, "failure")}`,
      ),
    );
  }

  const recognition = renderCharacterRecognition(entry);

  row.append(identity, met, contexts, checks, evidence, recognition);
  return row;
}

function renderCharacterRecognition(entry: CharacterView): HTMLTableCellElement {
  const cell = wordCell("Recognition");
  if (!entry.last_evidence_at) {
    cell.append(
      textElement("p", "character-recognition-value", "Unrated"),
      textElement("p", "word-count-detail", "No completed-reading evidence yet"),
    );
    return cell;
  }

  const currentPercent = Math.round(entry.retrievability * 100);
  const masteryPercent = Math.round(entry.mastery * 100);
  const uncertaintyPoints = Math.round(entry.mastery_uncertainty * 100);
  cell.append(textElement("p", "character-recognition-value", `${currentPercent}% now`));

  cell.append(
    masteryMeter(
      currentPercent,
      `${entry.character} estimated recognition now`,
      "character-recognition-meter",
    ),
    textElement(
      "p",
      "character-recognition-detail",
      `${masteryPercent}% at last evidence · ±${uncertaintyPoints} pts`,
    ),
    textElement(
      "p",
      "word-due",
      `${formatDue(entry.next_due_at)} · ${formatStability(entry.stability_days)}`,
    ),
  );
  cell.title = `Last evidence ${formatLibraryDate(entry.last_evidence_at)}`;
  return cell;
}

export function initCharactersView(): void {
  byId("characters-search").addEventListener("input", renderCharacterTable);
  byId("characters-sort").addEventListener("change", (event) => {
    if (!(event.currentTarget instanceof HTMLSelectElement)) return;
    characterSortState = parseCharacterSortSelection(event.currentTarget.value);
    renderCharacterTable();
  });
  attachSortHeaderButtons(CHARACTER_SORT_CONTROLS, (sort) => {
    characterSortState = nextCharacterSort(characterSortState, sort);
    renderCharacterTable();
  });
}
