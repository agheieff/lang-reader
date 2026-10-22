/** The words view: vocabulary evidence table with search and sort. */

import type { WordsPayload, WordView } from "./contracts.js";
import { byId, plural, summaryStat, textElement } from "./dom.js";
import { isChineseLanguage } from "./language-tag.js";
import { renderHanziTones, renderPinyin } from "./pinyin.js";
import {
  attachSortHeaderButtons,
  type SortControlsConfig,
  updateSortControls,
} from "./sort-controls.js";
import { formatEvidence, masteryMeter, metricCell, wordCell } from "./table-cells.js";
import {
  DEFAULT_WORD_SORT,
  nextWordSort,
  parseWordSortSelection,
  reverseWordSortResult,
  type WordSortState,
  wordSortSelection,
} from "./word-sort-ui.js";
import { formatDue, selectWords, WORD_SORTS, type WordSort } from "./words.js";

let wordsPayload: WordsPayload | null = null;

let wordSortState: WordSortState = DEFAULT_WORD_SORT;

export function renderWordsView(state: WordsPayload): void {
  wordsPayload = state;
  const totals = state.words.reduce(
    (result, word) => ({
      appearances: result.appearances + word.occurrence_count,
      clicks: result.clicks + word.raw_reveal_count,
      counted: result.counted + word.counted_reveal_sessions,
      cleanReads: result.cleanReads + word.qualified_exposures,
    }),
    { appearances: 0, clicks: 0, counted: 0, cleanReads: 0 },
  );
  const summary = byId("words-summary");
  summary.replaceChildren(
    summaryStat("words-summary-stat", state.words.length, "words"),
    summaryStat("words-summary-stat", totals.appearances, "appearances"),
    summaryStat("words-summary-stat", totals.clicks, "popup clicks"),
    summaryStat("words-summary-stat", totals.counted, "SRS-attributed checks"),
    summaryStat("words-summary-stat", totals.cleanReads, "clean reads"),
  );
  renderWordTable();
}

const WORD_SORT_CONTROLS: SortControlsConfig<WordSort> = {
  selectId: "words-sort",
  buttonSelector: "[data-word-sort]",
  datasetKey: "wordSort",
  sorts: WORD_SORTS,
};

function renderWordTable(): void {
  const state = wordsPayload;
  if (!state) return;
  const selected = selectWords(
    state.words,
    byId<HTMLInputElement>("words-search").value,
    wordSortState.sort,
  );
  if (reverseWordSortResult(wordSortState)) selected.reverse();
  updateWordSortControls();
  const body = byId<HTMLTableSectionElement>("words-table-body");
  body.replaceChildren(...selected.map((word) => renderWordRow(word, state.learning_language)));
  byId("words-empty").hidden = selected.length > 0;
}

function updateWordSortControls(): void {
  updateSortControls(WORD_SORT_CONTROLS, wordSortState, wordSortSelection(wordSortState));
}

function renderWordRow(word: WordView, learningLanguage: string): HTMLTableRowElement {
  const row = document.createElement("tr");
  const identity = wordCell("Word");
  const lemma = textElement("p", "word-lemma", "");
  renderHanziTones(lemma, word.lemma, word.pronunciation ?? "");
  identity.append(lemma);
  if (word.pronunciation) {
    const pronunciation = textElement("p", "word-pronunciation", "");
    if (isChineseLanguage(learningLanguage)) {
      renderPinyin(pronunciation, word.pronunciation, word.lemma);
    } else {
      pronunciation.textContent = word.pronunciation;
    }
    identity.append(pronunciation);
  }
  if (word.frequency_rank) {
    identity.append(
      textElement("p", "word-meta", `Frequency #${word.frequency_rank.toLocaleString()}`),
    );
  }

  const meaning = wordCell("Meaning");
  const gloss = textElement("p", "word-gloss", "");
  gloss.append(textElement("span", "word-class", word.pos), document.createTextNode(word.gloss));
  meaning.append(gloss);
  if (word.related_senses.length > 0) {
    const related = document.createElement("div");
    related.className = "word-related-senses";
    related.append(textElement("p", "word-related-label", "Also in your texts"));
    const list = document.createElement("ul");
    for (const sense of word.related_senses) {
      const item = document.createElement("li");
      item.append(
        textElement("span", "word-class", sense.pos),
        document.createTextNode(sense.gloss),
      );
      if (sense.pronunciation && sense.pronunciation !== word.pronunciation) {
        const pronunciation = textElement("span", "word-related-pronunciation", "");
        if (isChineseLanguage(learningLanguage)) {
          renderPinyin(pronunciation, sense.pronunciation, word.lemma);
        } else {
          pronunciation.textContent = sense.pronunciation;
        }
        item.append(pronunciation);
      }
      list.append(item);
    }
    related.append(list);
    meaning.append(related);
  }
  const showForms =
    word.surface_forms.length > 1 ||
    word.surface_forms.some((surface) => surface.text !== word.lemma);
  if (showForms) {
    const forms = word.surface_forms
      .slice(0, 4)
      .map((surface) =>
        surface.occurrences > 1 ? `${surface.text} ×${surface.occurrences}` : surface.text,
      )
      .join(" · ");
    const remainder = word.surface_forms.length > 4 ? ` · +${word.surface_forms.length - 4}` : "";
    meaning.append(textElement("p", "word-forms", `Forms: ${forms}${remainder}`));
  }

  const met = metricCell(
    "Met",
    word.occurrence_count,
    `${word.exposed_lesson_count} ${plural(word.exposed_lesson_count, "lesson")}`,
  );
  const clicks = metricCell(
    "Clicks",
    word.raw_reveal_count,
    `${word.counted_reveal_sessions} SRS-attributed ${plural(
      word.counted_reveal_sessions,
      "check",
    )}`,
  );
  clicks.title =
    "Literal popup openings, with direct or compound-attributed lesson sessions underneath";
  const cleanReads = metricCell(
    "Clean reads",
    word.qualified_exposures,
    word.reveal_failures > 0
      ? `${formatEvidence(word.reveal_failures)} weighted reveal evidence`
      : "No counted reveals",
  );

  const mastery = wordCell("Mastery");
  const percent = Math.round(word.mastery * 100);
  mastery.append(textElement("p", "word-count", `${percent}%`));
  mastery.append(
    masteryMeter(percent, `${word.lemma} estimated mastery`),
    textElement("p", "word-due", formatDue(word.next_due_at)),
  );

  row.append(identity, meaning, met, clicks, cleanReads, mastery);
  return row;
}

export function initWordsView(): void {
  byId("words-search").addEventListener("input", renderWordTable);
  byId("words-sort").addEventListener("change", (event) => {
    if (!(event.currentTarget instanceof HTMLSelectElement)) return;
    wordSortState = parseWordSortSelection(event.currentTarget.value);
    renderWordTable();
  });
  attachSortHeaderButtons(WORD_SORT_CONTROLS, (sort) => {
    wordSortState = nextWordSort(wordSortState, sort);
    renderWordTable();
  });
}
