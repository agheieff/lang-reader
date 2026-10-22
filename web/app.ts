import { showActionStatus } from "./action-status.js";
import { initActivationView, loadActivation } from "./activation-view.js";
import { ActiveTimer, TOUCH_READING_ACTIVITY_WINDOW_MS } from "./active-time.js";
import { getJson, profileApi, profileId } from "./api.js";
import { appPath } from "./base-path.js";
import type { CharactersPayload } from "./characters.contract.js";
import { parseCharacters } from "./characters.contract.js";
import { initCharactersView, renderCharactersView } from "./characters-view.js";
import type { WordsPayload } from "./contracts.js";
import {
  type EventType,
  type GrammarCatalogEntry,
  type JsonRecord,
  type LearningEvent,
  type Lesson,
  parseReader,
  parseWords,
  type ReaderPayload,
  type ReaderProgress,
  type Run,
  type Sentence,
  type TermBand,
} from "./contracts.js";
import {
  ActiveTimeStore,
  activeTimeStorageKey,
  DurableEventStore,
  eventQueueStorageKey,
  LessonSessionStore,
  lessonSessionStorageKey,
  migrateLegacyPersistence,
} from "./contracts.persistence.js";
import { isDialogueSentence } from "./dialogue-format.js";
import { byId, plural, textElement } from "./dom.js";
import { EventQueue } from "./event-queue.js";
import { type GrammarState, parseGrammar } from "./grammar.js";
import { visibleGrammarOccurrences } from "./grammar-markers.js";
import { initGrammarView, renderGrammarView } from "./grammar-view.js";
import { isChineseLanguage } from "./language-tag.js";
import {
  configureLessonAudio,
  initLessonAudio,
  stopLessonAudio,
  suspendLessonAudio,
} from "./lesson-audio.js";
import { renderHanziTones, renderPinyin } from "./pinyin.js";
import { copyReaderSelection } from "./reader-copy.js";
import { readerLevelPresentation } from "./reader-level.js";
import {
  parseReaderLaunch,
  type ReaderLaunch,
  type ReaderMode,
  readerApiPath,
  readerModeRecordsEvidence,
} from "./reader-mode.js";
import { contentExperiment } from "./reading-preferences.js";
import {
  LONG_PRESS_MOVE_TOLERANCE_PX,
  SENTENCE_LONG_PRESS_MS,
  SentenceRevealState,
  shouldHandleSentenceClick,
  TOUCH_SENTENCE_HELP_GRACE_MS,
} from "./reveal-state.js";
import { pronunciationForRun } from "./run-pronunciation.js";
import { initSettingsView, loadSettings } from "./settings-view.js";
import { parseStatistics, type StatisticsPayload } from "./statistics.js";
import { renderStatisticsView } from "./statistics-view.js";
import { attachTermBand } from "./term-band.js";
import { parseTextLibrary, parseTextPreview } from "./texts.contract.js";
import {
  clearTextsPoll,
  initTextsView,
  postTextDisposition,
  renderTextsView,
} from "./texts-view.js";
import {
  applyTheme,
  parseThemeMode,
  persistThemePreference,
  readThemePreference,
  type ThemeMode,
} from "./theme.js";
import { ToneColorPreference, toneColorStorageKey } from "./tone-color.js";
import { splitClosingPunctuation } from "./typography.js";
import { initWordsView, renderWordsView } from "./words-view.js";

const loadingView = byId("loading-view");
const errorView = byId("error-view");
const activationView = byId("activation-view");
const readerView = byId("reader-view");
const textsView = byId("texts-view");
const wordsView = byId("words-view");
const charactersView = byId("characters-view");
const grammarView = byId("grammar-view");
const statisticsView = byId("statistics-view");
const settingsView = byId("settings-view");
const lessonContent = byId<HTMLElement>("lesson-content");
const fullPanel = byId("full-translation");
const fullContent = byId("full-translation-content");
const fullButton = byId<HTMLButtonElement>("full-translation-button");
const completeButton = byId<HTMLButtonElement>("complete-button");
const skipLessonButtons = [
  byId<HTMLButtonElement>("skip-lesson-button"),
  byId<HTMLButtonElement>("skip-lesson-completion-button"),
];
const statusMessage = byId("status-message");
const popover = byId("gloss-popover");
const sentencePopover = byId("sentence-translation-popover");
const sentencePopoverSource = byId("sentence-source-text");
const sentencePopoverText = byId("sentence-translation-text");
const grammarPopover = byId("grammar-popover");
const grammarPopoverContent = byId("grammar-popover-content");
const sentenceHelpModeButton = byId<HTMLButtonElement>("sentence-help-mode");
const feedbackDetails = byId<HTMLDetailsElement>("expert-feedback");
const profileSwitch = byId<HTMLSelectElement>("profile-switch");
const toneColorToggle = byId<HTMLButtonElement>("tone-color-toggle");
const toneColorState = byId("tone-color-state");
const themeColorMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
const readOnlySnapshot = document.body.dataset.readOnly === "true";
type InitialView =
  | "characters"
  | "grammar"
  | "reading"
  | "settings"
  | "statistics"
  | "texts"
  | "words";
const requestedInitialView = document.body.dataset.initialView;
const initialView: InitialView =
  requestedInitialView === "words" ||
  requestedInitialView === "characters" ||
  requestedInitialView === "grammar" ||
  requestedInitialView === "statistics" ||
  requestedInitialView === "settings" ||
  requestedInitialView === "texts"
    ? requestedInitialView
    : "reading";
const VIEW_ERROR_COPY: Partial<Record<InitialView, { eyebrow: string; heading: string }>> = {
  characters: {
    eyebrow: "Characters unavailable",
    heading: "We couldn’t load your character exposure.",
  },
  grammar: {
    eyebrow: "Grammar unavailable",
    heading: "We couldn’t load your grammar evidence.",
  },
  settings: {
    eyebrow: "Settings unavailable",
    heading: "We couldn’t load your settings.",
  },
  statistics: {
    eyebrow: "Statistics unavailable",
    heading: "We couldn’t load your reading statistics.",
  },
  texts: {
    eyebrow: "Texts unavailable",
    heading: "We couldn’t load your text library.",
  },
  words: {
    eyebrow: "Words unavailable",
    heading: "We couldn’t load your vocabulary.",
  },
};
const toneColorPreference = new ToneColorPreference(
  window.localStorage,
  toneColorStorageKey(profileId),
);

function setTheme(theme: ThemeMode, announce = false): void {
  applyTheme(document.documentElement, theme);
  if (themeColorMeta) themeColorMeta.content = theme === "dark" ? "#111714" : "#f4f1e9";
  for (const input of document.querySelectorAll<HTMLInputElement>('input[name="color-theme"]')) {
    input.checked = input.value === theme;
  }
  byId("theme-status").textContent = announce
    ? `${theme === "dark" ? "Dark" : "Light"} theme applied.`
    : "";
}

setTheme(readThemePreference(window.localStorage));

const feedbackConflicts: Record<string, string> = {
  shorter: "longer",
  longer: "shorter",
  easier: "more_challenging",
  more_challenging: "easier",
  more_grammar: "less_grammar",
  less_grammar: "more_grammar",
  same_topic: "new_topic",
  new_topic: "same_topic",
};
for (const input of feedbackDetails.querySelectorAll<HTMLInputElement>(
  'input[name="lesson-feedback"]',
)) {
  input.addEventListener("change", () => {
    const conflict = feedbackConflicts[input.value];
    if (!input.checked || !conflict) return;
    const opposite = feedbackDetails.querySelector<HTMLInputElement>(
      `input[name="lesson-feedback"][value="${conflict}"]`,
    );
    if (opposite) opposite.checked = false;
  });
}

let lesson: Lesson | null = null;
let lessonId: number | null = null;
let sessionId = "";
let rating: -1 | 1 | null = null;
let glossAnchor: HTMLButtonElement | null = null;
let sentenceHelpAnchor: HTMLButtonElement | null = null;
let sentenceHelpUnit: HTMLElement | null = null;
let pendingSentenceEvidence: number | null = null;
let grammarHelpAnchor: HTMLButtonElement | null = null;
let sentenceReveals = new SentenceRevealState();
let revealedGrammarOccurrences = new Set<string>();
let readerGrammarCatalog = new Map<string, GrammarCatalogEntry>();
let comfortableGrammarConstructionKeys = new Set<string>();
let sentenceSelectionMode = false;
let fullTranslationRevealed = false;
let lessonPollTimer: number | null = null;
let currentLearningLanguage = "";
let readerMode: ReaderMode = "standard";

if (!readOnlySnapshot) migrateLegacyPersistence(window.localStorage, profileId);
const lessonSessions = new LessonSessionStore(
  window.localStorage,
  lessonSessionStorageKey(profileId),
);
const events = new EventQueue(
  new DurableEventStore(window.localStorage, eventQueueStorageKey(profileId)),
  {
    endpoint: `${profileApi}/events`,
    readOnly: readOnlySnapshot,
    onFlushError: showSyncWarning,
    onLessonCompleted: (completedLessonId, completedSessionId) => {
      lessonSessions.complete(completedLessonId, completedSessionId);
      activeTimer.complete(completedLessonId, completedSessionId);
    },
  },
);
const activeTimeStore = new ActiveTimeStore(window.localStorage, activeTimeStorageKey(profileId));
const activeTimer = new ActiveTimer(activeTimeStore, {
  clock: () => performance.now(),
  isHidden: () => document.hidden,
  render: (activeMs) => {
    const seconds = Math.floor(activeMs / 1_000);
    const minutes = Math.floor(seconds / 60);
    byId("active-time").textContent =
      `Active reading · ${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  },
});
const coarsePointer = window.matchMedia("(pointer: coarse)");
window.addEventListener(
  "pointermove",
  (event) =>
    activeTimer.noteMovement(
      event.pointerType === "touch" ? TOUCH_READING_ACTIVITY_WINDOW_MS : undefined,
    ),
  { passive: true },
);
window.addEventListener(
  "pointerdown",
  (event) => {
    if (event.pointerType === "touch") activeTimer.noteMovement(TOUCH_READING_ACTIVITY_WINDOW_MS);
  },
  { passive: true },
);
window.addEventListener(
  "scroll",
  () => {
    // Touch scrolling cancels pointer events, so scroll itself is the reading signal there.
    if (coarsePointer.matches) activeTimer.noteMovement(TOUCH_READING_ACTIVITY_WINDOW_MS);
  },
  { passive: true },
);
document.addEventListener("visibilitychange", () => activeTimer.visibilityChanged());
window.setInterval(() => activeTimer.tick(), 1_000);

function now(): string {
  return new Date().toISOString();
}

function makeEvent(type: EventType, payload: JsonRecord): LearningEvent {
  if (!readerModeRecordsEvidence(readerMode)) {
    throw new Error("Preview mode cannot create learning events");
  }
  if (lessonId === null || !sessionId) throw new Error("No active lesson session");
  return {
    event_id: crypto.randomUUID(),
    session_id: sessionId,
    lesson_id: lessonId,
    type,
    occurred_at: now(),
    payload,
  };
}

function recordLearningEvent(type: EventType, payload: JsonRecord, immediate = false): void {
  if (!readerModeRecordsEvidence(readerMode)) return;
  events.add(makeEvent(type, payload), immediate);
}

function setView(
  view:
    | "loading"
    | "error"
    | "activation"
    | "characters"
    | "reader"
    | "settings"
    | "statistics"
    | "texts"
    | "words"
    | "grammar",
): void {
  loadingView.hidden = view !== "loading";
  errorView.hidden = view !== "error";
  activationView.hidden = view !== "activation";
  readerView.hidden = view !== "reader";
  textsView.hidden = view !== "texts";
  wordsView.hidden = view !== "words";
  charactersView.hidden = view !== "characters";
  grammarView.hidden = view !== "grammar";
  statisticsView.hidden = view !== "statistics";
  settingsView.hidden = view !== "settings";
  if (view !== "reader") sentenceHelpModeButton.hidden = true;
  if (view !== "texts") clearTextsPoll();
  if (view !== "reader") {
    closeGloss();
    closeSentenceHelp();
    closeGrammarHelp();
    suspendLessonAudio();
  }
}

function renderActivation(): void {
  setView("activation");
  byId("activation-status").textContent = "";
}

function configureToneColors(learningLanguage: string): void {
  currentLearningLanguage = learningLanguage;
  const available = isChineseLanguage(learningLanguage);
  const enabled = available && toneColorPreference.enabled();
  toneColorToggle.hidden = !available;
  toneColorToggle.setAttribute("aria-pressed", String(enabled));
  toneColorToggle.setAttribute(
    "aria-label",
    enabled ? "Turn Hanzi tone colors off" : "Color Chinese characters by tone",
  );
  toneColorToggle.title = enabled ? "Turn Hanzi tone colors off" : "Color Hanzi by tone";
  toneColorState.textContent = enabled ? "On" : "Off";
  document.body.dataset.hanziToneColors = String(enabled);
}

interface LessonRenderState {
  lesson_id: number;
  lesson: Lesson;
  progress: ReaderProgress;
  term_bands: Record<string, TermBand>;
  grammar_catalog?: GrammarCatalogEntry[];
  comfortable_grammar_construction_keys?: string[];
}

function renderReaderMode(mode: ReaderMode, selectedFromLibrary: boolean): void {
  const banner = byId("reader-mode-banner");
  const title = byId("reader-mode-title");
  const detail = byId("reader-mode-detail");
  banner.classList.toggle("reader-mode-banner-preview", mode === "preview");
  banner.classList.toggle("reader-mode-banner-reread", mode === "reread");
  if (mode === "preview") {
    title.textContent = "Peek mode";
    detail.textContent =
      "Nothing here is recorded. Word help and translations are available without changing progress.";
    banner.hidden = false;
  } else if (mode === "reread") {
    title.textContent = "Reading again";
    detail.textContent =
      "This is a fresh session. Reveals and completion update your learning evidence.";
    banner.hidden = false;
  } else if (selectedFromLibrary) {
    title.textContent = "Reading from your library";
    detail.textContent = "Progress is recorded normally, just like in the main reading queue.";
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

function renderLesson(
  state: LessonRenderState,
  mode: ReaderMode = "standard",
  selectedFromLibrary = false,
): void {
  if (readOnlySnapshot) mode = "preview";
  readerMode = mode;
  resetReaderSkipButtons();
  for (const button of skipLessonButtons) {
    button.hidden = mode !== "standard" || state.progress.completed;
  }
  configureToneColors(state.lesson.learning_language);
  clearLessonPoll();
  lesson = state.lesson;
  const experiment = lesson ? contentExperiment(lesson.metadata) : null;
  byId("content-experiment").hidden = experiment === null;
  byId("content-experiment-kind").textContent = experiment?.kind ?? "";
  byId("content-experiment-text").textContent = experiment?.question ?? "";
  lessonId = state.lesson_id;
  if (readerModeRecordsEvidence(mode)) {
    const preferredSession = mode === "reread" ? crypto.randomUUID() : state.progress.session_id;
    sessionId = lessonSessions.getOrCreate(
      lessonId,
      preferredSession,
      crypto.randomUUID.bind(crypto),
    );
  } else {
    sessionId = "";
  }
  rating = null;
  fullTranslationRevealed = mode === "preview" ? false : state.progress.full_translation_revealed;
  closeGloss();
  closeSentenceHelp();
  closeGrammarHelp();
  setSentenceSelectionMode(false);
  sentenceReveals = new SentenceRevealState(
    mode === "preview" ? [] : state.progress.revealed_sentence_keys,
  );
  revealedGrammarOccurrences = new Set(
    mode === "preview" ? [] : state.progress.revealed_grammar_occurrence_keys,
  );
  readerGrammarCatalog = new Map((state.grammar_catalog ?? []).map((entry) => [entry.key, entry]));
  comfortableGrammarConstructionKeys = new Set(state.comfortable_grammar_construction_keys ?? []);
  lessonContent.replaceChildren();
  fullContent.replaceChildren();
  fullPanel.hidden = true;
  fullButton.textContent = "Show full translation";
  completeButton.disabled = !readerModeRecordsEvidence(mode);
  byId("completion-card").hidden = mode === "preview";
  renderReaderMode(mode, selectedFromLibrary);
  feedbackDetails.open = false;
  for (const input of feedbackDetails.querySelectorAll<HTMLInputElement>("input:checked")) {
    input.checked = false;
  }

  const revealedSentences = new Set(state.progress.revealed_sentence_keys);
  const title = byId("lesson-title");
  title.dir = "auto";
  if (lesson.title_sentence) {
    title.replaceChildren(
      renderSentence(
        lesson.title_sentence,
        revealedSentences.has(lesson.title_sentence.key),
        state.term_bands,
      ),
    );
  } else {
    title.textContent = lesson.title;
  }
  const level = readerLevelPresentation(lesson.learning_language, lesson.level, lesson.difficulty);
  const levelDetail = byId("lesson-level-detail");
  levelDetail.textContent = level.label;
  levelDetail.title = level.title ?? "";
  const difficultyDetail = byId("lesson-difficulty-detail");
  difficultyDetail.textContent = `${Math.round(lesson.difficulty * 100)}%`;
  difficultyDetail.title = "Fine-grained adaptive scale used to tune generated texts";
  const topicDetail = byId("lesson-topic-detail");
  const topicDetailItem = byId("lesson-topic-detail-item");
  topicDetail.textContent = lesson.topic ?? "";
  topicDetailItem.hidden = !lesson.topic;
  lessonContent.dir = "auto";
  let hasTranslations = Boolean(lesson.title_sentence?.translation);

  if (lesson.title_sentence?.translation) {
    fullContent.append(
      textElement("p", "translation-paragraph", lesson.title_sentence.translation),
    );
  }

  for (const block of lesson.blocks) {
    const paragraph = document.createElement("p");
    paragraph.className = "lesson-block";
    const translatedParagraph = document.createElement("p");
    translatedParagraph.className = "translation-paragraph";

    for (const sentence of block.sentences) {
      const sourceText = sentence.runs.map((run) => run.text).join("");
      const dialogue = isDialogueSentence(sourceText);
      paragraph.append(
        renderSentence(sentence, revealedSentences.has(sentence.key), state.term_bands, dialogue),
      );
      if (sentence.translation) {
        hasTranslations = true;
        translatedParagraph.append(
          textElement(
            "span",
            dialogue
              ? "translation-sentence translation-sentence-dialogue"
              : "translation-sentence",
            sentence.translation,
          ),
          document.createTextNode(" "),
        );
      }
    }
    lessonContent.append(paragraph);
    fullContent.append(translatedParagraph);
  }

  sentenceHelpModeButton.hidden = !hasTranslations;
  fullButton.hidden = !hasTranslations;
  if (readerModeRecordsEvidence(mode)) {
    renderRatings();
    activeTimer.start(lessonId, sessionId);
  }
  setView("reader");
  void configureLessonAudio(state.lesson_id);
  if (!state.progress.started) recordLearningEvent("lesson.started", {}, true);
}

function renderGrammar(state: GrammarState): void {
  configureToneColors(state.learning_language);
  renderGrammarView(state);
  setView("grammar");
}

function renderWords(state: WordsPayload): void {
  configureToneColors(state.learning_language);
  renderWordsView(state);
  setView("words");
}

function renderCharacters(state: CharactersPayload): void {
  configureToneColors(state.learning_language);
  renderCharactersView(state);
  setView("characters");
}

function renderStatistics(state: StatisticsPayload): void {
  configureToneColors(state.learning_language);
  renderStatisticsView(state);
  setView("statistics");
}

function renderSentence(
  sentence: Sentence,
  previouslyRevealed: boolean,
  termBands: Readonly<Record<string, TermBand>>,
  dialogue = false,
): HTMLElement {
  const unit = document.createElement("span");
  unit.className = dialogue ? "sentence-unit sentence-dialogue" : "sentence-unit";
  for (let index = 0; index < sentence.runs.length; index += 1) {
    const run = sentence.runs[index];
    if (!run) continue;
    if (!run.term) {
      unit.append(document.createTextNode(run.text));
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "term";
    const termBand = termBands[run.term.key];
    attachTermBand(button, termBand);
    renderHanziTones(button, run.text, pronunciationForRun(run) ?? "");
    const readingCue = termBand ? `, ${termBand} reading cue` : "";
    button.setAttribute("aria-label", `${run.text}${readingCue}, show translation`);
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", () => revealTerm(button, sentence, run));

    const nextRun = sentence.runs[index + 1];
    if (!nextRun?.term) {
      const punctuation = splitClosingPunctuation(nextRun?.text ?? "");
      if (punctuation.attached) {
        const cluster = document.createElement("span");
        cluster.className = "punctuation-cluster";
        cluster.append(button, document.createTextNode(punctuation.attached));
        unit.append(cluster);
        if (punctuation.remainder) {
          unit.append(document.createTextNode(punctuation.remainder));
        }
        index += 1;
        continue;
      }
    }
    unit.append(button);
  }

  const grammarOccurrences = visibleGrammarOccurrences(
    sentence.grammar,
    comfortableGrammarConstructionKeys,
  );
  if (grammarOccurrences.length > 0) {
    const grammarButton = document.createElement("button");
    grammarButton.type = "button";
    grammarButton.className = "grammar-reveal";
    grammarButton.dataset.readerCopyIgnore = "true";
    grammarButton.setAttribute("aria-controls", "grammar-popover");
    grammarButton.setAttribute("aria-expanded", "false");
    const count = grammarOccurrences.length;
    grammarButton.setAttribute(
      "aria-label",
      `${count} grammar ${plural(count, "note")} in this sentence`,
    );
    grammarButton.title = count === 1 ? "Grammar note" : `${count} grammar notes`;
    grammarButton.append(textElement("span", "grammar-reveal-mark", "G"));
    if (count > 1) {
      grammarButton.append(textElement("span", "grammar-reveal-count", String(count)));
    }
    if (grammarOccurrences.some((occurrence) => revealedGrammarOccurrences.has(occurrence.key))) {
      grammarButton.dataset.revealed = "true";
    }
    grammarButton.addEventListener("click", (event) => {
      event.stopPropagation();
      if (grammarHelpAnchor === grammarButton && !grammarPopover.hidden) {
        closeGrammarHelp();
        return;
      }
      openGrammarHelp(grammarButton, sentence, grammarOccurrences);
    });
    unit.append(grammarButton);
  }

  if (!sentence.translation) {
    unit.append(sentenceSeparator());
    return unit;
  }
  const reveal = document.createElement("button");
  reveal.type = "button";
  reveal.className = "sentence-reveal";
  reveal.dataset.readerCopyIgnore = "true";
  const icon = textElement("span", "", "A·文");
  icon.setAttribute("aria-hidden", "true");
  reveal.append(icon);
  reveal.setAttribute("aria-controls", "sentence-translation-popover");
  reveal.setAttribute("aria-expanded", "false");
  setSentenceButtonState(reveal, previouslyRevealed, false);

  const recordReveal = (): void => {
    if (sentenceReveals.reveal(sentence.key) && lesson) {
      recordLearningEvent("translation.revealed", {
        sentence_key: sentence.key,
        scope: "sentence",
      });
    }
    setSentenceButtonState(reveal, true, sentenceHelpAnchor === reveal);
  };

  const toggle = (): void => {
    setSentenceSelectionMode(false);
    positionSentenceControl(reveal, unit);
    if (sentenceHelpAnchor === reveal && !sentencePopover.hidden) {
      closeSentenceHelp();
      return;
    }
    openSentenceHelp(
      reveal,
      unit,
      sentence.runs.map((run) => run.text).join(""),
      sentence.translation ?? "",
    );
    if (!coarsePointer.matches) {
      recordReveal();
      return;
    }
    // Only count touch help that stays open: a quick open-and-close was a mistap.
    setSentenceButtonState(reveal, sentenceReveals.has(sentence.key), true);
    pendingSentenceEvidence = window.setTimeout(() => {
      pendingSentenceEvidence = null;
      if (sentenceHelpAnchor === reveal && !sentencePopover.hidden) recordReveal();
    }, TOUCH_SENTENCE_HELP_GRACE_MS);
  };

  let pressTimer: number | null = null;
  let pressOrigin: { x: number; y: number } | null = null;
  let suppressNextClick = false;
  const cancelPress = (): void => {
    if (pressTimer !== null) window.clearTimeout(pressTimer);
    pressTimer = null;
    pressOrigin = null;
  };
  unit.addEventListener(
    "pointerdown",
    (event) => {
      suppressNextClick = false;
      if (event.pointerType !== "touch" || sentenceSelectionMode) return;
      const target = event.target;
      if (target instanceof Element && target.closest(".sentence-reveal, .grammar-reveal")) return;
      cancelPress();
      pressOrigin = { x: event.clientX, y: event.clientY };
      pressTimer = window.setTimeout(() => {
        pressTimer = null;
        pressOrigin = null;
        suppressNextClick = true;
        window.getSelection()?.removeAllRanges();
        toggle();
      }, SENTENCE_LONG_PRESS_MS);
    },
    { passive: true },
  );
  unit.addEventListener(
    "pointermove",
    (event) => {
      if (
        pressOrigin &&
        Math.hypot(event.clientX - pressOrigin.x, event.clientY - pressOrigin.y) >
          LONG_PRESS_MOVE_TOLERANCE_PX
      ) {
        cancelPress();
      }
    },
    { passive: true },
  );
  unit.addEventListener("pointerup", cancelPress, { passive: true });
  unit.addEventListener("pointercancel", cancelPress, { passive: true });
  unit.addEventListener("contextmenu", (event) => {
    if (suppressNextClick || pressTimer !== null) event.preventDefault();
  });

  reveal.addEventListener("click", (event) => {
    event.stopPropagation();
    toggle();
  });
  reveal.addEventListener("focus", () => positionSentenceControl(reveal, unit));
  unit.addEventListener("pointermove", (event) => {
    positionSentenceControl(reveal, unit, event.clientY);
  });
  unit.addEventListener(
    "click",
    (event) => {
      if (suppressNextClick) {
        // The finger lifted after a long press; that release must not also reveal a word.
        suppressNextClick = false;
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      const target = event.target;
      const targetIsTerm = target instanceof Element && Boolean(target.closest(".term"));
      const targetIsControl =
        target instanceof Element && Boolean(target.closest(".sentence-reveal, .grammar-reveal"));
      if (
        !shouldHandleSentenceClick(
          sentenceSelectionMode,
          targetIsTerm,
          targetIsControl,
          coarsePointer.matches,
        )
      ) {
        return;
      }
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;
      event.preventDefault();
      event.stopPropagation();
      toggle();
    },
    { capture: true },
  );
  unit.append(reveal, sentenceSeparator());
  return unit;
}

function sentenceSeparator(): HTMLElement {
  const separator = document.createElement("span");
  separator.className = "sentence-separator";
  separator.dataset.readerCopyIgnore = "true";
  separator.setAttribute("aria-hidden", "true");
  separator.textContent = " ";
  return separator;
}

function setSentenceSelectionMode(enabled: boolean): void {
  sentenceSelectionMode = enabled;
  sentenceHelpModeButton.setAttribute("aria-pressed", String(enabled));
  sentenceHelpModeButton.setAttribute(
    "aria-label",
    enabled ? "Cancel sentence translation selection" : "Choose a sentence to translate",
  );
  byId("sentence-help-mode-label").textContent = enabled ? "Tap a sentence" : "Sentence";
}

function setSentenceButtonState(
  button: HTMLButtonElement,
  revealed: boolean,
  expanded: boolean,
): void {
  button.dataset.revealed = String(revealed);
  button.setAttribute("aria-expanded", String(expanded));
  const label = expanded
    ? "Hide sentence translation"
    : revealed
      ? "Review sentence translation"
      : "Show sentence translation";
  button.setAttribute("aria-label", label);
  button.title = label;
}

function closestSentenceFragment(fragments: readonly DOMRect[], pointerY: number): number {
  let closest = 0;
  let closestDistance = Number.POSITIVE_INFINITY;
  for (const [index, fragment] of fragments.entries()) {
    const distance =
      pointerY < fragment.top
        ? fragment.top - pointerY
        : pointerY > fragment.bottom
          ? pointerY - fragment.bottom
          : 0;
    if (distance < closestDistance) {
      closest = index;
      closestDistance = distance;
    }
  }
  return closest;
}

function positionSentenceControl(
  button: HTMLButtonElement,
  unit: HTMLElement,
  pointerY?: number,
): void {
  const fragments = Array.from(unit.getClientRects()).filter((rect) => rect.width > 0);
  const storedIndex = Number(button.dataset.fragmentIndex ?? 0);
  const fallbackIndex =
    Number.isInteger(storedIndex) && storedIndex >= 0 && storedIndex < fragments.length
      ? storedIndex
      : 0;
  const fragmentIndex =
    pointerY === undefined ? fallbackIndex : closestSentenceFragment(fragments, pointerY);
  const fragment = fragments[fragmentIndex];
  if (!fragment) return;
  button.dataset.fragmentIndex = String(fragmentIndex);
  const content = lessonContent.getBoundingClientRect();
  const size = 28;
  let left = content.left - size;
  if (left < 4) left = Math.min(window.innerWidth - size - 4, content.right - size);
  button.style.setProperty("--sentence-help-left", `${Math.round(left)}px`);
  button.style.setProperty("--sentence-help-top", `${Math.round(fragment.top + 4)}px`);
}

function openSentenceHelp(
  anchor: HTMLButtonElement,
  unit: HTMLElement,
  source: string,
  translation: string,
): void {
  closeGloss();
  closeGrammarHelp();
  if (sentenceHelpAnchor && sentenceHelpAnchor !== anchor) {
    setSentenceButtonState(
      sentenceHelpAnchor,
      sentenceHelpAnchor.dataset.revealed === "true",
      false,
    );
  }
  sentenceHelpAnchor = anchor;
  sentenceHelpUnit = unit;
  sentencePopoverSource.textContent = source;
  sentencePopoverText.textContent = translation;
  sentencePopover.hidden = false;
  setSentenceButtonState(anchor, true, true);
  positionSentenceHelp();
}

function positionSentenceHelp(): void {
  if (!sentenceHelpAnchor || !sentenceHelpUnit || sentencePopover.hidden) return;
  positionSentenceControl(sentenceHelpAnchor, sentenceHelpUnit);
}

function closeSentenceHelp(returnFocus = false): void {
  if (pendingSentenceEvidence !== null) {
    window.clearTimeout(pendingSentenceEvidence);
    pendingSentenceEvidence = null;
  }
  const anchor = sentenceHelpAnchor;
  if (anchor) {
    setSentenceButtonState(anchor, anchor.dataset.revealed === "true", false);
  }
  sentenceHelpAnchor = null;
  sentenceHelpUnit = null;
  sentencePopover.hidden = true;
  if (returnFocus) anchor?.focus({ preventScroll: true });
}

function revealTerm(anchor: HTMLButtonElement, sentence: Sentence, run: Run): void {
  if (!lesson || !run.term) return;
  closeSentenceHelp();
  closeGrammarHelp();
  if (glossAnchor) glossAnchor.setAttribute("aria-expanded", "false");
  glossAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  byId("gloss-lemma").textContent = run.term.lemma;
  byId("gloss-pos").textContent = run.term.pos;
  byId("gloss-text").textContent = run.term.gloss;
  const pronunciation = byId("gloss-pronunciation");
  const pronunciationText = pronunciationForRun(run) ?? "";
  if (isChineseLanguage(lesson.learning_language)) {
    renderPinyin(
      pronunciation,
      pronunciationText,
      run.pronunciation === undefined ? run.term.lemma : run.text,
    );
  } else {
    pronunciation.textContent = pronunciationText;
  }
  pronunciation.hidden = !pronunciationText;
  popover.hidden = false;
  positionGloss();
  recordLearningEvent("term.revealed", {
    term_key: run.term.key,
    sentence_key: sentence.key,
  });
}

function positionPopover(
  anchorElement: HTMLElement,
  popoverElement: HTMLElement,
  offset: number,
): void {
  const anchor = anchorElement.getBoundingClientRect();
  const box = popoverElement.getBoundingClientRect();
  const margin = 12;
  let left = anchor.left + anchor.width / 2 - box.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - box.width - margin));
  let top = anchor.bottom + offset;
  if (top + box.height > window.innerHeight - margin) top = anchor.top - box.height - offset;
  popoverElement.style.left = `${left}px`;
  popoverElement.style.top = `${Math.max(margin, top)}px`;
}

function positionGloss(): void {
  if (!glossAnchor || popover.hidden) return;
  positionPopover(glossAnchor, popover, 10);
}

function closeGloss(): void {
  glossAnchor?.setAttribute("aria-expanded", "false");
  glossAnchor = null;
  popover.hidden = true;
}

function openGrammarHelp(
  anchor: HTMLButtonElement,
  sentence: Sentence,
  occurrences: readonly Sentence["grammar"][number][],
): void {
  closeGloss();
  closeSentenceHelp();
  if (grammarHelpAnchor && grammarHelpAnchor !== anchor) {
    grammarHelpAnchor.setAttribute("aria-expanded", "false");
  }
  grammarHelpAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  grammarPopoverContent.replaceChildren(
    ...occurrences.map((occurrence) => renderGrammarOccurrenceHelp(sentence, occurrence)),
  );
  grammarPopover.hidden = false;
  positionGrammarHelp();
}

function renderGrammarOccurrenceHelp(
  sentence: Sentence,
  occurrence: Sentence["grammar"][number],
): HTMLElement {
  const definition = readerGrammarCatalog.get(occurrence.construction_key);
  const item = document.createElement("section");
  item.className = "grammar-help-item";
  const header = document.createElement("div");
  header.className = "grammar-help-heading";
  header.append(
    textElement("span", "grammar-help-category", definition?.category ?? "Grammar"),
    textElement("h2", "grammar-help-label", definition?.label ?? occurrence.construction_key),
  );
  const source = textElement(
    "p",
    "grammar-help-source",
    sentence.runs
      .slice(occurrence.run_start, occurrence.run_end)
      .map((run) => run.text)
      .join(""),
  );
  source.dir = "auto";

  const explanation = document.createElement("div");
  explanation.className = "grammar-help-explanation";
  explanation.hidden = true;
  explanation.append(
    textElement(
      "p",
      "grammar-help-description",
      definition?.description ?? "This construction is not in the reader catalog.",
    ),
  );
  if (occurrence.note) {
    const note = document.createElement("p");
    note.className = "grammar-help-note";
    note.append(textElement("strong", "", "Here: "), document.createTextNode(occurrence.note));
    explanation.append(note);
  }

  const alreadyRevealed = revealedGrammarOccurrences.has(occurrence.key);
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "grammar-help-toggle";
  toggle.dataset.revealed = String(alreadyRevealed);
  toggle.setAttribute("aria-expanded", "false");
  toggle.textContent = alreadyRevealed ? "Review explanation" : "Show explanation";
  toggle.addEventListener("click", () => {
    const show = explanation.hidden;
    explanation.hidden = !show;
    toggle.setAttribute("aria-expanded", String(show));
    toggle.textContent = show
      ? "Hide explanation"
      : revealedGrammarOccurrences.has(occurrence.key)
        ? "Review explanation"
        : "Show explanation";
    if (show && !revealedGrammarOccurrences.has(occurrence.key)) {
      revealedGrammarOccurrences.add(occurrence.key);
      toggle.dataset.revealed = "true";
      grammarHelpAnchor?.setAttribute("data-revealed", "true");
      recordLearningEvent("translation.revealed", {
        scope: "grammar",
        sentence_key: sentence.key,
        occurrence_key: occurrence.key,
        construction_key: occurrence.construction_key,
      });
    }
    positionGrammarHelp();
  });

  item.append(header, source, toggle, explanation);
  return item;
}

function positionGrammarHelp(): void {
  if (!grammarHelpAnchor || grammarPopover.hidden) return;
  positionPopover(grammarHelpAnchor, grammarPopover, 9);
}

function closeGrammarHelp(returnFocus = false): void {
  const anchor = grammarHelpAnchor;
  anchor?.setAttribute("aria-expanded", "false");
  grammarHelpAnchor = null;
  grammarPopover.hidden = true;
  if (returnFocus) anchor?.focus({ preventScroll: true });
}

function toggleFullTranslation(show: boolean): void {
  if (!lesson) return;
  fullPanel.hidden = !show;
  fullButton.textContent = show ? "Hide full translation" : "Show full translation";
  fullButton.setAttribute("aria-expanded", String(show));
  if (show && !fullTranslationRevealed) {
    fullTranslationRevealed = true;
    recordLearningEvent("translation.revealed", { scope: "lesson" });
  }
  if (show) fullPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

const ratings = [
  [-1, "Not for me"],
  [1, "Good lesson"],
] as const;

function renderRatings(): void {
  const container = byId("rating-options");
  container.replaceChildren();
  for (const [value, label] of ratings) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rating-button";
    button.dataset.rating = String(value);
    button.setAttribute("aria-pressed", "false");
    button.append(textElement("strong", "rating-number", value === 1 ? "+" : "−"));
    button.append(textElement("span", "rating-label", label));
    button.addEventListener("click", () => selectRating(value));
    container.append(button);
  }
}

function selectRating(value: -1 | 1): void {
  rating = value;
  for (const button of document.querySelectorAll<HTMLButtonElement>(".rating-button")) {
    button.setAttribute("aria-pressed", String(button.dataset.rating === String(value)));
  }
}

function selectedFeedback(): string[] {
  return Array.from(
    feedbackDetails.querySelectorAll<HTMLInputElement>('input[name="lesson-feedback"]:checked'),
    (input) => input.value,
  );
}

async function completeLesson(): Promise<void> {
  if (!lesson || !readerModeRecordsEvidence(readerMode)) return;
  completeButton.disabled = true;
  completeButton.textContent = "Saving…";
  recordLearningEvent("lesson.completed", {
    active_seconds: activeTimer.seconds(),
    completion_ratio: 1,
  });
  const feedback = selectedFeedback();
  if (rating !== null || feedback.length > 0) {
    recordLearningEvent("lesson.rated", { rating, feedback });
  }
  try {
    await events.flush();
  } catch (error) {
    completeButton.disabled = false;
    completeButton.textContent = "Finish & next";
    showSyncWarning(error);
    return;
  }
  await loadLesson().catch(showError);
}

function resetReaderSkipButtons(): void {
  readerView.inert = false;
  readerView.removeAttribute("aria-busy");
  for (const button of skipLessonButtons) {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = "Skip for now";
  }
}

async function skipCurrentLesson(): Promise<void> {
  if (
    !lesson ||
    lessonId === null ||
    readerMode !== "standard" ||
    skipLessonButtons.every((button) => button.hidden || button.disabled)
  ) {
    return;
  }
  const skippedLessonId = lessonId;
  const skippedTitle = lesson.title;
  for (const button of skipLessonButtons) {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = "Skipping…";
  }
  readerView.inert = true;
  readerView.setAttribute("aria-busy", "true");
  activeTimer.pause();
  stopLessonAudio();
  closeGloss();
  closeSentenceHelp();
  closeGrammarHelp();
  setSentenceSelectionMode(false);
  toggleFullTranslation(false);

  try {
    await events.flush();
    await postTextDisposition(skippedLessonId, "skip");
  } catch (error) {
    activeTimer.resume();
    resetReaderSkipButtons();
    showActionStatus(`Couldn’t skip “${skippedTitle}”. This text is still open; please try again.`);
    console.error(error);
    return;
  }

  showActionStatus(`Skipped “${skippedTitle}” for now. Opening the next text…`);
  window.history.replaceState(null, "", appPath(`/p/${encodeURIComponent(profileId)}`));
  try {
    await loadLesson();
    if (!readerView.hidden) byId("lesson-title").focus({ preventScroll: true });
  } catch (error) {
    showError(error);
  }
}

function emptyProgress(): ReaderProgress {
  return {
    started: false,
    completed: false,
    rating: null,
    revealed_term_keys: [],
    revealed_sentence_keys: [],
    revealed_grammar_occurrence_keys: [],
    full_translation_revealed: false,
  };
}

async function loadLesson(
  showLoading = true,
  launch: ReaderLaunch = { mode: "standard" },
): Promise<void> {
  readerMode = launch.mode;
  if (showLoading) {
    byId("loading-message").textContent =
      launch.mode === "preview"
        ? "Opening a no-progress preview…"
        : launch.mode === "reread"
          ? "Starting a fresh reread…"
          : "Preparing your next lesson…";
    setView("loading");
  }
  const payload = await getJson(
    readerApiPath(profileApi, launch),
    launch.mode === "preview" ? "Preview" : "Reader",
  );
  if (launch.mode === "preview") {
    const preview = parseTextPreview(payload);
    renderLesson(
      {
        lesson_id: preview.lesson_id,
        lesson: preview.lesson,
        progress: emptyProgress(),
        term_bands: preview.term_bands,
        grammar_catalog: preview.grammar_catalog,
        comfortable_grammar_construction_keys: preview.comfortable_grammar_construction_keys,
      },
      "preview",
      true,
    );
    completeButton.textContent = "Finish & next";
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }
  const state = parseReader(payload);
  configureToneColors(state.profile.learning_language);
  if (!state.lesson || state.lesson_id === null) {
    showEmpty(state.generation_status);
    return;
  }
  renderLesson(
    {
      lesson_id: state.lesson_id,
      lesson: state.lesson,
      progress: state.progress,
      term_bands: state.term_bands,
      grammar_catalog: state.grammar_catalog,
      comfortable_grammar_construction_keys: state.comfortable_grammar_construction_keys,
    },
    launch.mode,
    launch.lessonId !== undefined,
  );
  completeButton.textContent = launch.mode === "reread" ? "Finish reread" : "Finish & next";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadTexts(showLoading = true): Promise<void> {
  if (showLoading) {
    byId("loading-message").textContent = "Opening your text library…";
    setView("loading");
  }
  const state = parseTextLibrary(await getJson(`${profileApi}/texts`, "Texts"));
  configureToneColors(state.learning_language);
  renderTextsView(state);
  setView("texts");
}

async function loadView(
  loadingMessage: string,
  resource: string,
  failureLabel: string,
  render: (payload: unknown) => void,
): Promise<void> {
  byId("loading-message").textContent = loadingMessage;
  setView("loading");
  render(await getJson(`${profileApi}/${resource}`, failureLabel));
}

function loadWords(): Promise<void> {
  return loadView("Gathering your vocabulary…", "words", "Words", (payload) =>
    renderWords(parseWords(payload)),
  );
}

function loadCharacters(): Promise<void> {
  return loadView("Gathering your character exposure…", "characters", "Characters", (payload) =>
    renderCharacters(parseCharacters(payload)),
  );
}

function loadGrammar(): Promise<void> {
  return loadView("Gathering your grammar evidence…", "grammar", "Grammar", (payload) =>
    renderGrammar(parseGrammar(payload)),
  );
}

function loadStatistics(): Promise<void> {
  return loadView("Gathering your reading statistics…", "statistics", "Statistics", (payload) =>
    renderStatistics(parseStatistics(payload)),
  );
}

async function bootstrap(): Promise<void> {
  setView("loading");
  const activation = await loadActivation();
  if (!activation.active) {
    if (!readOnlySnapshot) {
      events.clear();
      lessonSessions.clear();
      activeTimeStore.clear();
    }
    renderActivation();
    return;
  }
  const launch = parseReaderLaunch(window.location.search);
  readerMode = launch.mode;
  if (launch.mode === "preview") {
    await loadLesson(true, launch);
    return;
  }
  if (initialView === "settings") {
    await loadSettings();
    setView("settings");
    void events.flush().catch(showSyncWarning);
    return;
  }
  await events.flush();
  if (initialView === "texts") await loadTexts();
  else if (initialView === "words") await loadWords();
  else if (initialView === "characters") await loadCharacters();
  else if (initialView === "grammar") await loadGrammar();
  else if (initialView === "statistics") await loadStatistics();
  else await loadLesson(true, launch);
}

function showError(error: unknown): void {
  clearLessonPoll();
  clearTextsPoll();
  const eyebrow = errorView.querySelector(".eyebrow");
  const heading = errorView.querySelector("h1");
  const viewCopy = VIEW_ERROR_COPY[initialView];
  const fallback = {
    eyebrow: readerMode === "preview" ? "Preview unavailable" : "Reader unavailable",
    heading: "We couldn’t open the lesson.",
  };
  const copy = viewCopy ?? fallback;
  if (eyebrow) eyebrow.textContent = copy.eyebrow;
  if (heading) heading.textContent = copy.heading;
  byId("retry-button").textContent = "Try again";
  byId("retry-button").hidden = false;
  byId("error-message").textContent =
    error instanceof Error ? error.message : "An unexpected error occurred.";
  setView("error");
}

function clearLessonPoll(): void {
  if (lessonPollTimer === null) return;
  window.clearTimeout(lessonPollTimer);
  lessonPollTimer = null;
}

function showEmpty(generationStatus: ReaderPayload["generation_status"]): void {
  clearLessonPoll();
  const eyebrow = errorView.querySelector(".eyebrow");
  const heading = errorView.querySelector("h1");
  const isPreparing = generationStatus === "pending" || generationStatus === "running";
  if (eyebrow) eyebrow.textContent = isPreparing ? "Lesson in progress" : "All caught up";
  if (heading) {
    heading.textContent = isPreparing
      ? "Your learning agent is preparing a new lesson."
      : "No lesson is ready yet.";
  }
  byId("error-message").textContent = isPreparing
    ? "This page will update automatically."
    : generationStatus === "failed"
      ? "The last preparation attempt failed. Try again when it has been resolved."
      : "Check again after a new lesson has been added.";
  byId("retry-button").textContent = "Check again";
  byId("retry-button").hidden = isPreparing;
  setView("error");
  if (isPreparing) {
    lessonPollTimer = window.setTimeout(() => {
      lessonPollTimer = null;
      void loadLesson(false).catch(showError);
    }, 5_000);
  }
}

function showSyncWarning(error: unknown): void {
  console.error(error);
  statusMessage.textContent = "Progress could not sync yet. We’ll retry with the next action.";
  statusMessage.hidden = false;
  window.setTimeout(() => {
    statusMessage.hidden = true;
  }, 4_000);
}

fullButton.addEventListener("click", () => toggleFullTranslation(fullPanel.hidden));
initLessonAudio({ lesson: () => lesson, lessonId: () => lessonId });
byId("close-translation-button").addEventListener("click", () => toggleFullTranslation(false));
byId("gloss-close").addEventListener("click", closeGloss);
byId("sentence-translation-close").addEventListener("click", () => closeSentenceHelp(true));
byId("grammar-popover-close").addEventListener("click", () => closeGrammarHelp(true));
sentenceHelpModeButton.addEventListener("click", () => {
  closeGloss();
  closeGrammarHelp();
  closeSentenceHelp();
  setSentenceSelectionMode(!sentenceSelectionMode);
});
toneColorToggle.addEventListener("click", () => {
  const enabled = toneColorToggle.getAttribute("aria-pressed") !== "true";
  toneColorPreference.set(enabled);
  configureToneColors(currentLearningLanguage);
});
for (const input of document.querySelectorAll<HTMLInputElement>('input[name="color-theme"]')) {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    const theme = parseThemeMode(input.value);
    if (!theme) return;
    setTheme(persistThemePreference(window.localStorage, theme), true);
  });
}
initSettingsView();
byId("retry-button").addEventListener("click", () => void bootstrap().catch(showError));
completeButton.addEventListener("click", () => void completeLesson());
for (const button of skipLessonButtons) {
  button.addEventListener("click", () => void skipCurrentLesson());
}
initTextsView({ reload: () => loadTexts(false), onError: showError });
initActivationView();
byId("reset-active-time").addEventListener("click", () => activeTimer.reset());
initGrammarView();
initWordsView();
initCharactersView();
profileSwitch.addEventListener("change", () => {
  if (profileSwitch.value === profileId) return;
  stopLessonAudio(true);
  activeTimer.save();
  if (readerModeRecordsEvidence(readerMode)) events.sendOnExit();
  const suffix = initialView === "reading" ? "" : `?view=${initialView}`;
  window.location.assign(appPath(`/p/${encodeURIComponent(profileSwitch.value)}${suffix}`));
});
document.addEventListener("pointerdown", (event) => {
  const target = event.target;
  if (target instanceof Node && !popover.contains(target) && !glossAnchor?.contains(target)) {
    closeGloss();
  }
  if (
    target instanceof Node &&
    !sentencePopover.contains(target) &&
    !sentenceHelpUnit?.contains(target)
  ) {
    closeSentenceHelp();
  }
  if (
    target instanceof Node &&
    !grammarPopover.contains(target) &&
    !grammarHelpAnchor?.contains(target)
  ) {
    closeGrammarHelp();
  }
});
document.addEventListener("copy", (event) => {
  if (!lesson || readerView.hidden) return;
  copyReaderSelection(
    event,
    window.getSelection(),
    byId("lesson-title"),
    lessonContent,
    lesson.learning_language,
  );
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeGloss();
    closeSentenceHelp(true);
    closeGrammarHelp(true);
    setSentenceSelectionMode(false);
  }
});
window.addEventListener("resize", positionGloss);
window.addEventListener("resize", positionSentenceHelp);
window.addEventListener("resize", positionGrammarHelp);
window.addEventListener(
  "scroll",
  () => {
    positionGloss();
    positionSentenceHelp();
    positionGrammarHelp();
  },
  { passive: true },
);
window.addEventListener("pagehide", () => {
  stopLessonAudio();
  activeTimer.pagehide();
  if (readerModeRecordsEvidence(readerMode)) events.sendOnExit();
});

void bootstrap().catch(showError);
