/** The grammar view: construction cards with search and sort. */

import { profileId } from "./api.js";
import { byId, plural, summaryStat, textElement } from "./dom.js";
import {
  CONSTRUCTION_STATUS_DETAILS,
  constructionStatus,
  DEFAULT_GRAMMAR_SORT,
  formatGrammarDue,
  formatGrammarStatus,
  GRAMMAR_SORTS,
  type GrammarConstruction,
  type GrammarSort,
  type GrammarState,
  selectGrammar,
} from "./grammar.js";
import { textActionHref } from "./reader-mode.js";

let grammarPayload: GrammarState | null = null;

let grammarSort: GrammarSort = DEFAULT_GRAMMAR_SORT;

export function renderGrammarView(state: GrammarState): void {
  grammarPayload = state;
  const statuses = state.constructions.map(constructionStatus);
  const directHelp = state.constructions.reduce(
    (total, construction) => total + construction.counted_help_sessions,
    0,
  );
  const inferredHelp = state.constructions.reduce(
    (total, construction) => total + construction.inferred_difficulty_signals,
    0,
  );
  byId("grammar-summary").replaceChildren(
    summaryStat("grammar-summary-stat", state.constructions.length, "constructions"),
    summaryStat(
      "grammar-summary-stat",
      statuses.filter((status) => status === "needs_attention").length,
      "need attention",
    ),
    summaryStat(
      "grammar-summary-stat",
      statuses.filter((status) => status === "comfortable").length,
      "comfortable",
    ),
    summaryStat("grammar-summary-stat", directHelp + inferredHelp, "help signals"),
  );
  renderGrammarCards();
}

function renderGrammarCards(): void {
  if (!grammarPayload) return;
  const selected = selectGrammar(
    grammarPayload.constructions,
    byId<HTMLInputElement>("grammar-search").value,
    grammarSort,
  );
  byId<HTMLSelectElement>("grammar-sort").value = grammarSort;
  byId("grammar-card-list").replaceChildren(...selected.map(renderGrammarCard));
  byId("grammar-empty").hidden = selected.length > 0;
}

function renderGrammarCard(construction: GrammarConstruction): HTMLElement {
  const card = document.createElement("article");
  const status = constructionStatus(construction);
  card.className = `grammar-card grammar-card-${status.replace("_", "-")}`;

  const heading = document.createElement("header");
  heading.className = "grammar-card-heading";
  const labels = document.createElement("div");
  labels.className = "grammar-card-labels";
  const statusLabel = textElement(
    "span",
    `grammar-status grammar-status-${status.replace("_", "-")}`,
    formatGrammarStatus(construction),
  );
  statusLabel.title = CONSTRUCTION_STATUS_DETAILS[status].description;
  labels.append(
    statusLabel,
    textElement("span", "grammar-category", construction.category),
    textElement(
      "span",
      "grammar-difficulty",
      `${Math.round(construction.difficulty * 100)}% level`,
    ),
  );
  heading.append(
    labels,
    textElement("h2", "grammar-card-title", construction.label),
    textElement("p", "grammar-card-description", construction.description),
  );

  const metrics = document.createElement("dl");
  metrics.className = "grammar-card-metrics";
  metrics.append(
    grammarMetric(
      "Met",
      construction.occurrence_count,
      `${construction.exposed_lesson_count} ${plural(construction.exposed_lesson_count, "text")}`,
    ),
    grammarMetric(
      "Help",
      construction.counted_help_sessions,
      construction.inferred_difficulty_signals > 0
        ? `+${construction.inferred_difficulty_signals} weak sentence ${plural(
            construction.inferred_difficulty_signals,
            "signal",
          )}`
        : "No sentence signals",
    ),
    grammarMetric(
      "Clean reads",
      construction.qualified_exposures,
      formatGrammarDue(construction.next_due_at),
    ),
    grammarMetric("Confidence", `${Math.round(construction.mastery * 100)}%`, "Estimate"),
  );

  const evidence = document.createElement("details");
  evidence.className = "grammar-evidence";
  const evidenceSummary = document.createElement("summary");
  evidenceSummary.textContent = "How this estimate was formed";
  const evidenceCopy = document.createElement("p");
  evidenceCopy.textContent = `${construction.counted_help_sessions} direct-help ${plural(
    construction.counted_help_sessions,
    "session",
  )}, ${construction.inferred_difficulty_signals} weak sentence-help ${plural(
    construction.inferred_difficulty_signals,
    "signal",
  )}, and ${construction.qualified_exposures} sufficiently read completion ${plural(
    construction.qualified_exposures,
    "signal",
  )}. Reopening the same explanation in one session does not add another penalty.`;
  evidence.append(evidenceSummary, evidenceCopy);
  if (construction.raw_help_count > construction.counted_help_sessions) {
    evidence.append(
      textElement(
        "p",
        "grammar-evidence-raw",
        `${construction.raw_help_count.toLocaleString()} raw explanation opens are retained for inspection.`,
      ),
    );
  }

  card.append(heading, metrics, evidence);
  if (construction.examples.length > 0) {
    const examples = document.createElement("section");
    examples.className = "grammar-examples";
    examples.append(textElement("h3", "", "From your texts"));
    const [first, ...rest] = construction.examples;
    if (first) examples.append(renderGrammarExample(first));
    if (rest.length > 0) {
      const more = document.createElement("details");
      more.className = "grammar-more-examples";
      const summary = document.createElement("summary");
      summary.textContent = `${rest.length} more ${plural(rest.length, "example")}`;
      more.append(summary, ...rest.map(renderGrammarExample));
      examples.append(more);
    }
    card.append(examples);
  }
  return card;
}

function grammarMetric(label: string, value: number | string, detail: string): HTMLElement {
  const item = document.createElement("div");
  item.append(
    textElement("dt", "", label),
    textElement("dd", "grammar-metric-value", String(value)),
    textElement("dd", "grammar-metric-detail", detail),
  );
  return item;
}

function renderGrammarExample(example: GrammarConstruction["examples"][number]): HTMLElement {
  const exampleCard = document.createElement("article");
  exampleCard.className = "grammar-example";
  const context = textElement("p", "grammar-example-source", example.text);
  context.dir = "auto";
  exampleCard.append(context);
  if (example.translation) {
    const translation = textElement("p", "grammar-example-translation", example.translation);
    translation.dir = "auto";
    exampleCard.append(translation);
  }
  if (example.note) exampleCard.append(textElement("p", "grammar-example-note", example.note));
  const footer = document.createElement("footer");
  footer.append(textElement("span", "grammar-example-title", example.title));
  const link = document.createElement("a");
  link.className = "grammar-example-link";
  link.href = textActionHref(profileId, example.lesson_id, "preview");
  link.textContent = "Peek at text";
  link.title = "Open this text without recording learning evidence";
  footer.append(link);
  exampleCard.append(footer);
  return exampleCard;
}

export function initGrammarView(): void {
  byId("grammar-search").addEventListener("input", renderGrammarCards);
  byId("grammar-sort").addEventListener("change", (event) => {
    if (!(event.currentTarget instanceof HTMLSelectElement)) return;
    const selected = event.currentTarget.value;
    if (!GRAMMAR_SORTS.includes(selected as GrammarSort)) return;
    grammarSort = selected as GrammarSort;
    renderGrammarCards();
  });
}
