/** The texts view: library groups, queue ordering, dispositions, topic requests, and polling. */

import { showActionStatus } from "./action-status.js";
import { postJson, profileApi, profileId } from "./api.js";
import { byId, summaryStat, textElement } from "./dom.js";
import { formatLibraryDate, formatLibraryDateTime } from "./library-date.js";
import { readerLevelPresentation } from "./reader-level.js";
import { textActionHref } from "./reader-mode.js";
import {
  setTextCardActionPending,
  type TextDisposition,
  textDispositionPath,
  textDispositionPayload,
} from "./text-disposition.js";
import {
  orderedReadyTexts,
  queueNeighborLessonId,
  readyQueuePositionLabel,
  type TextQueueMoveDirection,
  textQueueMovePath,
  textQueueMovePayload,
} from "./text-queue.js";
import {
  parseTextRequest,
  type TextLibraryItem,
  type TextLibraryPayload,
  type TextPreparationItem,
  type TextRequestItem,
  type TextStatus,
} from "./texts.contract.js";

let textsPollTimer: number | null = null;

/** What the texts view needs from the app shell. */
export interface TextsHooks {
  reload(): Promise<void>;
  onError(error: unknown): void;
}

let hooks: TextsHooks;

let textsPayload: TextLibraryPayload | null = null;

export function renderTextsView(state: TextLibraryPayload): void {
  textsPayload = state;
  const groups: Record<TextStatus, TextLibraryItem[]> = {
    queued: [],
    in_progress: [],
    skipped: [],
    read: [],
  };
  for (const item of state.texts) groups[item.status].push(item);
  const summary = byId("texts-summary");
  const summaryItems = [
    summaryStat("texts-summary-stat", groups.in_progress.length, "in progress"),
    summaryStat("texts-summary-stat", groups.queued.length, "ready"),
    summaryStat("texts-summary-stat", groups.skipped.length, "skipped"),
    summaryStat("texts-summary-stat", groups.read.length, "read"),
  ];
  if (state.preparations.length > 0) {
    summaryItems.unshift(summaryStat("texts-summary-stat", state.preparations.length, "preparing"));
  }
  summary.replaceChildren(...summaryItems);
  renderTextPreparations(state.preparations);
  for (const status of ["in_progress", "queued", "skipped", "read"] as const) {
    renderTextGroup(status, groups[status], state.learning_language);
  }
  renderTextRequests(state.requests);
  scheduleTextsPoll(state.preparations, state.requests);
}

const preparationStateLabels: Record<TextPreparationItem["state"], string> = {
  pending: "Queued",
  running: "Writing",
};

function renderTextPreparations(preparations: readonly TextPreparationItem[]): void {
  const group = byId("texts-preparing-group");
  const list = byId("texts-preparing-list");
  group.hidden = preparations.length === 0;
  byId("texts-preparing-count").textContent = String(preparations.length);
  if (preparations.length === 0) {
    list.replaceChildren();
    return;
  }
  list.replaceChildren(...preparations.map(renderTextPreparation));
}

function renderTextPreparation(preparation: TextPreparationItem): HTMLElement {
  const card = document.createElement("article");
  card.className = `text-card text-preparation-card text-preparation-${preparation.state}`;
  card.setAttribute("aria-busy", "true");
  const body = document.createElement("div");
  body.className = "text-card-body";
  const heading = document.createElement("div");
  heading.className = "text-card-heading";
  heading.append(
    textElement("span", "text-status", preparationStateLabels[preparation.state]),
    textElement(
      "h3",
      "",
      preparation.request_kind === "topic_request" ? "Requested text" : "Adaptive text",
    ),
  );
  body.append(heading);
  if (preparation.requested_topic !== null) {
    const topic = textElement("p", "text-card-topic", preparation.requested_topic);
    topic.dir = "auto";
    body.append(topic);
  }
  const meta = document.createElement("div");
  meta.className = "text-card-meta";
  meta.append(
    textElement(
      "span",
      "",
      preparation.state === "running"
        ? "The learning agent is preparing this now"
        : preparation.waiting_for_capacity
          ? "Waiting for model capacity; retrying automatically"
          : "Waiting for the learning agent",
    ),
    textElement("span", "", `Queued ${formatLibraryDateTime(preparation.created_at)}`),
  );
  body.append(meta);
  card.append(body);
  return card;
}

function renderTextGroup(
  status: TextStatus,
  texts: readonly TextLibraryItem[],
  learningLanguage: string,
): void {
  const suffix = status === "in_progress" ? "progress" : status;
  const selected = status === "queued" ? orderedReadyTexts(texts) : texts;
  byId(`texts-${suffix}-count`).textContent = String(selected.length);
  const list = byId(`texts-${suffix}-list`);
  if (selected.length === 0) {
    const empty = textElement(
      "p",
      "texts-group-empty",
      status === "in_progress"
        ? "No text is currently underway."
        : status === "queued"
          ? "No prepared text is waiting right now."
          : status === "skipped"
            ? "Texts you set aside will remain available here."
            : "Completed texts will collect here.",
    );
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(
    ...selected.map((item) =>
      renderTextCard(item, learningLanguage, status === "queued" ? selected : undefined),
    ),
  );
}

function renderTextCard(
  item: TextLibraryItem,
  learningLanguage: string,
  readyTexts?: readonly TextLibraryItem[],
): HTMLElement {
  const card = document.createElement("article");
  card.className = `text-card text-card-${item.status.replace("_", "-")}`;
  card.dataset.lessonId = String(item.id);
  const body = document.createElement("div");
  body.className = "text-card-body";
  const heading = document.createElement("div");
  heading.className = "text-card-heading";
  const queuePosition = readyQueuePositionLabel(item);
  if (queuePosition !== null) {
    const status = textElement("span", "text-status", queuePosition);
    status.setAttribute("aria-label", `Queue position ${queuePosition}`);
    status.title = `Queue position ${queuePosition}`;
    heading.append(status);
  }
  const title = textElement("h3", "", item.title);
  title.dir = "auto";
  heading.append(title);
  body.append(heading);
  if (item.topic) {
    const topic = textElement("p", "text-card-topic", item.topic);
    topic.dir = "auto";
    body.append(topic);
  }

  const level = readerLevelPresentation(learningLanguage, item.level, item.difficulty);
  const meta = document.createElement("div");
  meta.className = "text-card-meta";
  meta.append(
    textElement("span", "", level.label),
    textElement("span", "", `${Math.round(item.difficulty * 100)}% difficulty`),
    textElement("span", "", `${item.lexical_token_count.toLocaleString()} words`),
  );
  if (item.known_share !== null) {
    const known = textElement("span", "", `≈${Math.round(item.known_share * 100)}% known`);
    known.title = "Predicted share of running words you currently know";
    meta.append(known);
  }
  if (item.status === "read" && item.last_completed_at) {
    meta.append(textElement("span", "", `Read ${formatLibraryDate(item.last_completed_at)}`));
  } else if (item.status === "in_progress" && item.opened_at) {
    meta.append(textElement("span", "", `Opened ${formatLibraryDate(item.opened_at)}`));
  } else if (item.status === "skipped" && item.skipped_at) {
    meta.append(textElement("span", "", `Skipped ${formatLibraryDate(item.skipped_at)}`));
  }
  if (item.completion_count > 1) {
    meta.append(textElement("span", "", `${item.completion_count} completed sessions`));
  }
  body.append(meta);

  const actions = document.createElement("div");
  actions.className = "text-card-actions";
  const preview = document.createElement("a");
  preview.className = "button button-quiet text-card-action";
  preview.href = textActionHref(profileId, item.id, "preview");
  preview.textContent = "Peek";
  preview.title = "Open with no learning evidence recorded";
  if (item.status === "skipped") {
    const read = textDispositionButton(item, "restore", "Read now", true);
    read.classList.add("button-primary", "text-card-primary");
    read.classList.remove("button-quiet");
    const restore = textDispositionButton(item, "restore", "Return to queue");
    actions.append(read, restore, preview);
  } else {
    const primaryAction =
      item.status === "read" ? "reread" : item.status === "in_progress" ? "continue" : "read";
    const primary = document.createElement("a");
    primary.className = "button button-primary text-card-action text-card-primary";
    primary.href = textActionHref(profileId, item.id, primaryAction);
    primary.textContent =
      item.status === "read"
        ? "Read again"
        : item.status === "in_progress"
          ? "Continue"
          : "Read now";
    actions.append(primary, preview);
    if (item.status === "queued" || item.status === "in_progress") {
      actions.append(textDispositionButton(item, "skip", "Skip for now"));
    }
    if (item.status === "queued" && readyTexts) {
      actions.append(textQueueMoveControls(item, readyTexts));
    }
  }
  card.append(body, actions);
  return card;
}

function textQueueMoveControls(
  item: TextLibraryItem,
  readyTexts: readonly TextLibraryItem[],
): HTMLElement {
  const controls = document.createElement("div");
  controls.className = "text-queue-controls";
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", `Queue position controls for “${item.title}”`);
  for (const direction of ["up", "down"] as const) {
    const neighborLessonId = queueNeighborLessonId(readyTexts, item.id, direction);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-quiet text-queue-button";
    button.dataset.queueDirection = direction;
    button.textContent = direction === "up" ? "↑" : "↓";
    button.disabled = neighborLessonId === null;
    button.setAttribute("aria-label", `Move “${item.title}” ${direction} in the ready queue`);
    button.title =
      neighborLessonId === null
        ? direction === "up"
          ? "Already first in the ready queue"
          : "Already last in the ready queue"
        : `Move ${direction} in the ready queue`;
    if (neighborLessonId !== null) {
      button.addEventListener("click", () => {
        void changeReadyQueueOrder(item, direction, neighborLessonId, button);
      });
    }
    controls.append(button);
  }
  return controls;
}

function textDispositionButton(
  item: TextLibraryItem,
  disposition: TextDisposition,
  label: string,
  openAfterRestore = false,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-quiet text-card-action text-card-disposition";
  button.textContent = label;
  button.setAttribute(
    "aria-label",
    disposition === "skip"
      ? `Skip “${item.title}” for now`
      : openAfterRestore
        ? `Return “${item.title}” to the queue and read it now`
        : `Return “${item.title}” to the reading queue`,
  );
  button.addEventListener("click", () => {
    void changeLibraryTextDisposition(item, disposition, button, openAfterRestore);
  });
  return button;
}

async function postTextQueueMove(
  selectedLessonId: number,
  direction: TextQueueMoveDirection,
  neighborLessonId: number,
): Promise<Response> {
  return postJson(
    textQueueMovePath(profileApi, selectedLessonId),
    textQueueMovePayload(crypto.randomUUID(), direction, neighborLessonId),
  );
}

function focusQueueMoveControl(
  selectedLessonId: number,
  preferredDirection: TextQueueMoveDirection,
): void {
  const card = document.querySelector<HTMLElement>(
    `.text-card[data-lesson-id="${selectedLessonId}"]`,
  );
  if (!card) return;
  const fallbackDirection = preferredDirection === "up" ? "down" : "up";
  const target =
    card.querySelector<HTMLButtonElement>(
      `.text-queue-button[data-queue-direction="${preferredDirection}"]:not(:disabled)`,
    ) ??
    card.querySelector<HTMLButtonElement>(
      `.text-queue-button[data-queue-direction="${fallbackDirection}"]:not(:disabled)`,
    ) ??
    card.querySelector<HTMLElement>(".text-card-primary, .text-card-action");
  target?.focus({ preventScroll: true });
}

function restoreTextCardAction(
  button: HTMLButtonElement,
  card: HTMLElement | null,
  label?: string,
): void {
  setTextCardActionPending(button, false, label, card);
  button.focus({ preventScroll: true });
}

async function refreshTextsAndFocus(focusAfterRefresh: () => void): Promise<boolean> {
  try {
    await hooks.reload();
    focusAfterRefresh();
    return true;
  } catch (error) {
    console.error(error);
    return false;
  }
}

async function changeReadyQueueOrder(
  item: TextLibraryItem,
  direction: TextQueueMoveDirection,
  neighborLessonId: number,
  button: HTMLButtonElement,
): Promise<void> {
  const card = setTextCardActionPending(button, true);
  const refresh = () => refreshTextsAndFocus(() => focusQueueMoveControl(item.id, direction));

  let response: Response;
  try {
    response = await postTextQueueMove(item.id, direction, neighborLessonId);
  } catch (error) {
    if (await refresh()) {
      showActionStatus(
        `Couldn’t confirm the move for “${item.title}”. The current queue has been reloaded.`,
      );
    } else {
      restoreTextCardAction(button, card);
      showActionStatus("The queue could not be reached. Nothing shown here has changed.");
    }
    console.error(error);
    return;
  }

  if (!response.ok) {
    const refreshed = await refresh();
    if (!refreshed) restoreTextCardAction(button, card);
    showActionStatus(
      response.status === 409
        ? refreshed
          ? "The queue changed before that move. It has been refreshed; please try again."
          : "The queue changed, but it could not be refreshed. Reload before trying again."
        : refreshed
          ? `Couldn’t move “${item.title}”. The current queue has been reloaded.`
          : `Couldn’t move “${item.title}”. Reload the library before trying again.`,
    );
    console.error(`Queue move returned ${response.status}`);
    return;
  }

  if (await refresh()) {
    showActionStatus(
      `Moved “${item.title}” ${direction === "up" ? "up" : "down"} in the ready queue.`,
    );
    return;
  }
  restoreTextCardAction(button, card);
  showActionStatus("The move was saved, but the library could not refresh. Reload to see it.");
}

export async function postTextDisposition(
  selectedLessonId: number,
  disposition: TextDisposition,
): Promise<void> {
  const response = await postJson(
    textDispositionPath(profileApi, selectedLessonId, disposition),
    textDispositionPayload(crypto.randomUUID()),
  );
  if (!response.ok) {
    const label = disposition === "skip" ? "Skip" : "Restore";
    throw new Error(`${label} returned ${response.status}`);
  }
}

function focusLibraryTextCard(selectedLessonId: number): void {
  document
    .querySelector<HTMLElement>(
      `.text-card[data-lesson-id="${selectedLessonId}"] a, ` +
        `.text-card[data-lesson-id="${selectedLessonId}"] button`,
    )
    ?.focus();
}

async function changeLibraryTextDisposition(
  item: TextLibraryItem,
  disposition: TextDisposition,
  button: HTMLButtonElement,
  openAfterRestore: boolean,
): Promise<void> {
  const originalLabel = button.textContent ?? "";
  const pendingLabel = disposition === "skip" ? "Skipping…" : "Returning…";
  const card = setTextCardActionPending(button, true, pendingLabel);
  try {
    await postTextDisposition(item.id, disposition);
  } catch (error) {
    restoreTextCardAction(button, card, originalLabel);
    const action = disposition === "skip" ? "skip" : "return";
    showActionStatus(
      `Couldn’t ${action} “${item.title}”. The text has not moved; please try again.`,
    );
    console.error(error);
    return;
  }

  if (openAfterRestore) {
    showActionStatus(`Returned “${item.title}” to the queue. Opening it now…`);
    window.location.assign(textActionHref(profileId, item.id, "read"));
    return;
  }

  showActionStatus(
    disposition === "skip"
      ? `Skipped “${item.title}” for now. It remains available under Skipped.`
      : `Returned “${item.title}” to the reading queue.`,
  );
  if (await refreshTextsAndFocus(() => focusLibraryTextCard(item.id))) return;
  restoreTextCardAction(button, card, originalLabel);
  showActionStatus("The change was saved, but the library could not refresh. Reload to see it.");
}

const requestStateLabels: Record<TextRequestItem["state"], string> = {
  pending: "Queued",
  running: "Writing",
  completed: "Ready",
  failed: "Failed",
};

function renderTextRequests(requests: readonly TextRequestItem[]): void {
  const list = byId("topic-request-list");
  const recent = [...requests]
    .filter((request) => request.state === "completed" || request.state === "failed")
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    .slice(0, 5);
  if (recent.length === 0) {
    list.replaceChildren();
    return;
  }
  const heading = textElement("p", "topic-request-list-heading", "Recent requests");
  const rows = recent.map((request) => {
    const row = document.createElement("div");
    row.className = `topic-request-item topic-request-item-${request.state}`;
    const copy = document.createElement("div");
    copy.className = "topic-request-item-copy";
    copy.append(
      textElement("strong", "", request.topic),
      textElement("span", "", request.error ?? requestStateLabels[request.state]),
    );
    row.append(copy, textElement("span", "topic-request-state", requestStateLabels[request.state]));
    if (request.lesson_id !== null) {
      const open = document.createElement("a");
      open.className = "topic-request-open";
      open.href = textActionHref(profileId, request.lesson_id, "read");
      open.textContent = "Open text";
      row.append(open);
    }
    return row;
  });
  list.replaceChildren(heading, ...rows);
}

export function clearTextsPoll(): void {
  if (textsPollTimer === null) return;
  window.clearTimeout(textsPollTimer);
  textsPollTimer = null;
}

function scheduleTextsPoll(
  preparations: readonly TextPreparationItem[],
  requests: readonly TextRequestItem[],
): void {
  clearTextsPoll();
  const hasActiveRequest = requests.some(
    (request) => request.state === "pending" || request.state === "running",
  );
  if (preparations.length === 0 && !hasActiveRequest) {
    return;
  }
  textsPollTimer = window.setTimeout(() => {
    textsPollTimer = null;
    void hooks.reload().catch(hooks.onError);
  }, 5_000);
}

async function submitTopicRequest(): Promise<void> {
  const input = byId<HTMLInputElement>("topic-request-input");
  const submit = byId<HTMLButtonElement>("topic-request-submit");
  const status = byId("topic-request-status");
  const topic = input.value.trim();
  if (!topic) {
    input.setCustomValidity("Enter a topic or scenario.");
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  submit.disabled = true;
  submit.textContent = "Requesting…";
  status.textContent = "Sending this idea to the learning agent…";
  try {
    const response = await postJson(`${profileApi}/text-requests`, {
      request_id: crypto.randomUUID(),
      topic,
    });
    if (!response.ok) throw new Error(`Topic request returned ${response.status}`);
    const request = parseTextRequest(await response.json());
    input.value = "";
    status.textContent = `Requested “${request.topic}”. Its status will update here.`;
    if (textsPayload) {
      const requests = [
        request,
        ...textsPayload.requests.filter((item) => item.request_id !== request.request_id),
      ];
      textsPayload = { ...textsPayload, requests };
      renderTextRequests(requests);
      scheduleTextsPoll(textsPayload.preparations, requests);
    }
  } catch (error) {
    status.textContent =
      error instanceof Error ? error.message : "The topic request could not be sent.";
  } finally {
    submit.disabled = false;
    submit.textContent = "Request text";
  }
}

export function initTextsView(textsHooks: TextsHooks): void {
  hooks = textsHooks;
  byId("topic-request-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void submitTopicRequest();
  });
  byId("topic-request-input").addEventListener("input", (event) => {
    if (event.currentTarget instanceof HTMLInputElement) event.currentTarget.setCustomValidity("");
  });
}
