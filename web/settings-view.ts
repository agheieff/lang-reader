/** The settings view: voice choice and reading preferences. (Theme and tone colour live in app.ts.) */

import { getJson, postJson, profileApi } from "./api.js";
import { byId } from "./dom.js";
import { formatLibraryDate } from "./library-date.js";
import { parseReadingPreferences, type ReadingPreferences } from "./reading-preferences.js";
import { parseTtsSettings, type TtsSettings } from "./tts.js";

const preferencesText = byId<HTMLTextAreaElement>("preferences-text");

const preferencesMessage = byId<HTMLTextAreaElement>("preferences-message");

export const preferencesSave = byId<HTMLButtonElement>("preferences-save");

export const preferencesSend = byId<HTMLButtonElement>("preferences-send");

export const preferencesStatus = byId("preferences-status");

let preferencesRevision: number | null = null;

export const ttsVoiceSelect = byId<HTMLSelectElement>("tts-voice-select");

export const ttsSettingsStatus = byId("tts-settings-status");

function renderTtsSettings(settings: TtsSettings): void {
  ttsVoiceSelect.replaceChildren();
  if (!settings.supported || settings.voices.length === 0) {
    const option = document.createElement("option");
    option.textContent = `Local neural speech is not available for ${settings.learning_language}`;
    ttsVoiceSelect.append(option);
    ttsVoiceSelect.disabled = true;
  } else {
    for (const voice of settings.voices) {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = `${voice.label} — ${voice.description} · native: ${voice.native_language}${
        voice.recommended ? " · recommended" : ""
      }`;
      option.selected = voice.id === settings.selected_voice_id;
      ttsVoiceSelect.append(option);
    }
    ttsVoiceSelect.disabled = false;
  }
  byId("tts-language-note").textContent =
    settings.note ??
    (settings.supported
      ? `Qwen uses its ${settings.model_language ?? "multilingual"} mode for this profile.`
      : "Listen will use a compatible browser or operating-system voice when one exists.");
  ttsSettingsStatus.textContent = settings.provider_installed
    ? "Qwen3-TTS is installed. Missing audio is prepared one text at a time."
    : "The optional local Qwen runtime is not installed; Listen will use a system voice.";
}

function renderReadingPreferences(state: ReadingPreferences): void {
  preferencesRevision = state.revision_id;
  preferencesText.value = state.text;
  preferencesText.disabled = false;
  preferencesSave.disabled = false;
  const updated = state.updated_at
    ? `${state.source === "agent" ? "Updated by the agent" : "Saved by you"} ${formatLibraryDate(
        state.updated_at,
      )}`
    : "Starting notes from your profile interests";
  byId("preferences-meta").textContent = state.last_agent_reason
    ? `${updated}. Last agent change: ${state.last_agent_reason}`
    : updated;
  const pending = state.pending_messages.length;
  byId("preferences-pending").textContent = pending
    ? `${pending} message${pending === 1 ? "" : "s"} waiting for the agent`
    : "";
}

async function loadReadingPreferences(): Promise<void> {
  renderReadingPreferences(
    parseReadingPreferences(await getJson(`${profileApi}/reading-preferences`, "Preferences")),
  );
}

export async function saveReadingPreferences(): Promise<void> {
  preferencesSave.disabled = true;
  preferencesStatus.textContent = "Saving notes…";
  try {
    const response = await postJson(
      `${profileApi}/reading-preferences`,
      { text: preferencesText.value, expected_revision_id: preferencesRevision },
      "PUT",
    );
    if (response.status === 409) {
      throw new Error("The notes changed meanwhile (the agent may have updated them). Reload.");
    }
    if (!response.ok) throw new Error(`Saving notes returned ${response.status}`);
    renderReadingPreferences(parseReadingPreferences(await response.json()));
    preferencesStatus.textContent = "Notes saved. New texts will follow them.";
  } catch (error) {
    preferencesSave.disabled = false;
    preferencesStatus.textContent =
      error instanceof Error ? error.message : "The notes could not be saved.";
  }
}

export async function sendPreferenceMessage(): Promise<void> {
  const message = preferencesMessage.value.trim();
  if (!message) return;
  preferencesSend.disabled = true;
  preferencesStatus.textContent = "Sending…";
  try {
    const response = await postJson(`${profileApi}/reading-preferences/messages`, {
      message_id: crypto.randomUUID(),
      text: message,
    });
    if (!response.ok) throw new Error(`Sending returned ${response.status}`);
    renderReadingPreferences(parseReadingPreferences(await response.json()));
    preferencesMessage.value = "";
    preferencesStatus.textContent = "Sent. The agent folds it into the notes within a few minutes.";
  } catch (error) {
    preferencesStatus.textContent =
      error instanceof Error ? error.message : "The message could not be sent.";
  } finally {
    preferencesSend.disabled = false;
  }
}

async function loadTtsSettings(): Promise<void> {
  renderTtsSettings(parseTtsSettings(await getJson(`${profileApi}/tts`, "Voice settings")));
}

export async function updateTtsVoice(voiceId: string): Promise<void> {
  ttsVoiceSelect.disabled = true;
  ttsSettingsStatus.textContent = "Saving voice and scheduling your text library…";
  try {
    const response = await postJson(`${profileApi}/tts`, { voice_id: voiceId }, "PUT");
    if (!response.ok) throw new Error(`Voice update returned ${response.status}`);
    renderTtsSettings(parseTtsSettings(await response.json()));
    ttsSettingsStatus.textContent =
      "Voice saved. New and existing texts will use it as their audio becomes ready.";
  } catch (error) {
    ttsVoiceSelect.disabled = false;
    ttsSettingsStatus.textContent =
      error instanceof Error ? error.message : "The voice could not be saved.";
  }
}

export async function loadSettings(): Promise<void> {
  try {
    await loadTtsSettings();
  } catch (error) {
    ttsVoiceSelect.disabled = true;
    ttsSettingsStatus.textContent =
      error instanceof Error ? error.message : "Voice settings could not be loaded.";
  }
  try {
    await loadReadingPreferences();
  } catch (error) {
    preferencesStatus.textContent =
      error instanceof Error ? error.message : "Reading preferences could not be loaded.";
  }
}

export function initSettingsView(): void {
  ttsVoiceSelect.addEventListener("change", () => {
    if (ttsVoiceSelect.value) void updateTtsVoice(ttsVoiceSelect.value);
  });
  preferencesSave.addEventListener("click", () => void saveReadingPreferences());
  preferencesSend.addEventListener("click", () => void sendPreferenceMessage());
}
