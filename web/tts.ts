import type { Lesson } from "./contracts.js";
import { createJsonContract } from "./json-contract.js";
import { STORAGE_PREFIX } from "./storage-prefix.js";

export const AUDIO_AVAILABILITIES = [
  "ready",
  "preparing",
  "failed",
  "unavailable",
  "unsupported",
  "disabled",
] as const;
export type AudioAvailability = (typeof AUDIO_AVAILABILITIES)[number];

const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] as const;
export type PlaybackRate = (typeof PLAYBACK_RATES)[number];

export const AUDIO_STATUS_REASONS = [
  "ready",
  "queued",
  "generating",
  "retrying",
  "generation_timeout",
  "invalid_audio",
  "provider_stopped",
  "generation_failed",
  "provider_unavailable",
  "unsupported_language",
  "profile_inactive",
] as const;
export type AudioStatusReason = (typeof AUDIO_STATUS_REASONS)[number];

export interface TtsVoice {
  id: string;
  label: string;
  description: string;
  native_language: string;
  recommended: boolean;
}

export interface TtsSettings {
  provider: string;
  model: string;
  learning_language: string;
  model_language: string | null;
  supported: boolean;
  provider_installed: boolean;
  profile_active: boolean;
  selected_voice_id: string | null;
  voices: TtsVoice[];
  note: string | null;
}

export interface LessonAudioStatus {
  lesson_id: number;
  state: AudioAvailability;
  reason: AudioStatusReason;
  voice_id: string | null;
  voice_label: string | null;
  audio_url: string | null;
  message: string | null;
}

export interface BrowserSpeechFallback {
  available: boolean;
  learningLanguage: string;
  matchingVoiceName: string | null;
}

export interface PlaybackRateStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const {
  record,
  text,
  nullishText: optionalText,
  boolean,
} = createJsonContract("Invalid TTS response", { textRequirement: "must be text" });

function audioStatusReason(value: unknown): AudioStatusReason {
  if (typeof value !== "string" || !AUDIO_STATUS_REASONS.includes(value as AudioStatusReason)) {
    throw new Error("Invalid TTS response: unsupported audio reason");
  }
  return value as AudioStatusReason;
}

export function parseTtsSettings(value: unknown): TtsSettings {
  const settings = record(value, "settings");
  if (!Array.isArray(settings.voices)) {
    throw new Error("Invalid TTS response: voices must be a list");
  }
  return {
    provider: text(settings.provider, "provider"),
    model: text(settings.model, "model"),
    learning_language: text(settings.learning_language, "learning_language"),
    model_language: optionalText(settings.model_language, "model_language"),
    supported: boolean(settings.supported, "supported"),
    provider_installed: boolean(settings.provider_installed, "provider_installed"),
    profile_active: boolean(settings.profile_active, "profile_active"),
    selected_voice_id: optionalText(settings.selected_voice_id, "selected_voice_id"),
    voices: settings.voices.map((value, index) => {
      const voice = record(value, `voices[${index}]`);
      return {
        id: text(voice.id, `voices[${index}].id`),
        label: text(voice.label, `voices[${index}].label`),
        description: text(voice.description, `voices[${index}].description`),
        native_language: text(voice.native_language, `voices[${index}].native_language`),
        recommended: boolean(voice.recommended, `voices[${index}].recommended`),
      };
    }),
    note: optionalText(settings.note, "note"),
  };
}

export function parseLessonAudioStatus(value: unknown): LessonAudioStatus {
  const status = record(value, "audio status");
  const lessonId = status.lesson_id;
  if (!Number.isInteger(lessonId) || (lessonId as number) < 1) {
    throw new Error("Invalid TTS response: lesson_id must be positive");
  }
  const state = status.state;
  if (typeof state !== "string" || !AUDIO_AVAILABILITIES.includes(state as AudioAvailability)) {
    throw new Error("Invalid TTS response: unsupported audio state");
  }
  const audioUrl = optionalText(status.audio_url, "audio_url");
  if (state === "ready" && audioUrl === null) {
    throw new Error("Invalid TTS response: ready audio needs a URL");
  }
  return {
    lesson_id: lessonId as number,
    state: state as AudioAvailability,
    reason: audioStatusReason(status.reason),
    voice_id: optionalText(status.voice_id, "voice_id"),
    voice_label: optionalText(status.voice_label, "voice_label"),
    audio_url: audioUrl,
    message: optionalText(status.message, "message"),
  };
}

function playbackRateStorageKey(profileId: string): string {
  return `${STORAGE_PREFIX}${profileId}:playback-rate:v1`;
}

function parsePlaybackRate(value: unknown): PlaybackRate | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return PLAYBACK_RATES.includes(parsed as PlaybackRate) ? (parsed as PlaybackRate) : null;
}

export function readPlaybackRate(storage: PlaybackRateStorage, profileId: string): PlaybackRate {
  return parsePlaybackRate(storage.getItem(playbackRateStorageKey(profileId))) ?? 1;
}

export function persistPlaybackRate(
  storage: PlaybackRateStorage,
  profileId: string,
  value: unknown,
): PlaybackRate {
  const rate = parsePlaybackRate(value) ?? 1;
  storage.setItem(playbackRateStorageKey(profileId), String(rate));
  return rate;
}

export function browserFallbackMessage(fallback: BrowserSpeechFallback): string {
  if (!fallback.available) {
    return "No browser speech engine is available here.";
  }
  if (fallback.matchingVoiceName) {
    return `${fallback.matchingVoiceName} (system voice) is available now.`;
  }
  return (
    `No matching ${fallback.learningLanguage} system voice is listed; ` +
    "Listen will ask the browser to try its default voice."
  );
}

export function audioStatusMessage(
  status: LessonAudioStatus,
  fallback: BrowserSpeechFallback,
): string {
  const voice = status.voice_label ?? "selected voice";
  const localCopy: Record<AudioStatusReason, string> = {
    ready: `${voice} · Local Qwen audio`,
    queued: `Local ${voice} audio is queued; the audio worker prepares one text at a time.`,
    generating: `Local ${voice} audio is being generated now. Long texts can take several minutes.`,
    retrying: "The first local generation attempt failed; one automatic retry is queued.",
    generation_timeout: "Local generation timed out after two attempts.",
    invalid_audio: "The local model returned an audio file the browser could not use.",
    provider_stopped: "The local audio process stopped twice before this text was finished.",
    generation_failed: "Local audio generation failed after two attempts.",
    provider_unavailable: "The optional local Qwen runtime is not installed or enabled.",
    unsupported_language: "Local neural audio does not support this language.",
    profile_inactive: "Activate this language before using listening audio.",
  };
  if (status.reason === "ready") return localCopy.ready;
  if (status.reason === "profile_inactive") return localCopy.profile_inactive;
  return `${localCopy[status.reason]} ${browserFallbackMessage(fallback)}`;
}

export function audioStatusFetchFailure(status: number | null): string {
  if (status === null) {
    return "Could not reach the local app to check audio. Confirm that it is still running.";
  }
  if (status === 404) {
    return "This text's audio entry was not found. Refresh the page to load the current text.";
  }
  if (status >= 500) {
    return `The local app could not check audio because its server returned ${status}.`;
  }
  return `The local app rejected the audio status request (${status}).`;
}

export function localAudioPlayFailure(error: unknown): string {
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? String((error as { name: unknown }).name)
      : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "The browser blocked audio playback. Allow audio for this site, then press Listen again.";
  }
  if (name === "NotSupportedError") {
    return "The browser could not decode the cached local audio.";
  }
  if (name === "AbortError") {
    return "Local playback was interrupted before it started. Press Listen to try again.";
  }
  if (name === "NetworkError") {
    return "The cached audio could not be loaded from the local app.";
  }
  return "The cached local audio could not start.";
}

export function localAudioElementFailure(code: number | null): string {
  if (code === 1) return "Local audio loading was interrupted. Press Listen to try again.";
  if (code === 2) {
    return "The cached audio could not be loaded from the local app. Check that the app is running.";
  }
  if (code === 3) return "The browser received the local audio but could not decode it.";
  if (code === 4) return "This browser does not support the cached audio format.";
  return "The cached local audio could not be played.";
}

export function browserSpeechFailure(reason: string): string {
  if (reason === "not-allowed") {
    return "The browser blocked system speech. Allow audio for this site, then press Listen again.";
  }
  if (reason === "language-unavailable") {
    return "No system speech voice for this language is installed. Add one in your OS settings.";
  }
  if (reason === "voice-unavailable") {
    return "The selected system voice is no longer available. Reload after checking OS voices.";
  }
  if (reason === "audio-busy") {
    return "The system audio device is busy. Close other speech playback and try again.";
  }
  if (reason === "audio-hardware") {
    return "The browser could not access an audio output device. Check the system audio output.";
  }
  if (reason === "network") {
    return "The chosen system voice needs a network service that could not be reached.";
  }
  if (reason === "synthesis-unavailable" || reason === "synthesis-failed") {
    return "The browser's speech engine could not synthesize this text. Try another system voice.";
  }
  if (reason === "text-too-long") {
    return "The browser speech engine rejected this section as too long.";
  }
  if (reason === "invalid-argument") {
    return "The browser speech engine rejected this text or language setting.";
  }
  return "The system voice stopped before the text was finished.";
}

export function browserVoicePriorities(languageTag: string): string[] {
  const normalized = languageTag.toLocaleLowerCase();
  if (normalized === "zh-hant" || normalized.startsWith("zh-hant-")) {
    return ["zh-tw", "zh-hk", "zh"];
  }
  if (normalized === "zh-hans" || normalized.startsWith("zh-hans-")) {
    return ["zh-cn", "zh"];
  }
  if (normalized === "es-419" || normalized.startsWith("es-419-")) {
    return ["es-mx", "es-us", "es"];
  }
  const base = normalized.split("-", 1)[0] ?? normalized;
  return normalized === base ? [base] : [normalized, base];
}

export function chooseBrowserVoice(
  voices: readonly SpeechSynthesisVoice[],
  languageTag: string,
): SpeechSynthesisVoice | null {
  for (const desired of browserVoicePriorities(languageTag)) {
    const exact = voices
      .filter((voice) => voice.lang.toLocaleLowerCase() === desired)
      .sort((left, right) => Number(right.localService) - Number(left.localService));
    if (exact[0]) return exact[0];
    if (!desired.includes("-")) {
      const family = voices
        .filter((voice) => voice.lang.toLocaleLowerCase().startsWith(`${desired}-`))
        .sort((left, right) => Number(right.localService) - Number(left.localService));
      if (family[0]) return family[0];
    }
  }
  return null;
}

export function lessonSpeechChunks(lesson: Lesson): string[] {
  return [
    lesson.title,
    ...lesson.blocks.flatMap((block) =>
      block.sentences.map((sentence) => sentence.runs.map((run) => run.text).join("")),
    ),
  ].filter((chunk) => chunk.trim().length > 0);
}
