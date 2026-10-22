import { profileId } from "./api.js";
import type { Lesson } from "./contracts.js";
import { localAudioElementFailure, persistPlaybackRate, readPlaybackRate } from "./tts.js";

/** Lesson audio: the local cached recording, with the browser voice as fallback. */

import { getJson, profileApi } from "./api.js";
import { byId } from "./dom.js";
import {
  audioStatusFetchFailure,
  audioStatusMessage,
  browserFallbackMessage,
  browserSpeechFailure,
  chooseBrowserVoice,
  type LessonAudioStatus,
  lessonSpeechChunks,
  localAudioPlayFailure,
  parseLessonAudioStatus,
} from "./tts.js";

const lessonAudio = byId<HTMLAudioElement>("lesson-audio");

const lessonAudioButton = byId<HTMLButtonElement>("lesson-audio-button");

const lessonAudioSpeed = byId<HTMLSelectElement>("lesson-audio-speed");

const lessonAudioStatus = byId("lesson-audio-status");

let currentAudioStatus: LessonAudioStatus | null = null;

/** The lesson currently shown, owned by the reader. */
export interface AudioSource {
  lesson(): Lesson | null;
  lessonId(): number | null;
}

let source: AudioSource;

let audioPollTimer: number | null = null;

let audioRequestVersion = 0;

let browserSpeechToken = 0;

let browserSpeechActive = false;

export function browserSpeechAvailable(): boolean {
  return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
}

function audioPlaybackRate(): number {
  const value = Number(lessonAudioSpeed.value);
  return Number.isFinite(value) && value >= 0.5 && value <= 2 ? value : 1;
}

export function browserSpeechFallback(learningLanguage: string) {
  const available = browserSpeechAvailable();
  const voice = available
    ? chooseBrowserVoice(window.speechSynthesis.getVoices(), learningLanguage)
    : null;
  return {
    available,
    learningLanguage,
    matchingVoiceName: voice?.name ?? null,
  };
}

class AudioStatusRequestError extends Error {
  constructor(readonly status: number) {
    super(`Audio status returned ${status}`);
  }
}

export function setAudioButton(playing: boolean): void {
  byId("lesson-audio-button-icon").textContent = playing ? "❚❚" : "▶";
  byId("lesson-audio-button-label").textContent = playing ? "Pause" : "Listen";
  lessonAudioButton.setAttribute(
    "aria-label",
    playing ? "Pause reading this text aloud" : "Listen to this text",
  );
}

export function clearAudioPoll(): void {
  if (audioPollTimer === null) return;
  window.clearTimeout(audioPollTimer);
  audioPollTimer = null;
}

function stopBrowserSpeech(): void {
  browserSpeechToken += 1;
  if (browserSpeechAvailable()) window.speechSynthesis.cancel();
  browserSpeechActive = false;
}

export function stopLessonAudio(resetPosition = false): void {
  stopBrowserSpeech();
  lessonAudio.pause();
  if (resetPosition && lessonAudio.readyState > 0) lessonAudio.currentTime = 0;
  setAudioButton(false);
}

function resetLessonAudio(): number {
  clearAudioPoll();
  stopLessonAudio(true);
  audioRequestVersion += 1;
  currentAudioStatus = null;
  lessonAudio.removeAttribute("src");
  lessonAudio.load();
  lessonAudioButton.disabled = true;
  lessonAudioStatus.textContent = "Checking listening audio…";
  return audioRequestVersion;
}

export async function configureLessonAudio(selectedLessonId: number): Promise<void> {
  const requestVersion = resetLessonAudio();
  await refreshLessonAudio(selectedLessonId, requestVersion);
}

async function refreshLessonAudio(selectedLessonId: number, requestVersion: number): Promise<void> {
  try {
    const status = parseLessonAudioStatus(
      await getJson(
        `${profileApi}/lessons/${selectedLessonId}/audio/status`,
        (failureStatus) => new AudioStatusRequestError(failureStatus),
      ),
    );
    if (
      requestVersion !== audioRequestVersion ||
      source.lessonId() !== selectedLessonId ||
      status.lesson_id !== selectedLessonId
    ) {
      return;
    }
    currentAudioStatus = status;
    const fallback =
      status.state === "disabled"
        ? {
            available: false,
            learningLanguage: source.lesson()?.learning_language ?? "this language",
            matchingVoiceName: null,
          }
        : browserSpeechFallback(source.lesson()?.learning_language ?? "this language");
    if (status.state === "ready" && status.audio_url) {
      lessonAudio.src = status.audio_url;
      lessonAudio.load();
      lessonAudioButton.disabled = false;
      lessonAudioStatus.textContent = audioStatusMessage(status, fallback);
      return;
    }
    lessonAudioButton.disabled = !fallback.available;
    lessonAudioStatus.textContent = audioStatusMessage(status, fallback);
    if (status.state === "preparing") {
      audioPollTimer = window.setTimeout(() => {
        audioPollTimer = null;
        void refreshLessonAudio(selectedLessonId, requestVersion);
      }, 5_000);
    }
  } catch (error) {
    if (requestVersion !== audioRequestVersion || source.lessonId() !== selectedLessonId) return;
    currentAudioStatus = null;
    const fallback = browserSpeechFallback(source.lesson()?.learning_language ?? "this language");
    lessonAudioButton.disabled = !fallback.available;
    const detail =
      error instanceof AudioStatusRequestError
        ? audioStatusFetchFailure(error.status)
        : error instanceof TypeError
          ? audioStatusFetchFailure(null)
          : "The local app returned an invalid audio status. Refresh the page to try again.";
    lessonAudioStatus.textContent = `${detail} ${browserFallbackMessage(fallback)}`;
  }
}

function speakWithBrowser(localFailure: string | null = null): void {
  const lesson = source.lesson();
  if (!lesson) return;
  if (currentAudioStatus?.state === "disabled") {
    lessonAudioStatus.textContent = "Activate this language before using listening audio.";
    return;
  }
  if (!browserSpeechAvailable()) {
    lessonAudioStatus.textContent = `${
      localFailure ? `${localFailure} ` : ""
    }No browser speech engine is available here.`;
    return;
  }
  stopBrowserSpeech();
  const token = browserSpeechToken;
  const chunks = lessonSpeechChunks(lesson);
  const learningLanguage = lesson.learning_language;
  const voice = chooseBrowserVoice(window.speechSynthesis.getVoices(), learningLanguage);
  let index = 0;
  browserSpeechActive = true;
  setAudioButton(true);
  const systemVoiceCopy = voice
    ? `${voice.name} · System voice${voice.localService ? "" : " (may use a network service)"}`
    : `No matching ${learningLanguage} system voice is listed; trying the browser default`;
  lessonAudioStatus.textContent = localFailure
    ? `${localFailure} ${systemVoiceCopy} is playing instead.`
    : systemVoiceCopy;

  const speakNext = (): void => {
    if (token !== browserSpeechToken) return;
    const chunk = chunks[index];
    if (chunk === undefined) {
      browserSpeechActive = false;
      setAudioButton(false);
      return;
    }
    index += 1;
    const utterance = new SpeechSynthesisUtterance(chunk);
    utterance.lang = voice?.lang ?? learningLanguage;
    utterance.rate = audioPlaybackRate();
    if (voice) utterance.voice = voice;
    utterance.onend = speakNext;
    utterance.onerror = (event) => {
      if (token !== browserSpeechToken) return;
      browserSpeechActive = false;
      setAudioButton(false);
      lessonAudioStatus.textContent = browserSpeechFailure(event.error);
    };
    try {
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      browserSpeechActive = false;
      setAudioButton(false);
      const blocked =
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError");
      lessonAudioStatus.textContent = browserSpeechFailure(
        blocked ? "not-allowed" : "synthesis-failed",
      );
    }
  };
  speakNext();
}

export async function toggleLessonAudio(): Promise<void> {
  if (currentAudioStatus?.state === "ready" && lessonAudio.src) {
    stopBrowserSpeech();
    if (!lessonAudio.paused) {
      lessonAudio.pause();
      return;
    }
    lessonAudio.playbackRate = audioPlaybackRate();
    try {
      await lessonAudio.play();
    } catch (error) {
      speakWithBrowser(localAudioPlayFailure(error));
    }
    return;
  }
  if (browserSpeechActive) {
    stopBrowserSpeech();
    setAudioButton(false);
  } else {
    speakWithBrowser();
  }
}

/** Called when the reader view is left: no audio may keep playing or polling. */
export function suspendLessonAudio(): void {
  clearAudioPoll();
  audioRequestVersion += 1;
  stopLessonAudio(true);
}

export function initLessonAudio(audioSource: AudioSource): void {
  source = audioSource;
  lessonAudioSpeed.value = String(readPlaybackRate(window.localStorage, profileId));
  lessonAudioButton.addEventListener("click", () => void toggleLessonAudio());
  lessonAudio.addEventListener("play", () => setAudioButton(true));
  lessonAudio.addEventListener("pause", () => setAudioButton(false));
  lessonAudio.addEventListener("ended", () => {
    lessonAudio.currentTime = 0;
    setAudioButton(false);
  });
  lessonAudio.addEventListener("error", () => {
    if (!lessonAudio.currentSrc) return;
    currentAudioStatus = null;
    const fallback = browserSpeechFallback(source.lesson()?.learning_language ?? "this language");
    lessonAudioButton.disabled = !fallback.available;
    lessonAudioStatus.textContent = `${localAudioElementFailure(
      lessonAudio.error?.code ?? null,
    )} ${browserFallbackMessage(fallback)}`;
  });
  lessonAudioSpeed.addEventListener("change", () => {
    const rate = persistPlaybackRate(window.localStorage, profileId, lessonAudioSpeed.value);
    lessonAudioSpeed.value = String(rate);
    lessonAudio.playbackRate = rate;
  });
  if (browserSpeechAvailable()) {
    window.speechSynthesis.addEventListener("voiceschanged", () => {
      const lesson = source.lesson();
      if (!lesson || !currentAudioStatus || currentAudioStatus.state === "ready") return;
      lessonAudioStatus.textContent = audioStatusMessage(
        currentAudioStatus,
        browserSpeechFallback(lesson.learning_language),
      );
    });
  }
}
