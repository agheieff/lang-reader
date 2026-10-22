import { describe, expect, it } from "vitest";

import type { Lesson } from "./contracts.js";
import {
  AUDIO_AVAILABILITIES,
  AUDIO_STATUS_REASONS,
  audioStatusFetchFailure,
  audioStatusMessage,
  browserSpeechFailure,
  browserVoicePriorities,
  chooseBrowserVoice,
  lessonSpeechChunks,
  localAudioElementFailure,
  localAudioPlayFailure,
  parseLessonAudioStatus,
  parseTtsSettings,
  persistPlaybackRate,
  readPlaybackRate,
} from "./tts.js";

function voice(lang: string, localService: boolean): SpeechSynthesisVoice {
  return { default: false, lang, localService, name: lang, voiceURI: lang };
}

describe("TTS contracts", () => {
  it("validates every availability and reason from their runtime sources of truth", () => {
    for (const state of AUDIO_AVAILABILITIES) {
      expect(
        parseLessonAudioStatus({
          lesson_id: 4,
          state,
          reason: "ready",
          voice_id: null,
          voice_label: null,
          audio_url: state === "ready" ? "/audio/4" : null,
          message: null,
        }).state,
      ).toBe(state);
    }
    for (const reason of AUDIO_STATUS_REASONS) {
      expect(
        parseLessonAudioStatus({
          lesson_id: 4,
          state: "failed",
          reason,
          voice_id: null,
          voice_label: null,
          audio_url: null,
          message: null,
        }).reason,
      ).toBe(reason);
    }
  });

  it("parses profile voice settings and ready audio", () => {
    const settings = parseTtsSettings({
      provider: "qwen3-tts",
      model: "Qwen/model",
      learning_language: "zh-Hant",
      model_language: "Chinese",
      supported: true,
      provider_installed: true,
      profile_active: true,
      selected_voice_id: "Serena",
      voices: [
        {
          id: "Serena",
          label: "Serena",
          description: "Warm voice",
          native_language: "Chinese",
          recommended: true,
        },
      ],
      note: "Mandarin",
    });
    const audio = parseLessonAudioStatus({
      lesson_id: 4,
      state: "ready",
      reason: "ready",
      voice_id: "Serena",
      voice_label: "Serena",
      audio_url: "/audio/4",
      message: null,
    });

    expect(settings.selected_voice_id).toBe("Serena");
    expect(settings.voices[0]?.recommended).toBe(true);
    expect(audio.audio_url).toBe("/audio/4");
  });

  it("parses inactive audio without offering fallback copy", () => {
    const audio = parseLessonAudioStatus({
      lesson_id: 4,
      state: "disabled",
      reason: "profile_inactive",
      voice_id: "Serena",
      voice_label: "Serena",
      audio_url: null,
      message: "Inactive",
    });
    expect(
      audioStatusMessage(audio, {
        available: true,
        learningLanguage: "zh-Hans",
        matchingVoiceName: "System voice",
      }),
    ).toBe("Activate this language before using listening audio.");
  });

  it("requires a URL for ready audio", () => {
    expect(() =>
      parseLessonAudioStatus({
        lesson_id: 1,
        state: "ready",
        reason: "ready",
        voice_id: "Aiden",
        voice_label: "Aiden",
        audio_url: null,
        message: null,
      }),
    ).toThrow(/needs a URL/);
  });

  it("requires a safe reason code for status-specific feedback", () => {
    expect(() =>
      parseLessonAudioStatus({
        lesson_id: 1,
        state: "failed",
        reason: "RuntimeError: /private/path",
        voice_id: "Aiden",
        voice_label: "Aiden",
        audio_url: null,
        message: "failed",
      }),
    ).toThrow(/audio reason/);
  });
});

describe("audio feedback", () => {
  const fallback = {
    available: true,
    learningLanguage: "de-DE",
    matchingVoiceName: "Anna",
  };

  it("distinguishes queued, active, retrying, and terminal generation states", () => {
    const status = (reason: "queued" | "generating" | "retrying" | "generation_timeout") => ({
      lesson_id: 3,
      state: reason === "generation_timeout" ? ("failed" as const) : ("preparing" as const),
      reason,
      voice_id: "Aiden",
      voice_label: "Aiden",
      audio_url: null,
      message: null,
    });

    expect(audioStatusMessage(status("queued"), fallback)).toMatch(/queued.*one text at a time/i);
    expect(audioStatusMessage(status("generating"), fallback)).toMatch(
      /generated now|generating now/i,
    );
    expect(audioStatusMessage(status("retrying"), fallback)).toMatch(/automatic retry/i);
    expect(audioStatusMessage(status("generation_timeout"), fallback)).toMatch(/timed out.*Anna/i);
  });

  it("explains unavailable fallbacks without promising a matching voice", () => {
    const status = {
      lesson_id: 3,
      state: "unavailable" as const,
      reason: "provider_unavailable" as const,
      voice_id: "Aiden",
      voice_label: "Aiden",
      audio_url: null,
      message: null,
    };
    expect(
      audioStatusMessage(status, {
        available: true,
        learningLanguage: "de-DE",
        matchingVoiceName: null,
      }),
    ).toMatch(/not installed.*No matching de-DE system voice/i);
    expect(
      audioStatusMessage(status, {
        available: false,
        learningLanguage: "de-DE",
        matchingVoiceName: null,
      }),
    ).toMatch(/No browser speech engine/i);
  });

  it("maps status, media, playback, and browser speech failures to actionable copy", () => {
    expect(audioStatusFetchFailure(null)).toMatch(/still running/i);
    expect(audioStatusFetchFailure(404)).toMatch(/Refresh/i);
    expect(audioStatusFetchFailure(503)).toContain("503");
    expect(localAudioPlayFailure({ name: "NotAllowedError" })).toMatch(/Allow audio/i);
    expect(localAudioPlayFailure({ name: "NotSupportedError" })).toMatch(/decode/i);
    expect(localAudioElementFailure(2)).toMatch(/app is running/i);
    expect(localAudioElementFailure(3)).toMatch(/decode/i);
    expect(browserSpeechFailure("language-unavailable")).toMatch(/OS settings/i);
    expect(browserSpeechFailure("not-allowed")).toMatch(/Allow audio/i);
    expect(browserSpeechFailure("network")).toMatch(/network service/i);
  });
});

describe("playback rate preference", () => {
  it("persists a supported rate separately for each profile and falls back safely", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };

    expect(readPlaybackRate(storage, "de-de")).toBe(1);
    expect(persistPlaybackRate(storage, "de-de", "1.75")).toBe(1.75);
    expect(readPlaybackRate(storage, "de-de")).toBe(1.75);
    expect(readPlaybackRate(storage, "zh-hans")).toBe(1);
    expect(persistPlaybackRate(storage, "de-de", "12")).toBe(1);
  });
});

describe("browser voice fallback", () => {
  it("keeps script and regional priorities distinct", () => {
    expect(browserVoicePriorities("zh-Hant")).toEqual(["zh-tw", "zh-hk", "zh"]);
    expect(browserVoicePriorities("zh-Hans")).toEqual(["zh-cn", "zh"]);
    expect(browserVoicePriorities("es-419")).toEqual(["es-mx", "es-us", "es"]);
    expect(browserVoicePriorities("es-ES")).toEqual(["es-es", "es"]);
  });

  it("prefers an exact local voice before a network voice or broad family", () => {
    const selected = chooseBrowserVoice(
      [voice("zh-HK", true), voice("zh-TW", false), voice("zh-TW", true)],
      "zh-Hant",
    );
    expect(selected?.lang).toBe("zh-TW");
    expect(selected?.localService).toBe(true);
  });
});

it("speaks title and source sentences but never translations or topic", () => {
  const lesson = {
    title: "标题",
    topic: "metadata topic",
    blocks: [
      {
        key: "b",
        sentences: [
          {
            key: "s",
            translation: "translation",
            runs: [{ text: "正文" }, { text: "。" }],
          },
        ],
      },
    ],
  } as Lesson;

  expect(lessonSpeechChunks(lesson)).toEqual(["标题", "正文。"]);
});
