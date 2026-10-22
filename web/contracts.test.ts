import { describe, expect, it } from "vitest";

import { type JsonRecord, parseReader, parseWords } from "./contracts.js";

function profile(): JsonRecord {
  return {
    learning_language: "zh-Hans",
    translation_language: "en-GB",
    level: "A2",
    difficulty: 0.32,
    interests: ["history", "food"],
    preferences: {},
  };
}

function progress(): JsonRecord {
  return {
    session_id: null,
    started: false,
    completed: false,
    rating: null,
    revealed_term_keys: [],
    revealed_sentence_keys: [],
    full_translation_revealed: false,
  };
}

function readerWithSentences(sentences: unknown[], title = "早晨的市场"): JsonRecord {
  return {
    profile: profile(),
    lesson_id: 7,
    lesson: {
      schema_version: 1,
      key: "market-morning",
      title,
      learning_language: "zh-Hans",
      translation_language: "en-GB",
      topic: "daily life",
      level: "A2",
      difficulty: 0.32,
      blocks: [{ key: "opening", sentences }],
      target_term_keys: ["zh:市场:NOUN"],
      metadata: { generator: "test" },
    },
    term_bands: { "zh:市场:NOUN": "expected" },
    progress: progress(),
  };
}

describe("parseReader", () => {
  it("accepts a nullable lesson state without inventing a lesson", () => {
    const parsed = parseReader({
      profile: profile(),
      lesson_id: null,
      lesson: null,
      term_bands: {},
      progress: progress(),
    });

    expect(parsed.lesson_id).toBeNull();
    expect(parsed.lesson).toBeNull();
    expect(parsed.progress.session_id).toBeUndefined();
    expect(parsed.generation_status).toBeNull();
    expect(parsed.profile.learning_language).toBe("zh-Hans");
    expect(parsed.comfortable_grammar_construction_keys).toEqual([]);
  });

  it("accepts provider-neutral generation progress", () => {
    const parsed = parseReader({
      profile: profile(),
      lesson_id: null,
      lesson: null,
      term_bands: {},
      progress: progress(),
      generation_status: "running",
    });

    expect(parsed.generation_status).toBe("running");
    expect(() =>
      parseReader({
        profile: profile(),
        lesson_id: null,
        lesson: null,
        term_bands: {},
        progress: progress(),
        generation_status: "unknown",
      }),
    ).toThrow(/generation_status/);
  });

  it("accepts only the five supported term bands", () => {
    const valid = readerWithSentences([
      { key: "bands", translation: "Bands.", runs: [{ text: "词。" }] },
    ]);
    valid.term_bands = {
      familiar: "familiar",
      expected: "expected",
      uncertain: "uncertain",
      focus: "focus",
      incidental: "incidental",
    };

    expect(parseReader(valid).term_bands).toEqual(valid.term_bands);
    expect(() => parseReader({ ...valid, term_bands: { word: "unknown" } })).toThrow(
      /unsupported band/,
    );
    expect(() => parseReader({ ...valid, term_bands: [] })).toThrow(/term_bands/);
    expect(() => parseReader({ ...valid, term_bands: undefined })).toThrow(/term_bands/);
  });

  it("preserves Unicode and repeated occurrences of the same term", () => {
    const term = {
      key: "zh:市场:NOUN",
      lemma: "市场",
      pos: "NOUN",
      gloss: "market / 市集",
      pronunciation: "shìchǎng",
      frequency_rank: 713,
    };
    const parsed = parseReader(
      readerWithSentences([
        {
          key: "sentence-你好",
          translation: "The market, the market — it is lively today!",
          runs: [
            { text: "市场", pronunciation: "shì chǎng", term },
            { text: "，" },
            { text: "市场", term: { ...term } },
            { text: "，今天很热闹！" },
          ],
        },
      ]),
    );

    const runs = parsed.lesson?.blocks[0]?.sentences[0]?.runs;
    expect(runs?.map((run) => run.text)).toEqual(["市场", "，", "市场", "，今天很热闹！"]);
    expect(runs?.filter((run) => run.term?.key === term.key)).toHaveLength(2);
    expect(runs?.[0]?.pronunciation).toBe("shì chǎng");
    expect(runs?.[1]?.pronunciation).toBeUndefined();
    expect(runs?.[0]?.term?.pronunciation).toBe("shìchǎng");
  });

  it("parses an optional annotated title as a sentence", () => {
    const reader = readerWithSentences([
      { key: "body", translation: "The market opens.", runs: [{ text: "市场开门。" }] },
    ]);
    const rawLesson = reader.lesson as JsonRecord;
    rawLesson.title_sentence = {
      key: "title",
      translation: "Morning market",
      runs: [
        {
          text: "早市",
          term: {
            key: "zh:早市:NOUN",
            lemma: "早市",
            pos: "NOUN",
            gloss: "morning market",
            pronunciation: "zǎoshì",
            frequency_rank: 8120,
          },
        },
      ],
    };

    const parsed = parseReader(reader);

    expect(parsed.lesson?.title_sentence?.key).toBe("title");
    expect(parsed.lesson?.title_sentence?.runs[0]?.term?.pronunciation).toBe("zǎoshì");
    expect(parsed.lesson?.title_sentence?.translation).toBe("Morning market");
  });

  it("parses anchored grammar occurrences, catalog definitions, and revealed help", () => {
    const reader = readerWithSentences([
      {
        key: "classifier-sentence",
        translation: "There is one market.",
        runs: [{ text: "有" }, { text: "一个" }, { text: "市场。" }],
        grammar: [
          {
            key: "classifier-sentence:grammar:1",
            construction_key: "zh:measure-word-phrase",
            run_start: 1,
            run_end: 3,
            note: "个 classifies the market here.",
          },
        ],
      },
    ]);
    reader.grammar_catalog = [
      {
        key: "zh:measure-word-phrase",
        label: "Number–classifier phrase",
        description: "A number or determiner combines with a classifier before a noun.",
        category: "noun phrases",
        difficulty: 0.2,
        generation_hint: null,
      },
    ];
    reader.comfortable_grammar_construction_keys = ["zh:measure-word-phrase"];
    reader.progress = {
      ...progress(),
      revealed_grammar_occurrence_keys: ["classifier-sentence:grammar:1"],
    };

    const parsed = parseReader(reader);
    const occurrence = parsed.lesson?.blocks[0]?.sentences[0]?.grammar[0];
    expect(occurrence).toMatchObject({
      construction_key: "zh:measure-word-phrase",
      run_start: 1,
      run_end: 3,
    });
    expect(parsed.grammar_catalog[0]?.label).toBe("Number–classifier phrase");
    expect(parsed.comfortable_grammar_construction_keys).toEqual(["zh:measure-word-phrase"]);
    expect(parsed.progress.revealed_grammar_occurrence_keys).toEqual([
      "classifier-sentence:grammar:1",
    ]);

    const rawLesson = reader.lesson as JsonRecord;
    const rawBlocks = rawLesson.blocks as JsonRecord[];
    const rawSentences = rawBlocks[0]?.sentences as JsonRecord[];
    const badSentence = { ...rawSentences[0], grammar: [{ ...occurrence, run_end: 4 }] };
    expect(() =>
      parseReader({
        ...reader,
        lesson: { ...rawLesson, blocks: [{ key: "bad", sentences: [badSentence] }] },
      }),
    ).toThrow(/non-empty range/);
  });

  it("rejects malformed or duplicate comfortable grammar keys", () => {
    const reader = readerWithSentences([
      { key: "body", translation: "The market opens.", runs: [{ text: "市场开门。" }] },
    ]);

    expect(() =>
      parseReader({ ...reader, comfortable_grammar_construction_keys: "zh:de-modifier" }),
    ).toThrow(/must be a list/);
    expect(() =>
      parseReader({ ...reader, comfortable_grammar_construction_keys: ["zh:de-modifier", ""] }),
    ).toThrow(/must not be blank/);
    expect(() =>
      parseReader({
        ...reader,
        comfortable_grammar_construction_keys: ["zh:de-modifier", "zh:de-modifier"],
      }),
    ).toThrow(/unique keys/);
  });

  it("keeps legacy titles and rejects malformed annotated titles", () => {
    const legacy = readerWithSentences([
      { key: "body", translation: "A market.", runs: [{ text: "市场。" }] },
    ]);
    expect(parseReader(legacy).lesson?.title_sentence).toBeUndefined();

    const rawLesson = legacy.lesson as JsonRecord;
    expect(() =>
      parseReader({
        ...legacy,
        lesson: { ...rawLesson, title_sentence: { key: "title", runs: [] } },
      }),
    ).toThrow(/sentence.runs/);
  });

  it("preserves literal script-looking strings as inert data", () => {
    const literal = '<script>alert("still text")</script>';
    const parsed = parseReader(
      readerWithSentences(
        [
          {
            key: "literal-data",
            translation: literal,
            runs: [
              {
                text: literal,
                term: {
                  key: "zh:市场:NOUN",
                  lemma: "市场",
                  pos: "NOUN",
                  gloss: literal,
                },
              },
            ],
          },
        ],
        literal,
      ),
    );

    const sentence = parsed.lesson?.blocks[0]?.sentences[0];
    expect(parsed.lesson?.title).toBe(literal);
    expect(sentence?.runs[0]?.text).toBe(literal);
    expect(sentence?.runs[0]?.term?.gloss).toBe(literal);
    expect(sentence?.translation).toBe(literal);
  });

  it("accepts null or omitted sentence translations", () => {
    const parsed = parseReader(
      readerWithSentences([
        { key: "null-translation", translation: null, runs: [{ text: "没有翻译。" }] },
        { key: "missing-translation", runs: [{ text: "也没有翻译。" }] },
      ]),
    );
    const sentences = parsed.lesson?.blocks[0]?.sentences;

    expect(sentences?.[0]?.translation).toBeUndefined();
    expect(sentences?.[1]?.translation).toBeUndefined();
  });

  it("rejects malformed or internally inconsistent responses", () => {
    const valid = readerWithSentences([
      { key: "valid", translation: "A market.", runs: [{ text: "市场。" }] },
    ]);
    const lesson = valid.lesson as JsonRecord;

    expect(() => parseReader(null)).toThrow(/response/);
    expect(() => parseReader({ ...valid, lesson_id: "7" })).toThrow(/lesson_id/);
    expect(() => parseReader({ ...valid, lesson: null })).toThrow(/both be present or null/);
    expect(() => parseReader({ ...valid, lesson: { ...lesson, difficulty: 1.2 } })).toThrow(
      /difficulty/,
    );
    expect(() => parseReader({ ...valid, lesson: { ...lesson, blocks: [] } })).toThrow(
      /non-empty list/,
    );
    expect(() =>
      parseReader({
        ...valid,
        lesson: {
          ...lesson,
          blocks: [
            {
              key: "bad",
              sentences: [{ key: "bad", runs: [{ text: 42 }], translation: null }],
            },
          ],
        },
      }),
    ).toThrow(/run.text/);
    expect(() =>
      parseReader({
        ...valid,
        lesson: {
          ...lesson,
          blocks: [
            {
              key: "bad-reading",
              sentences: [{ key: "bad-reading", runs: [{ text: "行", pronunciation: "  " }] }],
            },
          ],
        },
      }),
    ).toThrow(/run.pronunciation/);
    expect(() =>
      parseReader({
        ...valid,
        progress: { ...progress(), revealed_term_keys: ["known", 3] },
      }),
    ).toThrow(/revealed_term_keys/);
  });
});

describe("parseWords", () => {
  const response = {
    learning_language: "es-ES",
    translation_language: "en",
    words: [
      {
        key: "es:ir:VERB",
        lemma: "ir",
        pos: "verb",
        gloss: "to go",
        pronunciation: null,
        frequency_rank: 20,
        surface_forms: [
          { text: "fui", occurrences: 1 },
          { text: "voy", occurrences: 2 },
        ],
        related_senses: [
          {
            key: "es:ir:NOUN",
            pos: "noun",
            gloss: "going",
            pronunciation: null,
          },
        ],
        occurrence_count: 3,
        exposed_lesson_count: 2,
        raw_reveal_count: 3,
        counted_reveal_sessions: 4,
        qualified_exposures: 1,
        mastery: 0.44,
        stability_days: 0.5,
        reveal_failures: 1.75,
        first_exposed_at: "2026-07-10T12:00:00Z",
        last_exposed_at: "2026-07-12T12:00:00Z",
        last_revealed_at: "2026-07-12T12:05:00Z",
        next_due_at: null,
      },
    ],
  };

  it("preserves canonical terms, surface forms, and raw versus counted clicks", () => {
    const parsed = parseWords(response);
    const word = parsed.words[0];

    expect(word?.lemma).toBe("ir");
    expect(word?.surface_forms.map((surface) => surface.text)).toEqual(["fui", "voy"]);
    expect(word?.related_senses).toEqual([
      { key: "es:ir:NOUN", pos: "noun", gloss: "going", pronunciation: undefined },
    ]);
    expect(word?.raw_reveal_count).toBe(3);
    expect(word?.counted_reveal_sessions).toBe(4);
    expect(word?.reveal_failures).toBe(1.75);
    expect(word?.next_due_at).toBeUndefined();
  });

  it("rejects inconsistent counts and duplicate identities", () => {
    expect(() =>
      parseWords({
        ...response,
        words: [{ ...response.words[0], occurrence_count: 4 }],
      }),
    ).toThrow(/surface occurrences/);
    expect(() =>
      parseWords({ ...response, words: [...response.words, ...response.words] }),
    ).toThrow(/unique/);
  });
});
