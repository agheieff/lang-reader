export type PinyinTone = 1 | 2 | 3 | 4 | 5;

export type PinyinPart =
  | { readonly kind: "separator"; readonly text: string }
  | { readonly kind: "syllable"; readonly text: string; readonly tone: PinyinTone };

export type HanziTonePart =
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "hanzi"; readonly text: string; readonly tone: PinyinTone };

interface LetterUnit {
  readonly base: string;
  readonly text: string;
  readonly tone: PinyinTone | null;
}

interface TokenOption {
  readonly syllables: readonly PinyinPart[];
}

const SYLLABLES = new Set(
  `
  a ai an ang ao
  ba bai ban bang bao bei ben beng bi bian biao bie bin bing bo bu
  ca cai can cang cao ce cen ceng cha chai chan chang chao che chen cheng chi
  chong chou chu chua chuai chuan chuang chui chun chuo ci cong cou cu cuan cui
  cun cuo
  da dai dan dang dao de dei den deng di dia dian diao die ding diu dong dou du
  duan dui dun duo
  e ei en eng er
  fa fan fang fei fen feng fo fou fu
  ga gai gan gang gao ge gei gen geng gong gou gu gua guai guan guang gui gun guo
  ha hai han hang hao he hei hen heng hong hou hu hua huai huan huang hui hun huo
  ji jia jian jiang jiao jie jin jing jiong jiu ju juan jue jun
  ka kai kan kang kao ke ken keng kong kou ku kua kuai kuan kuang kui kun kuo
  la lai lan lang lao le lei leng li lia lian liang liao lie lin ling liu lo long
  lou lu luan lun luo lü lüe
  ma mai man mang mao me mei men meng mi mian miao mie min ming miu mo mou mu
  na nai nan nang nao ne nei nen neng ni nian niang niao nie nin ning niu nong
  nou nu nuan nuo nü nüe
  o ou
  pa pai pan pang pao pei pen peng pi pian piao pie pin ping po pou pu
  qi qia qian qiang qiao qie qin qing qiong qiu qu quan que qun
  ran rang rao re ren reng ri rong rou ru rua ruan rui run ruo
  sa sai san sang sao se sen seng sha shai shan shang shao she shei shen sheng
  shi shou shu shua shuai shuan shuang shui shun shuo si song sou su suan sui sun
  suo
  ta tai tan tang tao te teng ti tian tiao tie ting tong tou tu tuan tui tun tuo
  wa wai wan wang wei wen weng wo wu
  xi xia xian xiang xiao xie xin xing xiong xiu xu xuan xue xun
  ya yan yang yao ye yi yin ying yo yong you yu yuan yue yun
  za zai zan zang zao ze zei zen zeng zha zhai zhan zhang zhao zhe zhei zhen
  zheng zhi zhong zhou zhu zhua zhuai zhuan zhuang zhui zhun zhuo zi zong zou zu
  zuan zui zun zuo
  hm hng m n ng
  `
    .trim()
    .split(/\s+/),
);

const TONE_MARKS = new Map<string, PinyinTone>([
  ["\u0304", 1],
  ["\u0301", 2],
  ["\u030c", 3],
  ["\u0300", 4],
]);

const MARKED_VOWELS: Readonly<Record<PinyinTone, Readonly<Record<string, string>>>> = {
  1: { a: "ā", e: "ē", i: "ī", o: "ō", u: "ū", ü: "ǖ" },
  2: { a: "á", e: "é", i: "í", o: "ó", u: "ú", ü: "ǘ" },
  3: { a: "ǎ", e: "ě", i: "ǐ", o: "ǒ", u: "ǔ", ü: "ǚ" },
  4: { a: "à", e: "è", i: "ì", o: "ò", u: "ù", ü: "ǜ" },
  5: {},
};

function normalizeUmlaut(value: string): string {
  return value
    .replaceAll(/u:/gi, (match) => (match[0] === "U" ? "Ü" : "ü"))
    .replaceAll(/v/gi, (match) => (match === "V" ? "Ü" : "ü"));
}

function toneMarked(text: string, tone: PinyinTone): string {
  const normalized = normalizeUmlaut(text);
  if (tone === 5) return normalized;
  const lower = normalized.toLocaleLowerCase();
  let index = lower.indexOf("a");
  if (index < 0) index = lower.indexOf("e");
  if (index < 0) {
    const ou = lower.indexOf("ou");
    if (ou >= 0) index = ou;
  }
  if (index < 0) {
    for (let cursor = lower.length - 1; cursor >= 0; cursor -= 1) {
      if ("iouü".includes(lower[cursor] ?? "")) {
        index = cursor;
        break;
      }
    }
  }
  if (index < 0) return normalized;
  const vowel = lower[index];
  if (!vowel) return normalized;
  const marked = MARKED_VOWELS[tone][vowel];
  if (!marked) return normalized;
  const source = normalized[index];
  const replacement = source === source?.toLocaleUpperCase() ? marked.toLocaleUpperCase() : marked;
  return `${normalized.slice(0, index)}${replacement}${normalized.slice(index + 1)}`;
}

function letterUnits(value: string): readonly LetterUnit[] {
  const normalized = normalizeUmlaut(value).normalize("NFD");
  const units: { base: string; text: string; tone: PinyinTone | null }[] = [];
  for (const character of normalized) {
    if (/\p{M}/u.test(character)) {
      const current = units.at(-1);
      if (!current) continue;
      current.text += character;
      const tone = TONE_MARKS.get(character);
      if (tone) current.tone = tone;
      if (character === "\u0308" && current.base === "u") current.base = "ü";
      continue;
    }
    units.push({ base: character.toLocaleLowerCase(), text: character, tone: null });
  }
  return units.map((unit) => ({ ...unit, text: unit.text.normalize("NFC") }));
}

function numberedOption(token: string): TokenOption | null {
  const normalized = normalizeUmlaut(token);
  const pattern = /([\p{L}\p{M}]+)([0-5])/gu;
  const matches = Array.from(normalized.matchAll(pattern));
  if (matches.length === 0 || matches.map((match) => match[0]).join("") !== normalized) return null;
  return {
    syllables: matches.map((match) => {
      const raw = match[1] ?? "";
      const digit = Number(match[2]);
      const tone = (digit === 0 ? 5 : digit) as PinyinTone;
      return { kind: "syllable", text: toneMarked(raw, tone), tone };
    }),
  };
}

function accentedOptions(token: string): readonly TokenOption[] {
  const units = letterUnits(token);
  const memo = new Map<number, readonly (readonly PinyinPart[])[]>();
  const visit = (start: number): readonly (readonly PinyinPart[])[] => {
    if (start === units.length) return [[]];
    const cached = memo.get(start);
    if (cached) return cached;
    const options: (readonly PinyinPart[])[] = [];
    for (let end = start + 1; end <= Math.min(units.length, start + 6); end += 1) {
      const slice = units.slice(start, end);
      if (!SYLLABLES.has(slice.map((unit) => unit.base).join(""))) continue;
      const tones = new Set(slice.flatMap((unit) => (unit.tone ? [unit.tone] : [])));
      if (tones.size > 1) continue;
      const tone = tones.values().next().value ?? 5;
      const syllable: PinyinPart = {
        kind: "syllable",
        text: slice
          .map((unit) => unit.text)
          .join("")
          .normalize("NFC"),
        tone,
      };
      for (const remainder of visit(end)) {
        options.push([syllable, ...remainder]);
      }
    }
    const limited = options.sort((left, right) => left.length - right.length).slice(0, 24);
    memo.set(start, limited);
    return limited;
  };
  return visit(0).map((syllables) => ({ syllables }));
}

function erhuaStem(token: string): string | null {
  return token.length > 1 && /r$/iu.test(token) ? token.slice(0, -1) : null;
}

function attachErhua(options: readonly TokenOption[], suffix: string): readonly TokenOption[] {
  return options.flatMap((option) => {
    const syllables = [...option.syllables];
    const last = syllables.at(-1);
    if (last?.kind !== "syllable") return [];
    syllables[syllables.length - 1] = { ...last, text: `${last.text}${suffix}` };
    return [{ syllables }];
  });
}

function erhuaOptions(token: string): readonly TokenOption[] {
  const stem = erhuaStem(token);
  if (!stem) return [];
  const numbered = numberedOption(stem);
  const options = numbered ? [numbered] : accentedOptions(stem);
  return attachErhua(options, token.slice(-1));
}

function tokenOptions(token: string): readonly TokenOption[] {
  const numbered = numberedOption(token);
  if (numbered) return [numbered];
  const accented = accentedOptions(token);
  if (accented.length > 0) return accented;
  const erhua = erhuaOptions(token);
  if (erhua.length > 0) return erhua;
  const tones = new Set(letterUnits(token).flatMap((unit) => (unit.tone ? [unit.tone] : [])));
  return [
    {
      syllables: [
        {
          kind: "syllable",
          text: normalizeUmlaut(token),
          tone: tones.size === 1 ? (tones.values().next().value ?? 5) : 5,
        },
      ],
    },
  ];
}

function hanCount(lemma: string): number {
  return Array.from(lemma).filter((character) => /\p{Script=Han}/u.test(character)).length;
}

function chooseTokens(tokens: readonly (readonly TokenOption[])[], expected: number): PinyinPart[] {
  let combinations: PinyinPart[][] = [[]];
  for (const options of tokens) {
    combinations = combinations
      .flatMap((prefix) => options.map((option) => [...prefix, ...option.syllables]))
      .sort((left, right) => left.length - right.length)
      .slice(0, 128);
  }
  return (
    combinations.find((combination) => combination.length === expected) ?? combinations[0] ?? []
  );
}

function isRecognizedOption(option: TokenOption): boolean {
  return option.syllables.every(
    (part) =>
      part.kind === "syllable" &&
      SYLLABLES.has(
        letterUnits(part.text)
          .map((unit) => unit.base)
          .join(""),
      ),
  );
}

function strictTokenOptions(token: string): readonly TokenOption[] {
  const numbered = numberedOption(token);
  if (numbered) return isRecognizedOption(numbered) ? [numbered] : [];
  const accented = accentedOptions(token).filter(isRecognizedOption);
  if (accented.length > 0) return accented;
  const stem = erhuaStem(token);
  if (!stem) return [];
  const numberedStem = numberedOption(stem);
  const stemOptions = numberedStem ? [numberedStem] : accentedOptions(stem);
  return attachErhua(stemOptions.filter(isRecognizedOption), token.slice(-1));
}

function alignedTones(pronunciation: string, expected: number): readonly PinyinTone[] | null {
  if (expected === 0 || pronunciation.trim() === "") return null;
  const chunks = pronunciation.match(/[\p{L}\p{M}\d:]+/gu) ?? [];
  const optionsByChunk = chunks.map(strictTokenOptions);
  if (optionsByChunk.length === 0 || optionsByChunk.some((options) => options.length === 0)) {
    return null;
  }

  let sequences: readonly (readonly PinyinTone[])[] = [[]];
  for (const options of optionsByChunk) {
    const unique = new Map<string, readonly PinyinTone[]>();
    for (const prefix of sequences) {
      for (const option of options) {
        const tones = option.syllables.flatMap((part) =>
          part.kind === "syllable" ? [part.tone] : [],
        );
        const sequence = [...prefix, ...tones];
        if (sequence.length <= expected) unique.set(sequence.join(","), sequence);
      }
    }
    // A highly ambiguous spelling should not produce potentially misleading character colors.
    if (unique.size > 256) return null;
    sequences = [...unique.values()];
  }

  const exact = sequences.filter((sequence) => sequence.length === expected);
  return exact.length === 1 ? (exact[0] ?? null) : null;
}

export function pinyinParts(pronunciation: string, lemma = ""): readonly PinyinPart[] {
  const chunks = pronunciation.match(/[\p{L}\p{M}\d:]+|[^\p{L}\p{M}\d:]+/gu) ?? [];
  const wordChunks = chunks.filter((chunk) => /[\p{L}\p{M}\d]/u.test(chunk));
  if (wordChunks.length === 0) return [{ kind: "separator", text: pronunciation }];
  const expected = hanCount(lemma);
  const syllables = chooseTokens(
    wordChunks.map((chunk) => tokenOptions(chunk)),
    expected,
  );
  if (expected > 0 && syllables.length === expected) {
    return syllables.flatMap((syllable, index) =>
      index === 0 ? [syllable] : [{ kind: "separator", text: " " }, syllable],
    );
  }

  const result: PinyinPart[] = [];
  let wordIndex = 0;
  for (const chunk of chunks) {
    if (!/[\p{L}\p{M}\d]/u.test(chunk)) {
      result.push({ kind: "separator", text: chunk });
      continue;
    }
    const option = tokenOptions(wordChunks[wordIndex] ?? chunk)[0];
    wordIndex += 1;
    if (!option) continue;
    for (const syllable of option.syllables) {
      if (result.at(-1)?.kind === "syllable") result.push({ kind: "separator", text: " " });
      result.push(syllable);
    }
  }
  return result;
}

export function hanziToneParts(surface: string, pronunciation: string): readonly HanziTonePart[] {
  const characters = Array.from(surface);
  const hanCharacters = characters.filter((character) => /\p{Script=Han}/u.test(character));
  let tones = alignedTones(pronunciation, hanCharacters.length);
  if (
    !tones &&
    hanCharacters.at(-1) === "儿" &&
    hanCharacters.length > 1 &&
    /r\s*$/iu.test(pronunciation)
  ) {
    const stemTones = alignedTones(pronunciation, hanCharacters.length - 1);
    const rhotacizedTone = stemTones?.at(-1);
    if (stemTones && rhotacizedTone) tones = [...stemTones, rhotacizedTone];
  }
  if (!tones) return [{ kind: "text", text: surface }];

  const result: HanziTonePart[] = [];
  let toneIndex = 0;
  for (const character of characters) {
    if (/\p{Script=Han}/u.test(character)) {
      const tone = tones[toneIndex];
      if (!tone) return [{ kind: "text", text: surface }];
      result.push({ kind: "hanzi", text: character, tone });
      toneIndex += 1;
      continue;
    }
    const previous = result.at(-1);
    if (previous?.kind === "text") {
      result[result.length - 1] = { kind: "text", text: previous.text + character };
    } else {
      result.push({ kind: "text", text: character });
    }
  }
  return result;
}

export function renderPinyin(container: HTMLElement, pronunciation: string, lemma = ""): void {
  const document = container.ownerDocument;
  const nodes = pinyinParts(pronunciation, lemma).map((part) => {
    if (part.kind === "separator") return document.createTextNode(part.text);
    const syllable = document.createElement("span");
    syllable.className = "pinyin-syllable";
    syllable.dataset.tone = String(part.tone);
    syllable.textContent = part.text;
    return syllable;
  });
  container.replaceChildren(...nodes);
}

export function renderHanziTones(
  container: HTMLElement,
  surface: string,
  pronunciation: string,
): void {
  const document = container.ownerDocument;
  const nodes = hanziToneParts(surface, pronunciation).map((part) => {
    if (part.kind === "text") return document.createTextNode(part.text);
    const character = document.createElement("span");
    character.className = "hanzi-tone";
    character.dataset.tone = String(part.tone);
    character.textContent = part.text;
    return character;
  });
  container.replaceChildren(...nodes);
}
