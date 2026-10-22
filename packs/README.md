# Language packs

The README's [Adding a language](../README.md#adding-a-language) walks through making a pack, with
a worked example (`examples/packs/fr/`) and `uv run lang pack check` to verify it. This file covers
what ships here and the licence of the word lists.

A pack is one directory per language under `packs/`, holding what the reader needs to teach that
language:

```
packs/<lang>/
  rules/*.toml      generation guidance, part-of-speech aliases, lemma conventions, decomposition
  grammar/*.toml    grammar catalog: construction keys, explanations, difficulty, generation hints
  wordlist/         optional frequency-ordered vocabulary (wordlist.tsv + pack.toml)
```

`server/packs.py` discovers packs at start, so adding a language is adding a directory: no code
change, no list to extend. `ARC_LANG_PACK_DIRS` (comma-separated paths) adds pack roots of your own,
which is how you keep a private pack out of this repository.

A BCP-47 variant uses its base language's pack (`de-AT` → `de`, `zh-Hant` → `zh`) unless it has its
own. A pack's contents enter generation fingerprints, so editing a rule invalidates stale
generation work instead of producing lessons from half-old guidance.

## What ships here

| Pack | Rules | Grammar catalog | Word list | Licence |
| --- | --- | --- | --- | --- |
| `es` | Spanish | Spanish | none | Apache-2.0 |
| `de` | German | German | none | Apache-2.0 |
| `zh` | Mandarin | Mandarin | HSK vocabulary with pinyin and glosses | rules/grammar Apache-2.0; word list CC BY-SA 4.0 |

The rules and grammar catalogs are original work written for this project: generic linguistic facts,
not copied from a textbook. Review them before relying on them for a specific curriculum.

## The Chinese word list is a separate licence

`packs/zh/wordlist/` is a rearrangement of other people's vocabulary data. Its glosses and pinyin
come from CC-CEDICT (CC BY-SA 4.0), so the table is distributed under CC BY-SA 4.0, not Apache-2.0.
See `packs/zh/wordlist/SOURCES.md` and `packs/zh/wordlist/LICENSE-CC-BY-SA-4.0.txt` for the full
attribution and terms.

If you need a word list under your own terms, delete `packs/zh/wordlist/wordlist.tsv` (Chinese
lessons still generate; new-word offers then rely on the model's own choices) or build your own list
with `scripts/build_zh_word_list.py` and publish it as your own pack. Word lists for other languages
are not bundled: drop a `wordlist.tsv` into a pack and it is used.

## Writing a word list

`wordlist/wordlist.tsv` is tab-separated with a header row:

```
lemma	pronunciation	gloss	frequency_rank	level
的	de	of; ~'s (possessive particle)	1	1
```

`pronunciation`, `frequency_rank` and `level` may be empty. `wordlist/pack.toml` names the base
languages the list serves and the source string stored on each entry:

```toml
languages = ["zh"]
source = "hsk"
```
