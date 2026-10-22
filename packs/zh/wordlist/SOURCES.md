# Chinese word list: sources and licence

`wordlist.tsv` is the only data pack in this repository that is not the project's
own work. It is a rearrangement of other people's vocabulary data, so it carries
their licence, not the Apache-2.0 licence of the code.

## What it is

Columns: `lemma`, `pronunciation` (lower-case pinyin with tone marks), `gloss`,
`frequency_rank`, `level` (HSK 3.0 level, where 7 covers 7-9; HSK 2.0 level when
no 3.0 level is published). Rows are ordered by frequency rank.

## Sources

| Source | What is used | Terms |
| --- | --- | --- |
| [drkameleon/complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary) | HSK levels, word selection, structure | MIT (`LICENSE-complete-hsk-vocabulary.txt`) |
| **CC-CEDICT** | Definitions (glosses) and pinyin | CC BY-SA 4.0 (`LICENSE-CC-BY-SA-4.0.txt`) |
| [krmanik/HSK-3.0-words-list](https://github.com/krmanik/HSK-3.0-words-list) | HSK 3.0 level assignments | Not verified at source; see below |
| SUBTLEX-CH, HanLP | Part-of-speech information used upstream | Not verified at source; see below |
| The official HSK 3.0 word lists | The lists the levels ultimately describe | A third-party publication; rights unconfirmed |

CC-CEDICT is the reason this table is share-alike. Please attribute CC-CEDICT
(<https://cc-cedict.org/>, by MDBG) when you distribute this table or a
derivative of it, and keep the same licence for derivatives.

This table was built by an agent from those sources; it has not been reviewed by
a lexicographer, and the upstream lists it derives from are themselves
compilations. If you need redistributable Chinese vocabulary with clear terms,
build your own list from CC-CEDICT directly with `scripts/build_zh_word_list.py`
and publish it as your own CC BY-SA 4.0 pack.

## Removing the pack

The list is optional. Delete `packs/zh/wordlist/wordlist.tsv` and Chinese
lessons still generate; new-word offers then rely on the model's own choices
instead of a frequency list.
