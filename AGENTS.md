# Contributor and agent notes

Lang Reader is a local, single-user language reader: it renders prepared lessons, records what the
learner needed help with, and replays that evidence into vocabulary, character and grammar state.
The [README](README.md) is the user guide; this file is for people and coding agents changing the
code. Where the two disagree, the code and tests win; please fix the document.

## Set up and check

```bash
./run.sh              # sync locked dependencies, build the UI, serve http://127.0.0.1:8000
./run_tests.sh        # ruff, mypy, pytest, biome, tsc, vitest, build, browser smoke test
uv run lang status    # machine-readable summary of the local state
```

Needs Python 3.10+, [uv](https://docs.astral.sh/uv/), Node.js 22+ and pnpm (pinned in
`package.json`; `corepack enable` provides it). The smoke test runs only when Chromium or Chrome is
installed. The Docker files and the CI workflow have not been run yet.

## Conventions

- Python: `uv` only, never pip or Poetry. Strict mypy, ruff, 100-column lines, pytest.
- Front end: strict TypeScript, ESM, `pnpm` (never npm or yarn), Biome, Vitest. `pnpm build`
  compiles `web/` and concatenates `web/styles/*.css` into `server/static/`.
- FastAPI, SQLAlchemy and SQLite; no framework on the front end. Keep dependencies few.
- Small, domain-named modules; a comment only for a reason, an invariant or a non-obvious failure.
  Fail fast; handle the errors that matter. Remove what a change made obsolete instead of keeping a
  compatibility layer.
- `.gitignore` is a whitelist: ignore everything, reopen directories, allow tracked paths
  explicitly. A new kind of file needs a line there.
- `server/settings.py` is the only place that reads `ARC_LANG_*` configuration; add new settings
  there and to the README table (`tests/test_docs.py` fails when they differ). Model providers are
  the exception: they live in `lang-reader.toml`, parsed by `server/providers.py`.
- Never commit state, credentials, lesson text, learning history or audio. Runtime data lives under
  `data/` (or `$ARC_LANG_DATA_DIR`) and is not tracked.

## Data and evidence rules

- Never edit or replace a profile database by hand. `interactions` is append-only truth;
  `lexeme_states`, `character_states`, `grammar_states` and `proficiency_state` are derived and are
  rebuilt by `uv run lang rebuild`. Queue skip and restore actions are separate append-only actions,
  never learning interactions.
- Do not reward a render or a scroll as knowledge, and do not change derived values directly. Rescore
  memory-model changes with `uv run lang memory evaluate --fit` against real history before adopting
  them.
- Profiles are dormant until explicitly activated (`profile activate` or the browser form). An
  inactive profile must not spend model calls, claim generation tasks, prepare audio or invoke
  browser speech. Do not forge the reserved `activation` or `onboarding` preference objects through
  `profile set`.
- Playback and previews never create learning interactions; `Peek` mode records nothing.
- Statistics and the Words, Characters and Grammar lists are read-only projections of the same
  events and the same "opened text" boundary.

## Lessons and generation

- Lesson text is `blocks -> sentences -> runs`. A run is plain text or a term-bearing fragment.
  Never use HTML or character offsets: the UI creates text nodes.
- A term key is stable for one `(language, lemma, part of speech)` meaning. Display and click spans
  are separate from canonical learning units, which alone enter Words, SRS and calibration.
- Language tags are BCP-47 variants (`es-ES`, `es-419`, `zh-Hans`, `zh-Hant`). `level` is the broad
  CEFR value; `difficulty` (0-1) is the fine one.
- Language-specific behaviour lives in a pack (`packs/<lang>/rules`, `grammar`, `wordlist`; see
  `packs/README.md`), not in shared reducers. Add a declarative rule or a catalog entry rather than a
  language branch or fuzzy stemming. Pack contents enter the profile fingerprint, so a change
  invalidates stale generation work.
- The model is reached only through the providers of `lang-reader.toml` (`server/providers.py`,
  `model_callback.py`, `model_http.py`, `agent_command.py`); keys stay in files outside the repo and
  never reach a log. The host validates every returned draft against the request
  and the stage schema; the model never writes to the database.
- Generation failures are retained as context for one bounded retry and then stay visible;
  maintenance must not loop on them. Availability waits (exit 3) do not consume the failure budget.
- Neural speech is an isolated runtime (`extras/tts-qwen/`). Never add Torch or Qwen packages to the
  main project dependencies.

## Tests

Tests use temporary data directories, stand-in model commands and local stand-in HTTP servers; none
needs network access or a real model. Add a test for the behaviour you change and for its important failure case. Do not run
the suite against a data directory that holds real learning history.
