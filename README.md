# Lang Reader

A self-hosted reading app for one language learner. A language model writes short reading lessons at
your level; you read them, click what you do not know, and the app replays that evidence into
vocabulary, character and grammar practice. It runs on your own machine, keeps everything in local
SQLite files, and uses **a model service that you choose and pay for** (OpenAI, Anthropic,
OpenRouter, DeepSeek, any OpenAI-compatible server, a local model, or your own command). There is no
account system, telemetry or cloud service of its own.

One instance serves one learner with any number of language profiles. The code is Apache-2.0
([LICENSE](LICENSE)); the bundled Chinese word list is a separate CC BY-SA 4.0 data pack
([packs/README.md](packs/README.md)). The package is `lang-reader`, the command line is `lang`.

## Quick start for anyone

You need: Python 3.10 or newer with [uv](https://docs.astral.sh/uv/), Node.js 22 or newer with
[pnpm](https://pnpm.io/) (`corepack enable` is enough), Linux (see [Run locally](#run-locally) for
other systems), and an API key for a model service or a local model server.

```bash
git clone https://github.com/agheieff/lang-reader.git && cd lang-reader

# 1. Tell the app which model to use (edit the copy; see "Connect a model" for other services).
cp lang-reader.example.toml lang-reader.toml
printf '%s\n' 'PASTE-YOUR-API-KEY' > openai.key && chmod 600 openai.key

# 2. Install, build and start the reader plus its generation worker.
./run.sh
```

Then:

1. Open <http://127.0.0.1:8000>. The first start creates one profile, Spanish for an English
   reader (`es-ES`/`en`).
2. In a second terminal, check the model connection: `uv run lang provider test main`. It sends one
   tiny request and prints the answer and the tokens it used.
3. Answer the short start screen in the browser (rough level, interests, text length). Starting the
   profile is what lets the worker spend tokens: it then writes the first texts, which appear in
   the library when ready ([Automatic generation](#automatic-generation); a careful model spends a
   few hundred thousand prompt tokens per lesson).
4. For another language, create a profile first, for example
   `uv run lang profile create fr-fr --label French --learning-language fr-FR --translation-language en --select`,
   then reload the page. Any BCP-47 language tag works; [Adding a language](#adding-a-language)
   explains how to teach the model language-specific conventions.

Without a key you can still look around: `ARC_LANG_AUTO_GENERATE=0 ./run.sh` starts only the
reader, and `uv run lang lesson import examples/es-demo.json` loads a demo lesson.

## Run locally

You need Python 3.10 or newer with [uv](https://docs.astral.sh/uv/), Node.js 22 or newer with
[pnpm](https://pnpm.io/) (front-end build only; `package.json` pins pnpm 11.3.0, so
`corepack enable` is enough), and `ffmpeg` if you ask for compressed audio. The checks have been run
on Linux with Python 3.11 only; `run.sh` uses `setsid` (util-linux) to supervise the workers, so on
other systems start with `ARC_LANG_AUTO_GENERATE=0` or run the pieces yourself.

```bash
git clone https://github.com/agheieff/lang-reader.git && cd lang-reader
./run.sh
```

`run.sh` syncs the locked Python and Node dependencies, builds the TypeScript UI, creates the SQLite
databases under `data/`, and starts FastAPI on <http://127.0.0.1:8000> with one generation worker
(`./run.sh --port 8001` picks another port). After installing dependencies the app itself needs no
network access; only the model you configure does. Browser speech voices are an operating-system
service and are not guaranteed to work offline. `uv run lang --help` lists the command line;
`./run_tests.sh` runs the checks, including a headless-Chromium smoke test
(`scripts/smoke_browser.py`, skipped when no Chromium or Chrome is installed) that opens every view
of a temporary instance.

### Docker

`docker compose up --build` starts the app (published on `127.0.0.1:8000` only) and the generation
worker with one data volume. Create `./config/` first and put `lang-reader.toml` and your API key
files in it (the same files as in the quick start; a relative `api_key_file` is read from the
directory of the configuration file, `/config` in the containers). Optional settings go in `.env`
(copy `.env.example`). **Untested:** the Dockerfile, `compose.yaml` and the CI workflow
`.github/workflows/ci.yml` (which runs `./run_tests.sh`) have never been executed, because the
machine this release was prepared on has no Docker and the workflow has not run on GitHub.

## Inspect a read-only recovery snapshot

Start a separate reader against a frozen, consistent copy of the state directory:

```bash
ARC_LANG_SNAPSHOT_DIR=/path/to/consistent-state PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync --offline uvicorn server.main:app --host 127.0.0.1 --port 8001
```

The directory must contain `registry.json` and its declared SQLite copies, at the current schema
version, without WAL/SHM/journal sidecars. Use a backup made with SQLite's backup API, never a raw
copy of a running database or the live state directory. The source can be mounted read-only. This
mode does not bootstrap, chmod, lock, migrate or write the source. It opens SQLite immutably, copies
each database into RAM, and runs the existing derived reducers there. Requests serialize access to
the in-memory connections; RAM use scales with the snapshot databases. Restart to inspect a new copy.

The banner identifies the snapshot, lessons use Peek mode, pending browser events are not flushed
or cleared, and the server rejects mutation requests with HTTP 403. Existing cached audio can play;
missing audio is not generated. No worker starts. This setting applies only to the web reader;
it is not a way to run the mutation CLI or workers against a backup. Serve it on a private loopback
port, with the same access restrictions as the ordinary reader.

## Connect a model

Every lesson is written by a language model that you reach through a **provider**. Providers are
declared once, in `lang-reader.toml` in the project directory (or the file named by
`ARC_LANG_CONFIG`), and assigned to the generation stages. Start from the commented example:

```bash
cp lang-reader.example.toml lang-reader.toml
```

`lang-reader.toml` is ignored by Git. API keys are never written in it: a provider names a file that
holds the key (`api_key_file`, relative paths are read from the directory of the configuration file)
or an environment variable (`api_key_env`). An inline `api_key` is rejected. Keep key files outside
version control and readable only by you (`chmod 600`). Keys are read when the worker starts, so
restart it after changing one; they are never written to the generation logs or shown by
`lang provider list`.

| `kind` | Speaks | Use it for |
| --- | --- | --- |
| `openai` | `POST {base_url}/chat/completions` with `Authorization: Bearer` | OpenAI, OpenRouter, DeepSeek, Groq, Together, Mistral, vLLM, llama.cpp, LM Studio, Ollama (`/v1`) |
| `anthropic` | `POST {base_url}/v1/messages` with `x-api-key` and `anthropic-version` | Anthropic's Messages API |
| `command` | an external program ([below](#the-command-kind)) | anything else: a CLI, a proxy, custom routing |

Options of `openai` and `anthropic`: `base_url`, `model`, `api_key_file` | `api_key_env`,
`max_output_tokens`, `timeout_seconds` (how long to wait for an answer; default 600) and
`extra_body` (a table merged into the request JSON, for example `{ temperature = 0.3 }`). `openai`
adds `schema_mode` and `max_tokens_param`; `anthropic` adds `anthropic_version` (default
`2023-06-01`) and defaults `max_output_tokens` to 16000, because the Messages API requires a limit.

- `schema_mode = "schema"` (default) sends the response schema as `response_format` (structured
  output). `"json_object"` asks for JSON mode and appends the schema to the prompt; `"prompt"` only
  appends it, for servers without structured output. The app validates every answer itself, so the
  weaker modes cost retries, not correctness.
- `max_tokens_param = "max_completion_tokens"` is for OpenAI models that reject `max_tokens`.
- Anthropic receives the schema in the prompt.

Model names change often; the ones below are examples, so use any chat model your account can call.
**No real completion has been run against a hosted service.** The request and response handling of
both HTTP kinds is tested against local stand-in servers that imitate the documented formats. With
a deliberately invalid key, OpenAI, Anthropic, DeepSeek and OpenRouter all answered HTTP 401 at the
configured URLs, which shows that the endpoints and authentication headers are reachable but not
that a full lesson works. The Anthropic and DeepSeek model names were read from their documentation
in October 2026; the OpenAI and OpenRouter ones are unchecked examples. Run
`uv run lang provider test NAME` after editing the file: it sends one tiny request and prints the
answer and token counts.

### OpenAI

```toml
[providers.main]
kind = "openai"
base_url = "https://api.openai.com/v1"
model = "gpt-5.5"
api_key_file = "openai.key"
max_tokens_param = "max_completion_tokens"
```

### Anthropic

```toml
[providers.main]
kind = "anthropic"
model = "claude-sonnet-5-5"
api_key_file = "anthropic.key"
```

### OpenRouter

One key for many vendors' models; pick a model id from <https://openrouter.ai/models>. Not every
model there supports structured output, so the schema goes in the prompt.

```toml
[providers.main]
kind = "openai"
base_url = "https://openrouter.ai/api/v1"
model = "vendor/model-name"
api_key_file = "openrouter.key"
schema_mode = "prompt"
```

### DeepSeek

DeepSeek's JSON mode needs the word "json" and an example in the prompt, which `json_object` mode
provides.

```toml
[providers.main]
kind = "openai"
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_key_file = "deepseek.key"
schema_mode = "json_object"
```

### A local server (Ollama, llama.cpp, LM Studio, vLLM)

No key is needed for a server on your own machine. Lessons are long, strictly structured answers:
a small local model will often fail validation and exhaust the retry budget, so try a large one.

```toml
[providers.main]
kind = "openai"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:14b"
schema_mode = "prompt"
timeout_seconds = 1800
```

The base URL above is Ollama's; llama.cpp's server usually answers on `http://127.0.0.1:8080/v1`,
LM Studio on `http://127.0.0.1:1234/v1`. Use the model name your server reports.

### Different models for different stages

Generation runs in stages: `prose` (the story), `translation`, `lexical` (annotating every word,
the bulk of the tokens), `grammar`, `preferences` (updating the reading notes) and `legacy`
(the single-call calibration lessons). `[stages]` assigns a provider to each; `default` covers every
stage not listed, and a list names fallbacks. A strong model can write the prose while a cheap one
annotates:

```toml
[providers.strong]
kind = "anthropic"
model = "claude-sonnet-5-5"
api_key_file = "anthropic.key"

[providers.cheap]
kind = "openai"
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_key_file = "deepseek.key"
schema_mode = "json_object"

[stages]
default = "strong"
lexical = ["cheap", "strong"]
translation = "cheap"
```

Here `lexical` tries `cheap` first and falls back to `strong` when the call is unavailable
(unreachable, HTTP 429 or 503) or fails. A fallback is used only if you list it; an answer that
arrives but does not pass the app's validation is a failed draft handled by the normal retry, not a
reason to switch provider. Only providers used by a stage need a readable key.

### The command kind

A `command` provider runs a program of your own once per request, started as an argv list without a
shell:

```toml
[providers.main]
kind = "command"
command = ["uv", "run", "python", "examples/command_adapter.py"]
pass_env = ["ADAPTER_URL", "ADAPTER_MODEL"]
timeout_seconds = 900
```

The child sees only a minimal environment (`PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`,
`TMPDIR`), the variables named in `pass_env`, and these:

| Input | Meaning |
| --- | --- |
| stdin | the canonical request JSON (`stage`, `schema_version`, frozen dependencies, task data) |
| `ARC_LANG_CALLBACK_PROMPT_PATH` | the same task as a ready chat prompt; send this to a chat model |
| `ARC_LANG_CALLBACK_RESPONSE_SCHEMA_PATH` | the JSON Schema the answer must satisfy |
| `ARC_LANG_CALLBACK_RESPONSE_PATH` | where to write the answer (stdout works too) |
| `ARC_LANG_CALLBACK_USAGE_PATH` | where to write token counts as JSON (optional but wanted) |
| `ARC_LANG_CALLBACK_STAGE` | `prose`, `lexical`, `translation`, `grammar`, `preferences`, `legacy` |
| `ARC_LANG_JOB_ID`, `ARC_LANG_REQUEST_SHA256`, `ARC_LANG_PROFILE`, `ARC_LANG_WORKSPACE` | scope of the request |

Exit status `0` with a schema-valid answer is success; exit `3` (`ARC_LANG_AGENT_UNAVAILABLE_EXIT`)
means "no capacity right now, nothing happened"; anything else is a failed attempt. Usage JSON uses
the keys `input_tokens`, `cached_input_tokens`, `cache_creation_input_tokens`, `output_tokens` and
`reasoning_output_tokens`. [`examples/command_adapter.py`](examples/command_adapter.py) is a
complete, tested adapter in the standard library. `lang provider test` cannot probe a command.

### What happens when a call goes wrong

For `openai` and `anthropic` providers, an unreachable server and HTTP 429 or 503 count as
*unavailable* (exit status 3 in the command protocol): nothing is charged against the task, which
stays pending and is retried after 60 seconds. A timeout, any other HTTP error, an empty or
truncated answer (`finish_reason: length`, `stop_reason: max_tokens`) is a failed attempt, kept with
the error text in the task's `callback.log` for the bounded retry. A refused key (HTTP 401 or 403)
is reported without the server's text, since some servers echo part of the key. The reported usage
maps to `input_tokens` (all input, cached or not), `cached_input_tokens`, `cache_creation_input_tokens`
(Anthropic), `output_tokens` and `reasoning_output_tokens`; `lang generation list` prints it per task.

Budget before you start: one ordinary lesson costs one prose call, one translation call, several
sentence-batch lexical calls and one grammar call; a careful model spends a few hundred thousand
prompt tokens per lesson. Nothing is generated while a profile is inactive, and
`ARC_LANG_AUTO_GENERATE=0` turns the worker off entirely; you can then import lessons by hand with
`lang lesson import`.

## Adding a language

Any BCP-47 tag works out of the box: create a profile and the app uses safe generic guidance
(`lang pack check fr-FR` reports `generic`). A **language pack** adds what a model gets wrong
without help: lemma conventions, part-of-speech names, a catalog of grammar constructions to
annotate, and an optional word list for proposing new words. Packs are plain files found by
directory, so adding a language needs no code change.

### Layout

```
packs/<name>/
  rules/<anything>.toml     generation guidance, part-of-speech aliases, lemma conventions
  grammar/<anything>.toml   grammar catalog: construction keys, explanations, difficulty, hints
  wordlist/                 optional frequency-ordered vocabulary
    pack.toml               which base languages it serves and its source label
    wordlist.tsv            lemma, pronunciation, gloss, frequency rank, level
    LICENSE*, SOURCES.md    licence text and origin of the data (required for anything you publish)
```

`server/packs.py` globs `<root>/*/rules/*.toml`, `<root>/*/grammar/*.toml` and
`<root>/*/wordlist/` for every root: the repository's `packs/` and each directory in
`ARC_LANG_PACK_DIRS` (comma-separated). Rules and grammar files are re-read when they change (checked
every couple of seconds); a pack's contents enter the generation fingerprints, so editing one
invalidates stale generation work instead of mixing old and new guidance. `lang pack check` loads
every file and exits with an error naming the first mistake.

### What each file contains

**`rules/*.toml`** (one language pack per file; unknown keys are rejected):

| Key | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | `1` |
| `id` | yes | unique across all packs; lowercase letters, digits, `-` |
| `tags` | yes | language tags the file serves, e.g. `["fr"]` or `["zh-Hant"]` |
| `description` | yes | one line |
| `generation_guidance` | no | sentences added to every request: how to choose lemmas, what to keep as one unit |
| `[pos_aliases]` | no | maps the labels a model may use to the canonical part of speech (`verbe = "verb"`) |
| `[[lemma_conventions]]` | no | `id`, `pos`, `guidance`, `severity` (`guidance` or `error`) and for errors a `lemma_regex` the lemma must match |
| `[[learning_overrides]]`, `[[learning_exceptions]]`, `[[non_learning_patterns]]`, `[[decomposition_patterns]]` | no | rules that keep names, particles or productive compounds out of, or split them for, vocabulary practice; see `packs/zh/rules/zh.toml` |
| `pronunciation_classes`, `contextual_pronunciation_rules` | no | declarative pronunciation changes in context (Mandarin tone sandhi uses them) |

**`grammar/*.toml`** (the catalog the model annotates against):

| Key | Required | Meaning |
| --- | --- | --- |
| `schema_version`, `id`, `tags`, `description` | yes | as above; `id` is the language code (`fr`, `pt-br`) |
| `[[constructions]]` | yes, 1 to 30 | each with `key` (`<lang>:<name>`, lowercase, the prefix must be the catalog's language), `label`, `description`, `category`, `difficulty` (0 to 1) and optional `generation_hint` |

**`wordlist/pack.toml`** holds `languages = ["fr"]` (base languages) and `source = "..."` (the label
stored on each entry); **`wordlist.tsv`** is tab-separated with the header
`lemma<TAB>pronunciation<TAB>gloss<TAB>frequency_rank<TAB>level`. `pronunciation`, `frequency_rank`
and `level` may be empty. One list serves a base language and all its variants.

### Variants

A tag matches a pack when it equals one of the pack's tags or starts with it followed by `-`. So a
pack with `tags = ["de"]` serves `de-DE`, `de-AT` and `de-CH`, and `zh` serves `zh-Hans` and
`zh-Hant`. A more specific pack stacks on the general one: with packs for `zh` and `zh-Hant`, a
`zh-Hant` profile gets the guidance of both and the second wins on conflicting ids. Two packs may
not claim the same tag. Grammar catalogs and word lists follow the same prefix rule (most specific
catalog wins; word lists match the base language).

### Steps, with a worked example

The repository ships a small French pack in [`examples/packs/fr/`](examples/packs/fr/). To add a
language yourself:

**Step 1 – Make a directory outside the repository** (or inside `packs/` if you plan to contribute
the pack): `mkdir -p ~/lang-packs/fr/{rules,grammar,wordlist}`.

**Step 2 – Write the rules.** `~/lang-packs/fr/rules/fr.toml`:

```toml
schema_version = 1
id = "fr"
tags = ["fr"]
description = "French lemma conventions."
generation_guidance = [
  "Use the infinitive as the lemma of a verb and the masculine singular as the lemma of an adjective, keeping the observed form in run.text.",
  "Treat elided forms such as l' and j' as function words whose lemmas are le and je.",
]

[pos_aliases]
v = "verb"
verbe = "verb"
n = "noun"
nom = "noun"
adj = "adjective"
adjectif = "adjective"

[[lemma_conventions]]
id = "verb-infinitive"
pos = ["verb"]
guidance = "French verb lemmas use the dictionary infinitive."
severity = "guidance"
```

**Step 3 – Write a grammar catalog** (start with a handful of constructions; the model annotates
those it meets). `~/lang-packs/fr/grammar/fr.toml`:

```toml
schema_version = 1
id = "fr"
tags = ["fr"]
description = "Core French constructions."

[[constructions]]
key = "fr:passe-compose"
label = "Passé composé"
description = "A past tense built from avoir or être in the present plus a past participle."
category = "tense"
difficulty = 0.25
generation_hint = "Use a short sentence with avoir and a regular past participle."
```

**Step 4 – Optionally add a word list.** `~/lang-packs/fr/wordlist/pack.toml`:

```toml
languages = ["fr"]
source = "my-list"
```

and `wordlist.tsv` with a header and your rows (tabs between columns):

```
lemma	pronunciation	gloss	frequency_rank	level
maison		house	1	1
manger		to eat	2	1
```

Without a word list the language still works; new-word offers then rely on the model's choices.

**Step 5 – Point the app at it and check:** `export ARC_LANG_PACK_DIRS=~/lang-packs`, then
`uv run lang pack check fr-CA`. The report lists every pack and, for the tag, the pack it
resolves to, the number of guidance items, the grammar catalog and the word-list size.
With the files above and the bundled packs the tag `fr-CA` resolves to rules `fr`, 2 guidance
items, catalog `fr`, 2 word-list entries.

**Step 6 – Create a profile** and restart the worker so it sees the new directory variable:
`uv run lang profile create fr-fr --label French --learning-language fr-FR --translation-language en --select`.
`uv run lang brief` now lists the construction `fr:passe-compose` and the guidance. Set
`ARC_LANG_PACK_DIRS` in the environment of `./run.sh` (and of the Docker containers) as well as
of the CLI, or the worker will not see the pack.

The picker shows a human name for common languages (a list in `server/language_display.py`); for
others it uses the profile label you gave. Listen uses the browser's own voice for any language; the
optional neural speech extra supports only the languages it lists (`extras/tts-qwen/README.md`).

### Word-list licence

A word list is data, usually derived from someone else's work, so its licence is not the code's.
Before a list goes into a pack that you publish: put the licence text next to it (`LICENSE*`), state
in `SOURCES.md` where every column came from, and include only data you may redistribute under
terms compatible with how you distribute it. The bundled Chinese list is CC BY-SA 4.0 for exactly
this reason ([packs/README.md](packs/README.md)); delete it or replace it if you need other terms.
Pack rules and grammar catalogs are code-like text under the repository's Apache-2.0 licence unless
the pack says otherwise. A private pack in `ARC_LANG_PACK_DIRS` is never part of this repository.

## Access and exposure

The default (`ARC_LANG_AUTH=none`) binds the server to `127.0.0.1` and refuses requests whose `Host`
or `Origin` is not configured, which stops DNS-rebinding and cross-site requests to a locally running
copy. Never expose that mode to a network. Two opt-in modes exist for a remote instance:

- `ARC_LANG_AUTH=password` with `ARC_LANG_AUTH_PASSWORD_HASH` from `lang auth hash-password`: one
  password and a signed session cookie. Set `ARC_LANG_ALLOWED_HOSTS` to the hostnames you use.
- `ARC_LANG_AUTH=proxy` with `ARC_LANG_AUTH_PROXY_ADDRESSES` and `ARC_LANG_AUTH_PROXY_HEADER`: a
  reverse proxy in front has already authenticated the user and says so in a header; every other
  client is refused.

There is deliberately no multi-user mode. One instance, one learner.

## First lesson

New workspaces are dormant until the learner explicitly starts that language. The browser's short
onboarding form can record a rough reading starting point, confidence, interests, and preferred
text length; an agent can perform the same explicit action with `profile activate` after the
user chooses to begin. Activation is what permits automatic generation and listening work. Merely
creating or configuring a profile does not spend callback tokens, enqueue neural audio, or invoke
browser speech.

Automatic generation is enabled by default, so activation can enqueue the first calibration or
ordinary text. Start with `ARC_LANG_AUTO_GENERATE=0` when you want to prepare and import lessons
by hand instead.

An agent can set up a learner and import a generated lesson with:

```bash
uv run lang profile set \
  --learning-language es-ES \
  --translation-language en \
  --level A2 \
  --difficulty 0.27 \
  --interest history \
  --interest cooking \
  --text-length 300

uv run lang profile activate --starting-point unsure

uv run lang brief
uv run lang lesson schema
uv run lang lesson validate examples/es-demo.json
uv run lang lesson import examples/es-demo.json
```

The CLI prints JSON so a script or an assistant can inspect and update the app without scraping
the UI.

## Language profiles

Each language track has its own SQLite database and private agent workspace under
`data/profiles/PROFILE_ID/`. Older installs kept the first profile in `data/lang.db`;
`uv run lang registry migrate-legacy` moves that layout into a profile directory.

```bash
uv run lang profile list
uv run lang profile create zh-hant \
  --label "Chinese (Traditional)" \
  --learning-language zh-Hant \
  --translation-language en \
  --select
uv run lang --profile es-es brief
uv run lang --profile zh-hant profile set --level A1 --difficulty 0.1

uv run lang profile create de-de \
  --label "German" \
  --learning-language de-DE \
  --translation-language en
```

`profile select` sets the CLI default. `--profile` takes precedence, followed by
`ARC_LANG_PROFILE`. The browser profile switch has its own URL, event queue, and lesson sessions, so
multiple profile tabs cannot mix progress.

`profile delete PROFILE_ID` removes one non-selected, isolated profile workspace in full, including
its database, generated artifacts, and audio. It refuses the selected profile.

`profile reset PROFILE_ID` clears one non-selected profile's lessons, evidence, generated artifacts,
and audio while preserving its language entry and SQLite file. The recreated profile is inactive
and ready for the starting questionnaire;  Stop the
development app first so generation and audio workers cannot race the reset.

The picker presents human language/variant names rather than raw tags. It uses region flags where a
real region is present, a Latin America marker for `es-419`, and script-aware Chinese markers.
`zh-Hans` uses China only as an explicit simplified-script picker marker; unqualified `zh-Hant`
stays region-neutral instead of implying Taiwan. Accessible option labels spell out the marker's
meaning.

### Activation and placement

New profiles remain inactive even if the agent configures their interests or level first. Until an
explicit browser or CLI activation, the app shows the start screen and automatic queue maintenance,
topic requests, generation-task claims, local audio preparation, cached playback, and browser-speech
fallback all remain dormant. Manual configuration and lesson imports are preserved, but inactive
imports do not enqueue audio. Activation is idempotent; existing installations are migrated as
active only when their workspace already contains lessons.

If the level is unknown, the stored near-B1 value is only a neutral internal seed. The optional
questionnaire replaces it with a broad prior based on reading comfort; the confidence answer changes
the prior's width, not its status as evidence. “Not sure” supplies no prior. Interests and requested
text length shape generation but do not count as proficiency evidence.

Automatic generation requests one calibration text at a time. A calibration draft has 12 unique,
previously unused probe terms spread over requested difficulties, while all of its ordinary
vocabulary remains annotated and clickable. An attempt qualifies only after at least 30 active
seconds, 80% completion, no full translation, and at least eight probes outside
sentence-translated scopes. The questionnaire is never counted as an attempt. Two qualified texts
normally produce a rough estimate; four or more can become stable once the uncertainty interval is
narrow enough. Older timestamped evidence fades gradually, so placement can adapt after a long gap.

Supplying `--level` or `--difficulty` marks the profile as self-reported; activation with
`--starting-point unsure` preserves it, and derived placement does not overwrite it. Choosing a
specific questionnaire starting point instead deliberately replaces that value with a revisable
unknown-level prior. After placement, the first qualified completion of each ordinary lesson adds
low-weight level evidence. Word reveals are deliberately weak because they may be pronunciation or
meaning checks; sentence/full translation and explicit easier/more-challenging feedback carry more
weight. Rereads do not move overall level. A manual level change creates a stable evidence anchor,
so older readings cannot immediately undo the correction.

## Learning loop

1. The learner reads a generated text.
2. Clicking a term reveals its gloss and records an uncertainty signal. One isolated check is weak
   evidence; reveals in later sessions progressively strengthen it. If a legacy click span contains
   known productive components, one bounded signal is divided between them, with more assigned to
   the component whose current mastery is lower.
3. Sentence, grammar, or full-text help is available in layers. Opening a grammar marker is neutral;
   revealing its explanation records one direct-help signal for that occurrence and session. With a
   mouse, clicking between words opens a sentence's translation. On touch, a stray tap does not:
   long-press the sentence or use the floating Sentence button, and a translation closed within
   1.5 seconds is treated as a mistap and not recorded.
4. Completing a meaningfully read lesson gives weak recognition evidence only to terms that were not
   revealed in a translated scope.
5. The app rebuilds deterministic per-term, per-character, and per-construction learning state from
   raw history.
6. Authored calibration probes establish placement. Qualified first readings then refine it with
   lower-weight evidence; rereads, short/incomplete sessions, and uncalibrated unknown profiles do
   not move it accidentally.
7. `uv run lang brief` ranks due, fragile, and unseen terms and grammar. The agent chooses a small,
   natural subset for the next text, balanced with level, interests, and recent content.

This keeps the distinctive part of the original prototype—implicit testing through reading—without
making every unclicked render count as knowledge.

Vocabulary state is replayed chronologically from append-only interactions through the FSRS-4.5
memory model (`server/memory_model.py`). A reveal is a lapse ("again"); a clean, qualified reading is
a partial success ("good", weighted by `passive_confidence` because not clicking is weaker evidence
than a correct answer). Rereading a text is discounted further and is not a new exposure. FSRS barely
raises stability while predicted recall is still high, so massed exposure adds almost nothing, and
lapses shorten stability according to how surprising they were. Multiple raw reveals remain
inspectable, but only the first reveal of a term in one lesson session affects the rebuilt state.

Mastery is the time-aware probability of still recalling a word 30 days from now; it decays without
evidence. A word without evidence uses a frequency prior around the learner's frontier rank, and that
prior also seeds the first review, so common words a learner already knows are not treated as new.
`uv run lang memory evaluate --fit` scores the model one step ahead against real reveal history (for
the author's own zh-hans reading history: log-loss 0.43 and AUC 0.84, against 0.54 and 0.73 for the earlier Beta model).

Review priority ranks reviewed words by 1 - predicted recall now, with a bonus once recall falls below
90%, and never-reviewed words by how likely they are still unknown, plus corpus frequency,
uncertainty, and repeated failures across lessons. A rare one-off context lookup is down-weighted.
New vocabulary comes from frequency word lists (`packs/<lang>/wordlist/`, currently HSK for Chinese)
restricted to unencountered words near the learner's level, plus a few words the agent chooses itself
so new words keep arriving after a list is exhausted. Each import records its predicted share of known
running words; new-word counts steer toward a 95% target and text cards show each text's share.
Terms offered but not naturally used are cooled for the next offer instead of being forced into the
text.

Subjects follow **reading preferences**: free-text notes in Settings that start from your listed
interests, which you can edit and which the agent maintains. Messages sent from Settings, ratings,
skips, and abandoned texts are the signals; after a message or every few reactions the worker asks
the agent for an updated revision (history is kept). Each new text is a favourite (60%), a variation
of an interest (25%), or a new subject (15%); variation and new-subject texts state the question they
test, shown when you rate the text. The agent sees the last 20 texts with their reactions and leans
toward what worked.

Variety is enforced, not only requested. Every text declares one `subject_domain` (24 broad areas,
`SubjectDomain` in `server/schemas/base.py`; stored as `content_domain` and shown in the content
history). The content plan lists the domains of the last six texts as `avoid_domains`, and the host
rejects a prose draft that omits its domain or repeats one of them (the task fails with the allowed
domains in its error and gets the usual one retry). A requested topic or "more of this" feedback is
exempt. The plan also rotates one of 13 structural archetypes, none repeating within four texts,
including three that need no problem to solve (portrait, practical guide, recollection), and tells
the model that a liked story mechanic is not a subject. The preference notes are written as
subjects and domains, with a recurring plot shape listed as overused rather than as a taste.

Each refused draft is logged in the `domain_rejections` table (schema version 6): task, attempt,
kind (`missing`, `repeat`, `duplicate`), the declared, avoided and allowed domains and the time.
`lang generation rejections` lists them with the task's state and the tokens the refused call
used, so the rejection rate and its extra cost can be read. `lang content backfill-domains`
assigns domains to texts written before the rule from a JSON file of lesson key to domain; it only
adds `content_domain` (marked `content_domain_source: backfill`), never changes the text, and a
rerun changes nothing. `lang preferences refresh` runs the agent's normal preference update now,
without waiting for three reactions; it uses the stored messages and reactions only.
For Chinese, inferred character retrievability can move an otherwise-unseen word's candidate order
by at most 15%, making a word built from readable characters slightly easier to introduce. This
hint fades to zero after three direct word-level signals and never changes that word's mastery,
evidence, knowledge band, or due date.

Term underlines expose the current estimate without changing it: familiar (hidden until hover),
expected (quiet green), uncertain (amber), deliberate learning focus (blue), and incidental context
above the current level (muted violet). Only reading events change learning state; the bands are
recomputed from that state for each lesson.

The Words rail shows canonical terms from lessons that were actually opened; prepared-but-unread
lessons stay out of the list. A row is one stable lemma, part of speech, and meaning, with observed
surface forms grouped underneath. Other observed meanings or grammatical roles are linked while
keeping their evidence separate. Every column is sortable, and search covers word classes plus
tone-insensitive, spacing-insensitive pinyin. The list separates raw popup clicks from
lesson/session-deduplicated reveals that affect SRS, and also shows occurrences, clean passive
reads, mastery, and due state. Opening the list is read-only. The small reset control beside active
reading time only resets the local clock for the current lesson session and does not record learning
evidence. The motion policy is deliberately small: mouse movement opens a five-second countable
window, and on touch devices a tap, touch movement, or scroll opens a fifteen-second window, since
phone reading has no continuous pointer movement. Counting then pauses until the next activity.
Starting, resuming, resetting, or returning to the tab does not open a window by itself.

Chinese profiles also expose a **Characters** rail. Every literal Han character in the title and
body of an opened text is counted, including characters in display-only or older plain runs;
simplified and traditional characters remain distinct. Alongside those observations, a replayable
derived state estimates character recognition from the same append-only reading history. Each clean,
qualified reading session supplies one bounded positive observation per character, plus a small
bonus when it is met in a new canonical word context. Repeated appearances inside one session do not
multiply into independent successes. A reveal of a one-character run supplies bounded negative
evidence; a multi-character reveal shares a smaller total across its characters, weighted toward
the currently weaker or less certain character. Repeats are deduplicated and each character is
capped per lesson session. Sentence and full-text help suppress passive credit in their scopes.

Characters use the same FSRS model with their own policy: inferred reveals are partial lapses sized by
the failure mass, and clean sessions are partial successes with a new-context bonus. Retrievability is
current predicted recall and mastery is recall 30 days ahead. The initial prior is deliberately
neutral, and a character with no qualifying evidence is displayed as unestimated rather than known. Word evidence therefore helps estimate its component characters;
character knowledge only provides the small, fading generation-order hint described above. It does
not leak back into word SRS. Direct character-test counters are reserved for a future dedicated
exercise and remain zero for ordinary reading.

The Statistics rail is another read-only projection. **Words met** uses the same canonical,
opened-text vocabulary boundary as Words and groups the 0-1 mastery estimate into low-confidence,
developing, familiar, and strong ranges. **Texts read** counts unique completed lessons, while
completed sessions also expose rereads. Tracked active time sums the active timers sent with
completed sessions; unfinished session time remains browser-local. Recent pace is a token/time
weighted WPM over at most ten latest completions that reached 30 active seconds and 80% completion,
which excludes accidental test finishes. Chinese pace counts authored segmented word units rather
than individual characters. The level card shows the same current adaptive estimate used by the
reader and lesson generator, together with its calibration range. A self-report or questionnaire
answer remains the replay prior rather than a competing current value.

The Grammar rail follows the same opened-text boundary. Its cards group authored construction
occurrences, explanations, contextual examples, direct explanation reveals, weak sentence-help
inferences, clean completed readings, and a conservative confidence/due estimate. A single check is
not labelled a problem, repeated raw taps remain inspectable without being double-counted, and the
UI never claims a construction is permanently mastered. One qualified clean construction/session
adds one bounded positive observation; its first clean reading in a new lesson receives a small
context bonus, while repeated occurrences inside that lesson do not multiply into independent
trials. Grammar stability has its own configurable days-per-evidence coefficient and delayed-recall
bonus. In the reader, a quiet `G` marker first shows which construction is present without recording
evidence; only **Show explanation** records help. After strong, varied, stable evidence, a
not-yet-due construction loses its marker just as a familiar word loses its underline. It returns
when review is due or later evidence weakens the estimate; mixed sentences keep only the still-useful
notes. `Peek` mode records nothing.

The Texts rail is the reading library. It separates ready, in-progress, skipped, and previously read
texts. Pending and running generation jobs appear separately under **Being prepared** without
inventing a title, difficulty, or word count before the lesson exists; the library polls until each
placeholder becomes a real ready text. **Skip for now** removes an unfinished text from the active
queue without completing it or creating learning evidence; any real evidence already recorded
before the skip remains intact. The text stays recoverable under **Skipped**, where it can be
returned to the queue, opened immediately, or peeked at. Because skipped texts no longer count
toward the usable queue, automatic generation may prepare a replacement. Ready texts have compact
up/down controls; each move is an append-only adjacent swap, and the resulting durable order also
chooses the next default Reading text. `Peek` opens the complete reader without creating a session,
timer, Words exposure, or learning event. `Read again`
deliberately starts a fresh evidence-bearing session. A topic request creates a durable generation
job even when the normal unread queue is already full.
Here, the **title** is the reader-facing target-language heading; the **topic** is a short
organizational subject used for requests, filtering, continuity, and generation diversity. Lesson
rating is optional: **Finish & next** always records completion, while rating or fine-tuning
feedback is recorded only when supplied.

### Language packs and learning units

Runs are the immutable display and click boundary; learning units are the narrower vocabulary/SRS
boundary. TOML language packs under `packs/<lang>/rules/` provide generation guidance,
part-of-speech aliases, lemma conventions, and conservative include/exclude rules. `de`, `es`, and
`zh` currently have specialized packs, their regional/script variants inherit the base rules, and
other tags use safe generic guidance. Pack contents are included in generation fingerprints, so a
rule change invalidates stale generation work. To teach the app another language, see
[Adding a language](#adding-a-language).

The learning-unit index never fuzzy-stems or merges by spelling alone. Exact matches of normalized
lemma, normalized part of speech, gloss, and dictionary pronunciation share the first canonical SRS
identity; frequency rank is deliberately not part of that identity. Different meanings,
pronunciations, or grammatical roles remain separate. Existing/manual lessons can retain a
clickable phrase annotation even when it is not a reusable unit. The phrase itself remains
context-only; configured reusable components may appear in Words and receive projected evidence.

Generated lessons are asked for adjacent reusable runs. When Mandarin output still contains a
configured productive combination, the host splits it if every exact component definition is
available; otherwise the natural display span can remain while the learning layer projects it onto
known components. This prevents a known segmentation choice from discarding a complete text.
An unrelated callback key is rebound only when the visible surface has one exact local or stored
identity. Unknown or ambiguous identities still fail; the retry then receives the rejected compact
draft and repairs its annotations in place rather than generating a replacement story. Spanish and
German guidance keeps dictionary lemmas without guessing through inflection or merging homographs.

Grammar catalogs under `packs/<lang>/grammar/` define stable construction keys, learner
explanations, categories, rough 0-1 difficulty, and generation hints. Base catalogs currently cover
Spanish, German, and Mandarin and apply to their regional/script variants. Lessons anchor a catalog
construction to a sentence with a lesson-unique occurrence key and a half-open run range; an
optional note explains only what is special about that use. Old lessons can omit grammar entirely.
Catalog contents enter generation fingerprints.

Mandarin pronunciation is displayed as space-separated pinyin syllables when that alignment can be
established. Erhua keeps its final `r` on the preceding syllable and extends that syllable's color
over a written `儿`; otherwise the usual one-syllable-per-Han alignment applies. Joined and spaced
historical spellings resolve to the same Chinese vocabulary identity when the lemma, part of speech,
meaning, and actual reading match; new generation uses one space between syllables. Tone colors
follow Pleco's order: red, green, blue, purple, and gray for tones one through four and the neutral
tone. Chinese profiles also expose a profile-scoped Hanzi-tone toggle; it colors source characters
only when the pronunciation alignment is unambiguous. The canonical term stores its dictionary
pronunciation. An optional
`run.pronunciation` overrides it only for that occurrence, allowing contextual readings and tone
sandhi without splitting one lemma into false vocabulary identities; the popup and Hanzi/pinyin
rendering prefer the override.

For presentation, Chinese lesson difficulty is shown as an explicitly approximate HSK 1-9 band;
the stored CEFR-compatible level and continuous 0-1 adaptation model remain unchanged. The reader
places level, fine-grained difficulty, and the labelled topic after the text rather than crowding the
title. Settings provides a browser-local light/dark preference shared across language profiles and a
separate reading voice saved for each language profile.

## Optional local speech

After a language is activated, Listen works immediately with a compatible browser or
operating-system voice. For cached local neural audio, install the isolated Qwen3-TTS runtime (the
commands below are for an AMD ROCm machine):

```bash
extras/tts-qwen/install.sh rocm   # or cuda, or cpu (works, but slow)
./run.sh
```

The installer uses `uv`, Python 3.12, AMD's pinned ROCm wheels, and a pinned revision of
`Qwen3-TTS-12Hz-0.6B-CustomVoice`. The heavyweight Torch environment stays under ignored
`data/tts/`; the main app never imports it. `./run.sh` starts one audio worker when that runtime is
ready. Every supported-language lesson imported into an active profile enqueues an immutable,
profile-local WAV without blocking the import; the queue can exist before the optional runtime is
installed. Changing the voice in an active profile supersedes obsolete pending work and backfills
all texts one at a time. A voice selected while inactive is saved without starting work. Activation
allows the worker or audio-status path to backfill existing lessons, while cached files and imported
content are retained throughout. Old audio never masquerades as the newly selected voice.

Qwen's compact model supports Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian with nine fixed voices. It reads both `zh-Hans` and `zh-Hant` as
Mandarin while preserving the source script; it does not imply Cantonese or a Taiwan voice. It reads
both `es-ES` and `es-419` as Spanish, but its presets do not guarantee a Spain or Latin American
accent. German uses Qwen's German mode and starts with Ryan; because none of the nine presets is a
native-German voice, the voice selector does not promise a native regional accent. The 0.6B model
has no dependable prompt-based accent, emotion, or speed control, so playback speed is handled by
the reader. Voice cloning and arbitrary synthesis endpoints are intentionally out of scope.

The defaults are the conservative choice because one AMD RDNA4 ROCm stack crashes in the faster
FP16 attention path. Other installations may explicitly set `ARC_LANG_TTS_ATTENTION=sdpa` and
`ARC_LANG_TTS_DTYPE=float16` after a synthesis smoke test. Set `ARC_LANG_TTS_ENABLED=0` to leave the
local worker off; browser speech remains available for activated profiles.

## Neural audio on another machine

The reader can run on a small always-on host while speech is synthesized on a stronger machine.
Set `ARC_LANG_TTS_REMOTE=1` for the host's web service so audio status reports "queued" instead of
"unavailable". On the machine with the Qwen runtime (`extras/tts-qwen/install.sh`), run
`uv run python -m server.remote_tts_worker --host HOST` or install `extras/tts-qwen/lang-tts-remote.service`.
It claims one task at a time from the host's HTTP audio-worker API (`ARC_LANG_REMOTE_URL` and a
bearer token that matches the host's `ARC_LANG_AUDIO_WORKER_TOKEN`; without that variable the API
is off) or, alternatively, with `lang tts claim` over SSH, synthesizes locally, and uploads the WAV
(`/api/audio-worker/…/complete`, or `lang tts complete`). Each claim returns a `claim_token`; complete, fail and release require
`--claim TOKEN`. Reassigned claims reject an old worker's results. Remote queue operations are
serialized on the host, and an exact repeated upload returns the existing completion. A claim left
unfinished for 45 minutes returns to the queue without using an attempt; synthesis times out after
40 minutes so reporting fits inside that lease. Network loss leaves uncertain work for owner
readback/reclaim, while reported synthesis failures retain the two-attempt cap. WAV validation checks
all declared frames before publication. The Qwen process, and so its GPU memory, exists only while work is flowing: the worker exits it
before every idle or unreachable wait and the next claim reloads the model. Over SSH nothing is exposed beyond SSH; over HTTP only the bearer-protected `/api/audio-worker/` paths are, and they bypass the password/proxy login by design. Run either the local worker
or the remote queue on a host, never both. Upgrade the host and the audio worker together: stop the idle
worker, update the owner, then restart the worker.

## Commands

```bash
uv run lang profile show
uv run lang profile list
uv run lang profile create --help
uv run lang profile delete PROFILE_ID
uv run lang profile reset PROFILE_ID
uv run lang profile select PROFILE_ID
uv run lang profile set --help
uv run lang profile activate --help
uv run lang brief
uv run lang lesson schema
uv run lang lesson validate PATH
uv run lang lesson import PATH
uv run lang lesson list
uv run lang generation ensure
uv run lang generation list
uv run lang generation retry TASK_ID
uv run lang generation rejections
uv run lang content backfill-domains FILE.json
uv run lang preferences refresh
uv run lang provider list
uv run lang provider test NAME
uv run lang pack check [TAG...]
uv run lang rebuild
uv run lang status
./run_tests.sh
```

The profile registry and runtime data live under `data/` and are ignored by Git. Set
`ARC_LANG_DATA_DIR` to use another root. Every profile database and its agent job artifacts are
isolated below `data/profiles/PROFILE_ID/`.

Languages use BCP-47-style tags, so variants remain explicit throughout learning state and content:
`de-DE` (Germany), `es-ES` (Spain), `es-419` (Latin America), `zh-Hans` (Simplified Chinese), and
`zh-Hant` (Traditional Chinese). Difficulty is stored twice on purpose: a broad CEFR `level`
(`A1`-`C2`) and a fine `difficulty` value from 0 to 1.

### Code map

The backend keeps immutable input, derived learning state, and orchestration separate:

- `settings.py` reads and validates every `ARC_LANG_*` setting (the model providers are in
  `lang-reader.toml`, parsed by `providers.py`); `packs.py` finds language packs;
  `access.py` is the Host/Origin and login boundary.
- `schemas.py` and `models.py` define external contracts and durable rows.
- `learning.py` imports lessons, records events and builds the reader state. Around it:
  `profile_state.py` (profile, level source, profile view), `session_feedback.py`,
  `agent_brief.py` (the brief and target policy given to the lesson writer),
  `profile_fingerprint.py`, and `generation_queue.py` (durable generation tasks and triggers).
- `lesson_content.py` is the canonical typed view of stored lesson JSON and owns shared sentence,
  run, and token traversal.
- `lesson_activity.py` projects append-only interactions and queue actions into completion, skip,
  exposure, and status facts; `lesson_queue.py` owns the ready/unread queue definitions built from
  that projection.
- `lexeme_learning.py`, `character_learning.py`, `grammar.py`, and `proficiency.py` independently
  replay append-only evidence into disposable derived tables. `character_word_support.py` contains
  the small one-way Chinese candidate-order hint and cannot mutate either reducer.
- `reading_evidence.py`, `spaced_repetition.py`, and `learning_policy.py` hold the small validated
  policies shared by those reducers and read-side projections.
- `language_packs.py`, `packs/`, and `learning_units.py` contain language-specific behavior
  and stable vocabulary identity rules.
- `providers.py` parses the provider file, `model_callback.py` routes each stage to its provider,
  `model_http.py` makes the OpenAI-compatible and Anthropic calls and `agent_command.py` runs
  command providers (read answer and usage); `prompting.py` shapes the chat prompt for them; `agent_worker.py` coordinates stage execution; the stage callback boundary and
  its artifacts live in the adjacent `generation_*` modules. `generation_tasks.py` alone defines
  durable task kinds, modes, requested topics, and generated lesson keys.

On the frontend, `app.ts` owns navigation and browser-side orchestration. Domain renderers and pure
presentation logic live beside it in focused modules such as `statistics-view.ts`. Closed runtime
values are declared once in `contracts.ts`; language-tag, review-due, search, and term-band rules
live in their named pure modules. New behavior should enter the narrowest domain module;
compatibility imports in a façade are preferable to making reducers import orchestration.

## Automatic generation

`./run.sh` starts automatic queue maintenance by default, keeping three unread lessons ready—the
current text and two ahead:

```bash
./run.sh
```

This starts one durable worker alongside the reader. It snapshots the generation brief, closes the
database transaction, runs a read-only callback, validates the returned drafts, and imports them on
the host. The development launcher supervises generation in fresh worker processes, restarting an
unexpected exit with bounded backoff. Relevant Python or language/grammar TOML changes never
interrupt an active callback; the next worker starts with the updated source after that task ends.
The callback is a provider, read-only; it never receives write access. Each
ordinary task first writes frozen learning-language prose. One batched translation call then
overlaps lexical jobs that tokenize batches of sentences (8 by default,
`ARC_LANG_LEXICAL_BATCH_SIZE`; each result stays sentence-local and is validated on its own); at
most three callbacks run at once, of which at most two are lexical. Per-sentence calls repeated a
large fixed prompt for every sentence and cost most of a lesson's tokens. Grammar follows the fully assembled and validated lexical runs so its ranges have
stable anchors. The host scopes model-authored term keys to one sentence, restores established
identities, merges exact cross-sentence identities, and invokes a small lexical reconciliation only
for same-lemma definitions that code cannot safely identify as one sense. It then merges the
four compact stage results and rejects changed source text, missing or reordered keys, invalid
ranges, and cross-stage drift. Calibration remains a single complete callback. After each lesson
the worker takes a fresh learning snapshot for the next slot. A provider-neutral maintenance sweep
repairs missed queue updates every 30 seconds.
Each invalid sentence from a batch gets one immediate sentence-local repair with the exact host
diagnostic, and a sentence missing from a batch (or a failed batch) gets a fresh single-sentence
attempt; valid siblings are checkpointed and reused if the durable task itself must retry. Identity reconciliation
is split into independently validated bounded chunks when a lesson has more ambiguity groups than
one callback can accept; chunks may overlap within the same two lexical lanes and are globally
validated again before assembly. Every other failed DAG stage likewise gets at most one repair
attempt. Every already-validated independent stage is materialized into the next attempt, and only
a failed unit or stage plus its dependents reruns. Ordinary staged work also has an absolute
six-attempt safety cap, enough for one failure in every stage followed by success; legacy calibration
retains its two-attempt cap. A terminal failure remains visible and maintenance does not loop on it;
a fresh accepted reading interaction or skip action can authorize one idempotent successor.
Maintenance compares durable relevant-event and skipped-action high-water marks, so it also recovers
if the process stops between committing evidence and waking the queue. A worker interruption uses
the same bounded failure budget and remains available as retry context. Tasks and callback logs are
inspectable with the `lang generation` commands above. Each task attempt also appends a private
`timings.jsonl` beside `callback.log`, recording callback duration, queue wait, request/response
bytes, available token usage, conflict-group counts, and host preparation, assembly, import, and
total-task spans without copying lesson content into the timing records.

Inactive profiles are skipped during queue maintenance and task claiming, so merely provisioning a
language never invokes an agent callback. If this worker is already running, explicit activation can
enqueue the first required text; otherwise the agent can still use the manual lesson workflow.

For an explicitly manual, token-saving run, disable the worker:

```bash
ARC_LANG_AUTO_GENERATE=0 ./run.sh
```

Unknown-level profiles temporarily use a queue target of one, allowing the second calibration text
to adapt to the first. Once placement is rough, the configured ordinary unread queue target applies.

Workers call the providers of `lang-reader.toml` (see [Connect a model](#connect-a-model)) and
nothing else, including manually started workers. `ARC_LANG_ADMISSION_COMMAND` may name a host admission command
that decides whether a model call may start at all: only exit 0 permits it, and an error or timeout
defers. Observations are cached for at most 60 seconds.

If the provider reports *unavailable* (exit 3 of a command, an unreachable server or HTTP 429/503) after admission, the same task stays
pending with a 60-second retry delay. Already validated stage artifacts survive. Dispatch numbers
(`attempts`) keep increasing to preserve separate artifact directories, while `admission_deferrals`
in the task payload discounts confirmed waits from the draft-attempt cap. Ordinary errors and
timeouts still consume that cap; availability never authorizes retrying an uncertain tool effect.
Preference updates also retain their pending input when the role is unavailable.

An unavailable provider never silently falls back to another account or model you did not choose: only the fallbacks you list in `[stages]` are tried.

A `command` provider is executed as an argv list without a shell. It receives canonical requests with `schema_version`, provider-neutral `stage`, and `task`
fields on stdin. Prose, sentence-local lexical annotation, optional lexical conflict reconciliation,
translation, and grammar each have a strict compact result schema; both lexical request kinds keep
the same provider-neutral `lexical` task name. Legacy calibration returns the complete lesson shape.
A command adapter must permit bounded lexical and translation calls to run concurrently. Stable
environment fields are
`ARC_LANG_PROTOCOL`, `ARC_LANG_JOB_ID`, `ARC_LANG_REQUEST_SHA256`, and the request/response path
fields prefixed with `ARC_LANG_CALLBACK_`; `ARC_LANG_CALLBACK_STAGE` identifies the current phase.
`ARC_LANG_PROFILE` and `ARC_LANG_WORKSPACE` identify the task scope. The host owns lesson keys,
languages, metadata, database writes, and queue reconciliation. `ARC_LANG_JOB_ID` is globally unique
in the form `PROFILE_ID:TASK_ID`; the local numeric task ID remains available as
`ARC_LANG_GENERATION_TASK_ID`.

The callback contract deliberately separates text coverage from curriculum choices. The title and
every body lexical token in a generated draft must be a term-bearing run with an approximate corpus
`frequency_rank`; proper names may leave the rank null and remain context-only. `title_sentence`
contains the clickable title, an exact text match, a unique sentence key, and its translation. Plain
runs may contain only whitespace and punctuation. `target_term_keys` is an advisory teaching subset
reconciled against the request's dynamic candidate policy and body terms actually present;
calibration requests carry their own 8-20 authored probe subset. The host rejects sparse annotation
and conflicting definitions for a stable term key. Manual lesson imports remain backward compatible
and may omit title annotation, full body annotation, or frequency ranks.

At import time, callback keys are provisional. A reference to a known key with the same normalized
lemma is restored to the complete library definition. If the key is already owned by a different
lemma, the host assigns the new definition a deterministic collision-safe key and rewrites run,
target, and calibration references. This also checks stored display-only terms, not just Words/SRS
entries, so an old non-learning span cannot make a later generated lesson fail at import.

For ordinary lessons, the host also owns the numeric challenge and requested length. The prompt
states both explicitly; drafts more than 0.05 away from the requested difficulty or below 60% of the
requested body lexical length are rejected with exact actual/requested diagnostics. Accepted lesson
metadata preserves the provider's declaration but the displayed/adaptive difficulty is the host
target, preventing a model from making easy prose harder merely by relabelling it.

Each request also carries the resolved language pack's generation guidance. Generated drafts must
use reusable language-aware learning units; a contextual `run.pronunciation` may override the
canonical dictionary pronunciation for one surface occurrence without changing the stable term
definition.

Requests also carry the complete resolved grammar catalog and a smaller ranked `priority_grammar`
choice pool. That pool is not a checklist: its preferred and maximum opportunity counts adapt to
text length, placement confidence, difficulty, calibration, and grammar feedback, and zero deliberate
uses is valid when none fit naturally. The callback must nevertheless annotate every clear catalog
construction that appears incidentally, using precise run ranges. The host records offered,
realized, and annotated grammar keys in lesson metadata so skipped choices can cool instead of being
forced into the next text.

Ordinary callback requests also carry one deterministic `content_plan`: a coherent discourse form,
perspective, concrete seed, progression, and ending shape selected to avoid recent plans. Recent
lesson openings and endings are included as short negative examples, while stale target/grammar job
bookkeeping and old calibration probe arrays are omitted from the brief. Linguistic difficulty
controls vocabulary and syntax, not whether the content may be specific or structurally varied. The
callback chooses the situation before selecting optional SRS terms, and the chosen plan is stored in
lesson metadata so later requests can rotate away from it. Calibration retains its narrower probe
contract without a content plan.

## Settings

`server/settings.py` is the single place that reads configuration from the environment: an unknown
or malformed value fails at start. Booleans take `0`/`1`. The common settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARC_LANG_DATA_DIR` | `./data` | Root of all runtime state (profiles, databases, audio, job artifacts) |
| `ARC_LANG_BASE_PATH` | empty | URL prefix when a reverse proxy serves the app under a path, such as `/lang` (no trailing slash) |
| `ARC_LANG_PROFILE_COOKIE` | `lang_profile` | Name of the cookie that remembers the browser's selected profile; change it only if another app on the same host uses the name |
| `ARC_LANG_PROFILE` | none | Default profile for the CLI (`--profile` wins) |
| `ARC_LANG_PACK_DIRS` | none | Comma-separated extra [pack](packs/README.md) roots, for private packs |
| `ARC_LANG_AUTO_GENERATE` | `1` | `0` turns the generation worker off (`run.sh`) |
| `ARC_LANG_QUEUE_TARGET` | `3` | Unread lessons to keep ready (1-10) |
| `ARC_LANG_CONFIG` | `./lang-reader.toml` | Path of the provider file; required to generate ([Connect a model](#connect-a-model)). Timeouts are per provider (`timeout_seconds`) |
| `ARC_LANG_AGENT_POLL_SECONDS` | `2` | How long an idle generation worker sleeps before looking for work again (1-60) |
| `ARC_LANG_AGENT_MAINTENANCE_SECONDS` | `30` | Interval of the queue maintenance sweep (1-3600) |
| `ARC_LANG_AGENT_UNAVAILABLE_EXIT` | `3` | Exit status of a command provider that means "no capacity, retry later" |
| `ARC_LANG_LEXICAL_BATCH_SIZE` | `8` | Sentences per lexical call (1-16) |
| `ARC_LANG_ADMISSION_COMMAND` | none | Optional command that must exit 0 before a model call starts (for example to respect a usage budget) |
| `ARC_LANG_AUTH`, `ARC_LANG_ALLOWED_HOSTS` | `none` | Access mode and allowed hostnames ([Access and exposure](#access-and-exposure)) |
| `ARC_LANG_AUDIO_FORMAT` | `wav` | Stored audio format, `wav` or `opus`; `lang audio compress --format opus\|mp3` converts an existing cache |
| `ARC_LANG_TTS_COMMAND` | the Qwen extra, if installed | Replaces the bundled speech provider. It runs as a long-lived process that exchanges JSON lines on stdin/stdout, the protocol `extras/tts-qwen/qwen_provider.py` implements; read that file for the message shapes |
| `ARC_LANG_TTS_TIMEOUT_SECONDS` | `3600` | Time limit for one local synthesis (60-7200) |
| `ARC_LANG_TTS_ENABLED`, `ARC_LANG_TTS_REMOTE` | `1`, `0` | Local audio worker on or off in `run.sh`; remote queue mode ([Neural audio on another machine](#neural-audio-on-another-machine)) |
| `ARC_LANG_SNAPSHOT_DIR` | none | Read-only recovery mode ([above](#inspect-a-read-only-recovery-snapshot)) |

The remote audio worker variables (`ARC_LANG_AUDIO_WORKER_TOKEN`, `ARC_LANG_REMOTE_*`) are described
in [Neural audio on another machine](#neural-audio-on-another-machine) and
`extras/tts-qwen/README.md`; `ARC_LANG_TTS_ATTENTION` and `ARC_LANG_TTS_DTYPE` in
[Optional local speech](#optional-local-speech).

## Content model direction

Keep independent axes instead of one flat genre list:

- discourse form: personal recount, story, article or explainer, dialogue, diary or message;
- fictionality: factual, fictional, or simulated real life;
- genre or tone: science fiction, mystery, slice of life, humour;
- topic or domain: daily life, history, food, technology;
- continuity: standalone, episodic, or serialized.

A normal “my day” text is usually a personal recount with a daily-life topic and slice-of-life tone.
It can be standalone or part of an episodic diary. Declared choices, agent inferences, and recent
feedback adjustments should retain their provenance instead of becoming one flat interest list.

A series should have a stable key, discourse form, fictionality, genres, premise, recurring cast or
concepts, compact continuity summary, open threads, and next episode number. Lessons reference that
key and store an episode summary; generation receives the compact series state rather than every
previous text.
Wordlists and explicit review should consume the same lexeme state and event history as reading.

## Licence and content

Code, language rules, grammar catalogs and the demo lessons are Apache-2.0. The Chinese word list in
`packs/zh/wordlist/` is CC BY-SA 4.0 because its glosses and pinyin come from CC-CEDICT; see
[packs/README.md](packs/README.md) and `packs/zh/wordlist/SOURCES.md`. The demo lessons in
`examples/` were generated by a model and are offered as CC0. No lesson text, learning history or
audio is redistributed with the code: `data/` and audio never enter Git.

## Scope

The first version covers reading, term/sentence/grammar/full translations, replayable vocabulary and
grammar adaptation, agent briefs, lesson import, a text library with explicit no-evidence previews,
topic requests, and basic cached TTS. Text series will become collections inside Texts rather than a
separate rail item. Explicit quizzes, richer grammar diagnostics, synchronized sentence audio, and
deeper progress views remain future work.
