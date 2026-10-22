"""A minimal `kind = "command"` provider: forward the prompt to a chat endpoint, print the answer.

It shows the whole protocol with the standard library only. Environment (pass the ADAPTER_*
names on with `pass_env` in lang-reader.toml): ADAPTER_URL (the full chat/completions URL),
ADAPTER_MODEL and, optionally, ADAPTER_KEY_FILE. Exit 3 means "no capacity right now".
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

prompt = Path(os.environ["ARC_LANG_CALLBACK_PROMPT_PATH"]).read_text(encoding="utf-8")
schema = Path(os.environ["ARC_LANG_CALLBACK_RESPONSE_SCHEMA_PATH"]).read_text(encoding="utf-8")
headers = {"Content-Type": "application/json"}
if key_file := os.environ.get("ADAPTER_KEY_FILE"):
    headers["Authorization"] = "Bearer " + Path(key_file).read_text(encoding="utf-8").strip()
body = {
    "model": os.environ["ADAPTER_MODEL"],
    "messages": [{"role": "user", "content": f"{prompt}\n\nOutput JSON schema:\n{schema}"}],
}
request = urllib.request.Request(
    os.environ["ADAPTER_URL"], json.dumps(body).encode(), headers, method="POST"
)
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.load(response)
except urllib.error.HTTPError as error:
    print(f"HTTP {error.code}", file=sys.stderr)
    sys.exit(3 if error.code in (429, 503) else 1)
except urllib.error.URLError as error:
    print(f"unreachable: {error.reason}", file=sys.stderr)
    sys.exit(3)
sys.stdout.write(answer["choices"][0]["message"]["content"])
usage = answer.get("usage", {})
if usage_path := os.environ.get("ARC_LANG_CALLBACK_USAGE_PATH"):
    Path(usage_path).write_text(
        json.dumps(
            {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        ),
        encoding="utf-8",
    )
