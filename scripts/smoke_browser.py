"""Browser smoke test: serve a temporary instance with the demo lessons and drive it in Chromium.

Needs a built front end (``pnpm build``) and a Chromium or Chrome binary (``CHROME`` or one on
PATH). It never touches your data directory. Exit status 0 means every check passed.

    uv run python scripts/smoke_browser.py [--if-available]

``--if-available`` skips (exit 0) when no browser is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome")
DEMOS = ("es-demo", "es-demo-lighthouse", "es-demo-market")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def find_browser() -> str | None:
    candidates = [os.environ.get("CHROME", ""), *(shutil.which(name) or "" for name in BROWSERS)]
    return next((candidate for candidate in candidates if candidate), None)


def wait_for(url: str, seconds: float = 30.0) -> Any:
    deadline = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read())
        except OSError:
            if time.monotonic() > deadline:
                raise SystemExit(f"Timed out waiting for {url}") from None
            time.sleep(0.2)


class Page:
    """One browser tab over the DevTools protocol, collecting console and page errors."""

    def __init__(self, debug_port: int) -> None:
        tabs = wait_for(f"http://127.0.0.1:{debug_port}/json")
        target = next(tab for tab in tabs if tab["type"] == "page")
        self.socket = connect(target["webSocketDebuggerUrl"], max_size=2**26)
        self.next_id = 0
        self.errors: list[str] = []
        for domain in ("Runtime", "Log", "Page"):
            self.call(f"{domain}.enable")

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.next_id += 1
        self.socket.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return dict(message["result"])
            self._note(message)

    def _note(self, message: dict[str, Any]) -> None:
        method, params = message.get("method"), message.get("params", {})
        if method == "Runtime.exceptionThrown":
            details = params["exceptionDetails"]
            self.errors.append(details.get("exception", {}).get("description", "exception"))
        elif method == "Runtime.consoleAPICalled" and params["type"] == "error":
            self.errors.append(
                "console.error " + " ".join(str(a.get("value")) for a in params["args"])
            )
        elif method == "Log.entryAdded" and params["entry"]["level"] == "error":
            self.errors.append(params["entry"]["text"])

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if "exceptionDetails" in result:
            raise RuntimeError(f"evaluating {expression[:80]!r} threw")
        return result["result"].get("value")

    def open(self, url: str, ready: str) -> None:
        self.call("Page.navigate", url=url)
        self.wait(f"location.href === {json.dumps(url)} && document.readyState === 'complete'")
        self.wait(ready)

    def wait(self, condition: str, seconds: float = 10.0) -> None:
        deadline = time.monotonic() + seconds
        while True:
            try:
                if self.evaluate(f"Boolean({condition})"):
                    return
            except RuntimeError:
                pass  # the page is between documents
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for: {condition}")
            time.sleep(0.1)


def cli(*arguments: str, data: Path) -> None:
    environment = {**os.environ, "ARC_LANG_DATA_DIR": str(data), "ARC_LANG_AUTO_GENERATE": "0"}
    subprocess.run(
        [sys.executable, "-m", "server.cli", *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )


def check_views(page: Page, base: str) -> None:
    page.open(base, "document.querySelector('#lesson-content button.term')")
    assert page.evaluate("document.getElementById('lesson-title').textContent.trim().length > 0")

    page.evaluate("document.querySelector('#lesson-content button.term').click()")
    page.wait("!document.getElementById('gloss-popover').hidden")
    page.evaluate("document.getElementById('gloss-close').click()")

    page.evaluate("document.getElementById('full-translation-button').click()")
    page.wait("!document.getElementById('full-translation').hidden")

    page.evaluate("document.getElementById('sentence-help-mode').click()")
    page.evaluate("document.querySelector('#lesson-content .sentence-reveal').click()")
    page.wait("!document.getElementById('sentence-translation-popover').hidden")

    page.open(f"{base}?view=texts", "document.querySelectorAll('#texts-view .text-card').length")
    assert page.evaluate("document.querySelectorAll('#texts-view .text-card').length") >= 3

    page.open(f"{base}?view=words", "document.querySelectorAll('#words-table-body tr').length")
    # Revealing a term in the reader must have reached the server as learning evidence.
    page.wait("/1\\s*popup clicks/i.test(document.getElementById('words-summary').innerText)")
    page.evaluate(
        "(() => { const s = document.getElementById('words-search'); s.value = 'zzzz';"
        " s.dispatchEvent(new Event('input', { bubbles: true })); })()"
    )
    page.wait("!document.getElementById('words-empty').hidden")

    page.open(f"{base}?view=grammar", "document.getElementById('grammar-summary').children.length")
    page.open(
        f"{base}?view=statistics", "document.getElementById('statistics-word-total').innerText"
    )
    page.open(f"{base}?view=settings", "document.querySelector('input[name=color-theme]')")
    page.evaluate("document.querySelector('input[name=color-theme][value=dark]').click()")
    page.wait("document.documentElement.dataset.theme === 'dark'")


def check_narrow_screen(page: Page, base: str) -> None:
    page.call(
        "Emulation.setDeviceMetricsOverride",
        width=390,
        height=844,
        deviceScaleFactor=2,
        mobile=True,
    )
    page.open(base, "document.querySelector('#lesson-content button.term')")
    overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the reader scrolls sideways by {overflow}px at 390px wide"


def main() -> None:
    if not (ROOT / "server" / "static" / "app.js").is_file():
        raise SystemExit("Build the front end first: pnpm build")
    browser = find_browser()
    if browser is None:
        if "--if-available" in sys.argv:
            print("browser smoke test skipped: no Chromium/Chrome found")
            return
        raise SystemExit("No Chromium/Chrome found; set CHROME to its path.")
    with tempfile.TemporaryDirectory(prefix="lang-smoke-") as temporary:
        data = Path(temporary) / "data"
        # A fresh data directory already holds the default es-es profile.
        cli("--profile", "es-es", "profile", "activate", "--starting-point", "unsure", data=data)
        for demo in DEMOS:
            cli(
                "--profile",
                "es-es",
                "lesson",
                "import",
                str(ROOT / "examples" / f"{demo}.json"),
                data=data,
            )
        app_port, debug_port = free_port(), free_port()
        environment = {
            **os.environ,
            "ARC_LANG_DATA_DIR": str(data),
            "ARC_LANG_AUTO_GENERATE": "0",
            "ARC_LANG_TTS_ENABLED": "0",
        }
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(app_port),
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        chrome = subprocess.Popen(
            [
                browser,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--remote-debugging-port={debug_port}",
                f"--user-data-dir={temporary}/chrome",
                "--window-size=1280,1600",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            base = f"http://127.0.0.1:{app_port}/p/es-es"
            wait_for(f"http://127.0.0.1:{app_port}/api/health")
            page = Page(debug_port)
            check_views(page, base)
            check_narrow_screen(page, base)
            assert not page.errors, "browser errors:\n" + "\n".join(page.errors)
        finally:
            chrome.terminate()
            server.terminate()
            chrome.wait(timeout=20)
            server.wait(timeout=20)
    print("browser smoke test passed")


if __name__ == "__main__":
    main()
