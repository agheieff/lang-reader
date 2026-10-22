"""Who may reach the web interface: Host/Origin checks plus an optional login.

Lang serves exactly one learner per instance. Three modes (``ARC_LANG_AUTH``):

* ``none``: loopback use. Only the Host/Origin checks apply, which stop DNS rebinding and
  cross-site requests to a locally run server. Do not expose this mode to a network.
* ``password``: one password (``ARC_LANG_AUTH_PASSWORD_HASH`` from ``lang auth hash-password``)
  and a signed session cookie.
* ``proxy``: a reverse proxy at ``ARC_LANG_AUTH_PROXY_ADDRESSES`` has already authenticated the
  user and says so with a header (``ARC_LANG_AUTH_PROXY_HEADER``); anyone else is refused.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import time
from collections.abc import Awaitable, Callable
from html import escape
from urllib.parse import parse_qs, urlsplit

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.responses import Response

from server.base_path import BASE_PATH, app_path
from server.passwords import verify_password
from server.settings import AuthSettings

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})
SESSION_COOKIE = "lang_session"
SESSION_SECONDS = 30 * 24 * 3600
LOGIN_FAILURE_LIMIT = 5
LOGIN_FAILURE_WINDOW_SECONDS = 60
WORKER_PATH_PREFIX = "/api/audio-worker/"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _sign(key: str, expires: int) -> str:
    mac = hmac.new(key.encode(), f"session:{expires}".encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{mac}"


def _session_valid(key: str, token: str | None, now: float) -> bool:
    if not token:
        return False
    expires_text, _, _mac = token.partition(".")
    try:
        expires = int(expires_text)
    except ValueError:
        return False
    return expires > now and hmac.compare_digest(_sign(key, expires), token)


def _host_name(value: str) -> str:
    """The host part of a Host header or an Origin URL, without port, lower-cased."""
    host = urlsplit(value if "//" in value else f"//{value}").hostname
    if host is None:
        return ""
    return f"[{host}]" if ":" in host else host.lower()


class Access:
    def __init__(self, auth: AuthSettings) -> None:
        self.auth = auth
        self.allowed_hosts = frozenset(auth.allowed_hosts) | LOOPBACK_HOSTS
        self._failures: list[float] = []

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host = _host_name(request.headers.get("host", ""))
        if host not in self.allowed_hosts:
            return JSONResponse({"detail": "Host not allowed"}, status_code=400)
        origin = request.headers.get("origin")
        cross_site = origin is not None and _host_name(origin) not in self.allowed_hosts
        if request.method in UNSAFE_METHODS and cross_site:
            return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        denied = self._authenticate(request)
        if denied is not None:
            return denied
        if request.url.path == "/login":
            return await self._login(request)
        return await call_next(request)

    def _authenticate(self, request: Request) -> Response | None:
        path = request.url.path
        mode = self.auth.mode
        if mode == "none" or path == "/api/health" or path.startswith(WORKER_PATH_PREFIX):
            return None  # the audio worker API checks its own bearer token
        if mode == "proxy":
            if self._from_trusted_proxy(request) and request.headers.get(self.auth.proxy_header):
                return None
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        if path == "/login":
            return None
        assert self.auth.password_hash is not None
        if _session_valid(
            self.auth.password_hash, request.cookies.get(SESSION_COOKIE), time.time()
        ):
            return None
        if request.method == "GET" and not path.startswith("/api/"):
            return RedirectResponse(app_path("/login"), status_code=303)
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    def _from_trusted_proxy(self, request: Request) -> bool:
        client = request.client.host if request.client else ""
        try:
            address = ipaddress.ip_address(client)
        except ValueError:
            return False
        return any(
            address in ipaddress.ip_network(entry, strict=False)
            for entry in self.auth.proxy_addresses
        )

    async def _login(self, request: Request) -> Response:
        if self.auth.mode != "password":
            return RedirectResponse(app_path("/"), status_code=303)
        if request.method == "GET":
            return HTMLResponse(_login_page(app_path("/login")))
        if request.method != "POST":
            return JSONResponse({"detail": "Method not allowed"}, status_code=405)
        now = time.time()
        self._failures = [t for t in self._failures if now - t < LOGIN_FAILURE_WINDOW_SECONDS]
        if len(self._failures) >= LOGIN_FAILURE_LIMIT:
            return HTMLResponse(_login_page(app_path("/login"), "Too many attempts."), 429)
        form = (await request.body()).decode(errors="replace")
        password = _form_value(form, "password")
        assert self.auth.password_hash is not None
        if not verify_password(password, self.auth.password_hash):
            self._failures.append(now)
            return HTMLResponse(_login_page(app_path("/login"), "Wrong password."), 401)
        response = RedirectResponse(app_path("/"), status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            _sign(self.auth.password_hash, int(now) + SESSION_SECONDS),
            max_age=SESSION_SECONDS,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path=BASE_PATH or "/",
        )
        return response


def _form_value(body: str, name: str) -> str:
    return parse_qs(body, keep_blank_values=True).get(name, [""])[0]


def _login_page(action: str, message: str = "") -> str:
    note = f"<p>{escape(message)}</p>" if message else ""
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        "<title>Sign in</title><main style='max-width:22rem;margin:20vh auto;font:1rem sans-serif'>"
        f"<h1>Sign in</h1>{note}<form method=post action='{escape(action)}'>"
        "<input type=password name=password autofocus required style='width:100%'>"
        "<button style='margin-top:.5rem'>Sign in</button></form></main>"
    )
