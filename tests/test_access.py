from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import access
from server.access import SESSION_COOKIE, Access
from server.passwords import hash_password, verify_password
from server.settings import AuthSettings, load

PASSWORD = "correct horse battery"


def _client(auth: AuthSettings, base: str = "http://localhost") -> Iterator[TestClient]:
    app = FastAPI()
    app.middleware("http")(Access(auth))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def home() -> dict[str, str]:
        return {"page": "home"}

    @app.post("/api/act")
    def act() -> dict[str, str]:
        return {"acted": "yes"}

    with TestClient(app, base_url=base, follow_redirects=False) as client:
        yield client


def _auth(mode: str = "none", **changes: object) -> AuthSettings:
    values: dict[str, object] = {
        "mode": mode,
        "password_hash": None,
        "proxy_header": "X-Forwarded-User",
        "proxy_addresses": (),
        "allowed_hosts": (),
    }
    values.update(changes)
    return AuthSettings(**values)  # type: ignore[arg-type]


@pytest.fixture
def open_client() -> Iterator[TestClient]:
    yield from _client(_auth())


def test_loopback_hosts_are_served_and_other_hosts_are_refused(open_client: TestClient) -> None:
    assert open_client.get("/").status_code == 200
    assert open_client.get("/", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert open_client.get("/", headers={"host": "[::1]:8000"}).status_code == 200
    # DNS rebinding: a hostile name that resolves to 127.0.0.1.
    assert open_client.get("/", headers={"host": "evil.example"}).status_code == 400
    assert open_client.get("/", headers={"host": ""}).status_code == 400


def test_configured_hosts_are_allowed() -> None:
    for client in _client(_auth(allowed_hosts=("lang.example",)), "http://lang.example"):
        assert client.get("/").status_code == 200


def test_cross_site_writes_are_refused_but_same_site_and_cli_clients_pass(
    open_client: TestClient,
) -> None:
    assert (
        open_client.post("/api/act", headers={"origin": "https://evil.example"}).status_code == 403
    )
    assert (
        open_client.post("/api/act", headers={"origin": "http://localhost:8000"}).status_code == 200
    )
    assert open_client.post("/api/act").status_code == 200  # curl, the CLI: no Origin
    # Reads are not state changes.
    assert open_client.get("/", headers={"origin": "https://evil.example"}).status_code == 200


def test_password_mode_requires_a_login_and_remembers_it() -> None:
    auth = _auth("password", password_hash=hash_password(PASSWORD))
    for client in _client(auth):
        assert client.get("/api/health").status_code == 200
        redirect = client.get("/")
        assert redirect.status_code == 303 and redirect.headers["location"] == "/login"
        assert client.post("/api/act").status_code == 401
        assert client.get("/login").status_code == 200

        wrong = client.post("/login", data={"password": "nope"})
        assert wrong.status_code == 401 and SESSION_COOKIE not in wrong.cookies
        right = client.post("/login", data={"password": PASSWORD})
        assert right.status_code == 303 and SESSION_COOKIE in right.cookies
        assert client.get("/").json() == {"page": "home"}
        assert client.post("/api/act").status_code == 200
        client.cookies.set(SESSION_COOKIE, "123.forged")
        assert client.get("/").status_code == 303


def test_an_expired_or_foreign_session_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = _auth("password", password_hash=hash_password(PASSWORD))
    other = _auth("password", password_hash=hash_password(PASSWORD))  # another salt, another key
    for client in _client(auth):
        client.cookies.set(
            SESSION_COOKIE, access._sign(other.password_hash or "", int(time.time()) + 99)
        )
        assert client.get("/").status_code == 303
        client.cookies.set(
            SESSION_COOKIE, access._sign(auth.password_hash or "", int(time.time()) - 1)
        )
        assert client.get("/").status_code == 303


def test_repeated_wrong_passwords_are_throttled() -> None:
    auth = _auth("password", password_hash=hash_password(PASSWORD))
    for client in _client(auth):
        statuses = [client.post("/login", data={"password": "bad"}).status_code for _ in range(7)]
        assert statuses[: access.LOGIN_FAILURE_LIMIT] == [401] * access.LOGIN_FAILURE_LIMIT
        assert statuses[-1] == 429
        assert client.post("/login", data={"password": PASSWORD}).status_code == 429


def test_proxy_mode_trusts_only_the_named_proxy_with_its_header() -> None:
    # TestClient connects from the address "testclient", which is not an IP: never trusted.
    for client in _client(_auth("proxy", proxy_addresses=("127.0.0.1",))):
        assert client.get("/", headers={"X-Forwarded-User": "me"}).status_code == 401
        assert client.get("/api/health").status_code == 200

    auth = _auth("proxy", proxy_addresses=("127.0.0.1",))
    gate = Access(auth)

    class Address:
        host = "127.0.0.1"

    class Fake:
        client = Address()
        headers = {"X-Forwarded-User": "me"}

    assert gate._from_trusted_proxy(Fake())  # type: ignore[arg-type]
    Fake.client.host = "10.0.0.9"  # type: ignore[misc]
    assert not gate._from_trusted_proxy(Fake())  # type: ignore[arg-type]


def test_password_hashes_verify_and_are_salted() -> None:
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and verify_password(PASSWORD, first)
    assert not verify_password("other", first)
    assert not verify_password(PASSWORD, "plaintext")


def test_settings_fail_fast_on_unusable_auth_configuration() -> None:
    with pytest.raises(ValueError, match="PASSWORD_HASH"):
        load({"ARC_LANG_AUTH": "password"})
    with pytest.raises(ValueError, match="PROXY_ADDRESSES"):
        load({"ARC_LANG_AUTH": "proxy"})
    with pytest.raises(ValueError, match="ARC_LANG_AUTH"):
        load({"ARC_LANG_AUTH": "magic"})


def test_the_real_application_checks_the_host() -> None:
    from server.main import app

    with TestClient(app, base_url="http://evil.example") as client:
        assert client.get("/api/health").status_code == 400
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").json()["status"] == "ok"
