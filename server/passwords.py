"""Password hashes for ``ARC_LANG_AUTH=password`` (scrypt from the standard library)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return "scrypt${}${}".format(*(base64.b64encode(part).decode() for part in (salt, digest)))


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_text, digest_text = stored.split("$")
        salt, expected = base64.b64decode(salt_text), base64.b64decode(digest_text)
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    actual = hashlib.scrypt(password.encode(), salt=salt, dklen=len(expected), **SCRYPT)
    return hmac.compare_digest(actual, expected)
