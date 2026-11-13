"""Who is calling: API keys per client, and the end user a service client acts for.

The model. The server is not called by end users directly but by client programs, each with its own secret key
(settings.api_keys, sent as "Authorization: Bearer <key>"):
- an admin client (settings.admin_clients, e.g. the web UI) sees and cancels every job;
- a service client (e.g. the Telegram bot) acts for an end user whose id it puts in the X-User-Id header, and sees
  only that user's jobs. The server trusts the header because it trusts the key: only the bot holds it, and
  Telegram users never talk to this API themselves.

A job's owner is "<client>:<user>" ("telegram:123456789", "web:admin"): ids from different clients never
collide, and the owner is fixed when the job is created.

With no keys configured, access control is off: every request is an admin ("local:<X-User-Id or admin>").
That keeps the single-user local setup working without configuration; the server listens on 127.0.0.1 by
default. Do not expose it like that."""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass

from .config import Settings

_USER_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


class AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Caller:
    client: str
    user: str
    is_admin: bool

    @property
    def owner(self) -> str:
        return f"{self.client}:{self.user}"

    def can_see(self, owner: str) -> bool:
        return self.is_admin or owner == self.owner


def _user_id(value: str | None, default: str | None) -> str:
    if value is None or value == "":
        if default is None:
            raise AuthError(400, "the X-User-Id header is required for this client")
        return default
    if not _USER_ID.match(value):
        raise AuthError(400, "X-User-Id must be 1-64 characters: letters, digits, '_', '.', '@' or '-'")
    return value


def resolve_caller(settings: Settings, authorization: str | None, user_id: str | None) -> Caller:
    """Header values -> Caller. Raises AuthError (401 for a missing or wrong key, 400 for a bad user id)."""
    if not settings.api_keys:
        return Caller(client="local", user=_user_id(user_id, "admin"), is_admin=True)
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError(401, "an API key is required: 'Authorization: Bearer <key>'")
    token = token.strip().encode()
    client = None
    for name, secret in settings.api_keys.items():
        # compare_digest takes the same time wherever the strings differ, so the key cannot be guessed
        # character by character from response times. Every key is compared, for the same reason.
        if hmac.compare_digest(token, secret.encode()):
            client = name
    if client is None:
        raise AuthError(401, "unknown API key")
    is_admin = client in settings.admin_clients
    return Caller(client=client, user=_user_id(user_id, "admin" if is_admin else None), is_admin=is_admin)
