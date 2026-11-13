"""Who is calling (auth.py) and the key settings (config.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from my_diffusion.auth import AuthError, resolve_caller
from my_diffusion.config import Settings

WEB_KEY = "w" * 32
BOT_KEY = "b" * 32


@pytest.fixture
def settings():
    return Settings(api_keys={"web": WEB_KEY, "telegram": BOT_KEY}, admin_clients=["web"])


def test_without_keys_everyone_is_a_local_admin():
    caller = resolve_caller(Settings(api_keys={}), None, None)
    assert caller.is_admin and caller.owner == "local:admin"
    assert resolve_caller(Settings(api_keys={}), None, "anna").owner == "local:anna"


def test_admin_client(settings):
    caller = resolve_caller(settings, f"Bearer {WEB_KEY}", None)
    assert caller.is_admin and caller.owner == "web:admin"
    assert caller.can_see("telegram:42")


def test_service_client_acts_for_its_user(settings):
    caller = resolve_caller(settings, f"bearer {BOT_KEY}", "123456789")
    assert not caller.is_admin and caller.owner == "telegram:123456789"
    assert caller.can_see("telegram:123456789")
    assert not caller.can_see("telegram:987")
    assert not caller.can_see("web:123456789")       # same id, another client: another person


@pytest.mark.parametrize("header", [None, "", WEB_KEY, "Basic abc", "Bearer ", f"Bearer {WEB_KEY}x", "Bearer wrong"])
def test_missing_or_wrong_key_is_401(settings, header):
    with pytest.raises(AuthError) as error:
        resolve_caller(settings, header, "1")
    assert error.value.status == 401


def test_service_client_must_name_the_user(settings):
    with pytest.raises(AuthError) as error:
        resolve_caller(settings, f"Bearer {BOT_KEY}", None)
    assert error.value.status == 400


@pytest.mark.parametrize("user", ["a b", "x" * 65, "../etc", "имя", "a:b"])
def test_user_id_format_is_checked(settings, user):
    with pytest.raises(AuthError) as error:
        resolve_caller(settings, f"Bearer {BOT_KEY}", user)
    assert error.value.status == 400


def test_short_or_shared_keys_are_refused():
    with pytest.raises(ValidationError, match="at least 16"):
        Settings(api_keys={"web": "short"})
    with pytest.raises(ValidationError, match="its own API key"):
        Settings(api_keys={"web": WEB_KEY, "bot": WEB_KEY})
