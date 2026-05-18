"""Tests for :class:`gz_lakehouse.LakehouseConfig`."""

import pytest

from gz_lakehouse import ConfigurationError, LakehouseConfig


def test_explicit_site_sets_header_value() -> None:
    """``site`` is the value sent as the ``gz-site`` header."""
    config = LakehouseConfig(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
        site_name="admin",
    )
    assert config.site_header == "admin"


def test_site_is_not_derived_from_url() -> None:
    """The tenant route must be supplied explicitly."""
    config = LakehouseConfig(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
        site_name="custom-site",
    )
    assert config.site_header == "custom-site"


def test_missing_required_field_raises() -> None:
    """Empty username should raise :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://a-b-c.dev.example.com",
            warehouse="wh",
            database="db",
            username="",
            password="secret",
            site_name="admin",
        )


def test_missing_site_raises() -> None:
    """A missing tenant route surfaces as :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://dev-admin-icebergprovider.dev.example.com",
            warehouse="wh",
            database="db",
            username="alice",
            password="secret",
        )


def test_compute_size_default_is_small() -> None:
    """``compute_size`` defaults to ``"small"`` when unspecified."""
    config = LakehouseConfig(
        lakehouse_url="http://a-b-c.dev.example.com",
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
        site_name="admin",
    )
    assert config.compute_size == "small"
    assert config.compute_id is None


def test_invalid_compute_size_raises() -> None:
    """An unknown t-shirt name surfaces as :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://a-b-c.dev.example.com",
            warehouse="wh",
            database="db",
            username="alice",
            password="secret",
            site_name="admin",
            compute_size="enormous",
        )


def test_compute_id_escape_hatch_accepted() -> None:
    """An explicit ``compute_id`` overrides the t-shirt size."""
    config = LakehouseConfig(
        lakehouse_url="http://a-b-c.dev.example.com",
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
        site_name="admin",
        compute_id=1009,
    )
    assert config.compute_id == 1009


def test_perf_knob_validation() -> None:
    """Negative or zero perf knobs should raise :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://a-b-c.dev.example.com",
            warehouse="wh",
            database="db",
            username="alice",
            password="secret",
            site_name="admin",
            parallel_workers=0,
        )


def test_repr_redacts_password() -> None:
    """``repr`` must never expose the password."""
    config = LakehouseConfig(
        lakehouse_url="http://a-b-c.dev.example.com",
        warehouse="wh",
        database="db",
        username="alice",
        password="super-secret",
        site_name="admin",
    )
    assert "super-secret" not in repr(config)
    assert "***" in repr(config)


def test_from_env_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``from_env`` loads required fields from environment variables."""
    monkeypatch.setenv("GZ_LAKEHOUSE_URL", "http://a-b-c.dev.example.com")
    monkeypatch.setenv("GZ_LAKEHOUSE_WAREHOUSE", "wh")
    monkeypatch.setenv("GZ_LAKEHOUSE_DATABASE", "db")
    monkeypatch.setenv("GZ_LAKEHOUSE_USERNAME", "alice")
    monkeypatch.setenv("GZ_LAKEHOUSE_PASSWORD", "secret")
    monkeypatch.setenv("GZ_LAKEHOUSE_SITE_NAME", "admin")

    config = LakehouseConfig.from_env()
    assert config.warehouse == "wh"
    assert config.username == "alice"
    assert config.site_name == "admin"


def test_from_env_overrides_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyword overrides win over environment values."""
    monkeypatch.setenv("GZ_LAKEHOUSE_URL", "http://a-b-c.dev.example.com")
    monkeypatch.setenv("GZ_LAKEHOUSE_WAREHOUSE", "wh-env")
    monkeypatch.setenv("GZ_LAKEHOUSE_DATABASE", "db")
    monkeypatch.setenv("GZ_LAKEHOUSE_USERNAME", "alice")
    monkeypatch.setenv("GZ_LAKEHOUSE_PASSWORD", "secret")
    monkeypatch.setenv("GZ_LAKEHOUSE_SITE_NAME", "admin")

    config = LakehouseConfig.from_env(warehouse="wh-override")
    assert config.warehouse == "wh-override"


def test_from_env_missing_field_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing required env var surfaces as :class:`ConfigurationError`."""
    for name in (
        "GZ_LAKEHOUSE_URL",
        "GZ_LAKEHOUSE_WAREHOUSE",
        "GZ_LAKEHOUSE_DATABASE",
        "GZ_LAKEHOUSE_USERNAME",
        "GZ_LAKEHOUSE_PASSWORD",
        "GZ_LAKEHOUSE_SITE",
        "GZ_LAKEHOUSE_SITE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError):
        LakehouseConfig.from_env()
