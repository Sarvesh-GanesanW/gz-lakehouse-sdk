"""Tests for :class:`gz_lakehouse.LakehouseConfig`."""

import pytest

from gz_lakehouse import ConfigurationError, LakehouseConfig


def test_derives_site_from_url() -> None:
    """``dev-admin-icebergprovider`` host yields site ``admin``."""
    config = LakehouseConfig(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
    )
    assert config.derived_site == "admin"


def test_explicit_site_overrides_url() -> None:
    """Passing ``site`` explicitly wins over URL derivation."""
    config = LakehouseConfig(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        warehouse="wh",
        database="db",
        username="alice",
        password="secret",
        site="custom-site",
    )
    assert config.derived_site == "custom-site"


def test_missing_required_field_raises() -> None:
    """Empty username should raise :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://a-b-c.dev.example.com",
            warehouse="wh",
            database="db",
            username="",
            password="secret",
        )


def test_unparseable_host_raises() -> None:
    """A host with fewer than two hyphen parts cannot derive a site."""
    with pytest.raises(ConfigurationError):
        LakehouseConfig(
            lakehouse_url="http://only.example.com",
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

    config = LakehouseConfig.from_env()
    assert config.warehouse == "wh"
    assert config.username == "alice"


def test_from_env_overrides_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyword overrides win over environment values."""
    monkeypatch.setenv("GZ_LAKEHOUSE_URL", "http://a-b-c.dev.example.com")
    monkeypatch.setenv("GZ_LAKEHOUSE_WAREHOUSE", "wh-env")
    monkeypatch.setenv("GZ_LAKEHOUSE_DATABASE", "db")
    monkeypatch.setenv("GZ_LAKEHOUSE_USERNAME", "alice")
    monkeypatch.setenv("GZ_LAKEHOUSE_PASSWORD", "secret")

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
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError):
        LakehouseConfig.from_env()
