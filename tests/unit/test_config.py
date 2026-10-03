import pytest

from app.core.config import Settings

DB_ENV_VARS = (
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_NAME",
    "DATABASE_USER",
    "DATABASE_PASSWORD",
    "DATABASE_URL",
    "APP_ENV",
    "APP_PORT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_load_without_env_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "icd-rag-service"
    assert settings.app_env == "local"
    assert settings.database_port == 5432


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_HOST", "db.internal")
    monkeypatch.setenv("DATABASE_PORT", "6543")
    monkeypatch.setenv("DATABASE_NAME", "icd")
    monkeypatch.setenv("DATABASE_USER", "svc")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss/word")
    monkeypatch.setenv("APP_ENV", "staging")

    settings = Settings(_env_file=None)
    url = settings.sqlalchemy_url()

    assert settings.app_env == "staging"
    assert url.drivername == "postgresql+asyncpg"
    assert (url.host, url.port, url.database, url.username) == ("db.internal", 6543, "icd", "svc")
    # Special characters survive without manual escaping.
    assert url.password == "p@ss/word"


def test_password_is_never_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_PASSWORD", "super-secret")

    settings = Settings(_env_file=None)

    assert "super-secret" not in repr(settings)
    assert "super-secret" not in str(settings.sqlalchemy_url())


def test_database_url_takes_precedence_and_forces_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_HOST", "ignored")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@url-host:5555/urldb")

    url = Settings(_env_file=None).sqlalchemy_url()

    assert url.drivername == "postgresql+asyncpg"
    assert (url.host, url.port, url.database) == ("url-host", 5555, "urldb")


def test_database_name_override() -> None:
    assert Settings(_env_file=None).sqlalchemy_url("other_db").database == "other_db"


def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PORT", "70000")

    with pytest.raises(ValueError):
        Settings(_env_file=None)
