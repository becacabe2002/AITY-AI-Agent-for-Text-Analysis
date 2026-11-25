"""Opt-in database tests: skipped unless AITY_DB_TESTS=1.

Not in pytest.ini's testpaths, so a plain `pytest` never needs PostgreSQL.
Run (with the compose.dev.yml database up):

    AITY_DB_TESTS=1 python -m pytest tests/aity_web/test_repository_contract.py tests/db -v

AITY_TEST_DATABASE_URL (default: the compose test database) must name a
database containing "test": these tests drop and recreate the schema.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url

DEFAULT_TEST_DATABASE_URL = "postgresql://aity:aity-dev@127.0.0.1:5432/aity_test"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("AITY_DB_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="database tests are opt-in: set AITY_DB_TESTS=1")
    for item in items:
        if "tests/db/" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def db_url() -> str:
    from aity_web.db import database_url

    url = database_url(
        os.environ.get("AITY_TEST_DATABASE_URL") or DEFAULT_TEST_DATABASE_URL,
        env="AITY_TEST_DATABASE_URL",
    )
    if "test" not in (make_url(url).database or ""):
        raise RuntimeError("AITY_TEST_DATABASE_URL must name a database containing 'test'")
    return url
