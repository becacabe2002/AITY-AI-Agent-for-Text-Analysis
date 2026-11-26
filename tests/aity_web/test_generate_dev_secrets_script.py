"""scripts/generate_dev_secrets.py writes files the startup loaders accept."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

from aity_web.services.crypto import FernetCipher, load_secrets

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_dev_secrets.py"
_spec = importlib.util.spec_from_file_location("generate_dev_secrets_script", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


def test_writes_secrets_the_loaders_accept(tmp_path, capsys):
    out = tmp_path / "secrets"
    assert gen.main(["--out-dir", str(out)]) == 0
    secrets = load_secrets(out / "master.key", out / "jwt_secret")
    FernetCipher(secrets.master_key)
    printed = capsys.readouterr().out
    assert f"AITY_MASTER_KEY_FILE={out / 'master.key'}" in printed
    assert f"AITY_JWT_SECRET_FILE={out / 'jwt_secret'}" in printed
    assert secrets.master_key.decode() not in printed and secrets.jwt_secret.decode() not in printed


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_files_are_owner_only(tmp_path):
    assert gen.main(["--out-dir", str(tmp_path)]) == 0
    for name in ("master.key", "jwt_secret"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_refuses_to_overwrite_without_force(tmp_path, capsys):
    assert gen.main(["--out-dir", str(tmp_path)]) == 0
    before = (tmp_path / "master.key").read_bytes()
    assert gen.main(["--out-dir", str(tmp_path)]) == 1
    assert "--force" in capsys.readouterr().err
    assert (tmp_path / "master.key").read_bytes() == before


def test_force_replaces_both(tmp_path):
    assert gen.main(["--out-dir", str(tmp_path)]) == 0
    before = {n: (tmp_path / n).read_bytes() for n in ("master.key", "jwt_secret")}
    assert gen.main(["--out-dir", str(tmp_path), "--force"]) == 0
    assert all((tmp_path / n).read_bytes() != b for n, b in before.items())
    load_secrets(tmp_path / "master.key", tmp_path / "jwt_secret")
