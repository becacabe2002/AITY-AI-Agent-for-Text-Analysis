#!/usr/bin/env python3
"""Generate a development master key and JWT secret as files (never for production)."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from collections.abc import Callable
from pathlib import Path

from cryptography.fernet import Fernet

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "var" / "secrets"

SECRETS: dict[str, tuple[str, Callable[[], bytes]]] = {
    "master.key": ("AITY_MASTER_KEY_FILE", Fernet.generate_key),
    "jwt_secret": ("AITY_JWT_SECRET_FILE", lambda: secrets.token_urlsafe(48).encode("ascii")),
}


def write_secret(path: Path, data: bytes, *, replace: bool) -> None:
    """Write one secret file readable by the owner only; refuse to replace unless asked."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data + b"\n")
    os.chmod(path, 0o600)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a fresh Fernet master key (master.key) and JWT secret (jwt_secret) "
            "for local development and tests. Production secrets come from the "
            "GitHub Actions production Environment, never from this script."
        ),
        epilog=(
            "Point the app at the files with AITY_MASTER_KEY_FILE and AITY_JWT_SECRET_FILE. "
            "Exit codes: 0 files written; 1 a file already exists or cannot be written; "
            "2 usage error."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for the two files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "replace existing files; API keys stored with the old master key "
            "become unreadable and every login token becomes invalid"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the files and print where they are; the secret values are never printed."""
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir.resolve()
    existing = [out_dir / name for name in SECRETS if (out_dir / name).exists()]
    if existing and not args.force:
        names = ", ".join(str(p) for p in existing)
        verb = "exists" if len(existing) == 1 else "exist"
        print(f"error: {names} already {verb}; use --force to replace", file=sys.stderr)
        return 1
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (_, make) in SECRETS.items():
            write_secret(out_dir / name, make(), replace=args.force)
    except OSError as exc:
        print(f"error: cannot write to {out_dir}: {exc.strerror or type(exc).__name__}", file=sys.stderr)
        return 1
    for name, (env, _) in SECRETS.items():
        print(f"{env}={out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
