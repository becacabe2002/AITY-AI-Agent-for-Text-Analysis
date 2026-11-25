"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Drafted with `alembic revision --autogenerate --rev-id 000N` (next number in
sequence), then reviewed. Before committing:
- The dev database was at head before autogenerating (`alembic upgrade head`);
  otherwise the draft is wrong or Alembic refuses.
- An empty draft (only `pass`) means tables.py did not change: delete this file.
- CHECK constraints on EXISTING tables are not detected: add/drop them by hand
  (op.create_check_constraint / op.drop_constraint).
- A rename shows up as drop + add, which loses data: replace it with
  op.alter_column(..., new_column_name=...) / op.rename_table.
- Data changes (filling a new NOT NULL column for existing rows) are never
  generated: add them with op.execute().
- Check every constraint and index name, and that downgrade() undoes upgrade().
- Run: AITY_DB_TESTS=1 python -m pytest tests/db  (up/down + no drift).
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
