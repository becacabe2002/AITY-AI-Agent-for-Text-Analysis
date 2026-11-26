# Database migrations

Schema changes are managed with Alembic. The database address comes from the `AITY_DATABASE_URL` environment variable and is never stored in a config file. A local database is described in [development database](development-database.md).

## Changing the schema

1. Edit the table definitions first (`src/aity_web/db/tables.py`). Autogenerate compares them with the database and does not read the domain models, so a new top-level model field must be added there by hand.
2. Bring the development database to the current version:

   ```
   alembic upgrade head
   ```

3. Draft the migration, using the next number in sequence:

   ```
   alembic revision --autogenerate -m "add user_api_keys" --rev-id 0002
   ```

   The file is written under `src/aity_web/db/migrations/versions/`.
4. Review the draft with the checklist at the top of the file. Autogenerate misses constraint changes on existing tables, treats renames as drop and add, and never writes data changes.
5. Apply it and run the migration tests, which check upgrade, downgrade and that the migrated database matches the table definitions:

   ```
   alembic upgrade head
   AITY_DB_TESTS=1 python -m pytest tests/db
   ```

An autogenerate run with no table changes produces an empty migration; delete it.

## Running without alembic.ini

`python -m aity_web.db.migrate` runs the migrations from any folder (with `src` on the path). Pass `base` to go back to an empty database.

The first migration creates the four base tables; see [database schema](database-schema.md).
