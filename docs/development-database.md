# Development database

The web service stores users, sessions, documents and analysis results in PostgreSQL 16.

## Set up

1. Set `AITY_DATABASE_URL` in your shell. Values for the development compose file are in `.env.example`. The app and Alembic read the environment, not `.env`. Tests default to the compose `aity_test` database, so `AITY_TEST_DATABASE_URL` is only needed for another database.
2. Start PostgreSQL (needs Docker Desktop on Windows or Docker Engine; the first start pulls `postgres:16-alpine`):

   ```
   docker compose -f compose.dev.yml up -d
   ```

3. Create or update the schema from the repo root:

   ```
   alembic upgrade head
   ```

   or `python -m aity_web.db.migrate` with `src` on the path.

## Database tests

They are opt-in and empty the `aity_test` database.

```
# bash
AITY_DB_TESTS=1 python -m pytest tests/aity_web/test_repository_contract.py tests/db -v
# PowerShell
$env:AITY_DB_TESTS = "1"; python -m pytest tests/aity_web/test_repository_contract.py tests/db -v
```

See also [database layer](database-layer.md) and [database migrations](database-migrations.md).
