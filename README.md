# AITY
AITY stands for **AI Agent for Text Analysis**.
It is an R&D Project directed and produced by Dorina Rajanen and Mikko Rajanen at University of Oulu, INTERACT Research Group, UX Lab.

## Contributors:
Sultan Alshehry, Ngbede Linus, Timo Pihlava, Elina Ronkainen, Vilma Santamäki, Jessica Sivonen (2026 Bachelor’s Project course)

## Project Description

AITY – AI Agent for text analysis. AITY is a concept or project of automatic text analysis created by Dorina Rajanen. AITY builds on the ability of an AI agent that analyses a text document (research article or new article) and identifies keywords, themes, topics. AITY then extracts these categories and stores them in a database. AITY also compares different documents and creates different analyses on the documents. Course project assignment includes the following tasks (selection of these can be discussed with the team):
-	Design, prototype and implement an AI agent for text analysis together with the documentation and literature review
-	Choose the AI platform suitable for the task and document the choice made
-	Document the ethical aspects of using research articles or news articles by the AI agent based on literature and available sources of the articles (e.g., news sites or scholarly databases or forums)
-	Document and calculate sustainability indicators such as carbon footprint for each document analysed
-	Design and implement a simple user interface of the agent
-	Document implementation options local versus public agent
-	Experimental results on a given dataset

## Repository layout

```
src/
├── aity_desktop/   # the original Tkinter desktop app (Gemini + BERTs modes) — frozen, bug fixes only
└── aity_web/       # the new multi-user web service (FastAPI + Vue 3) — in development
web_ui/             # web service backend (FastAPI) and frontend (Vue 3) — in development
tests/
```

`aity_desktop` is kept fully runnable but frozen until a further decision. The web service is built separately in `aity_web` and never imports `aity_desktop`. See `src/aity_desktop/README.md`.

## How to use (desktop app)

1. Download the repository:

```
git clone https://github.com/UXLAB-OULU/AITY-AI-Agent-for-Text-Analysis-
```


2. Enter the source directory:

```
cd AITY-AI-Agent-for-text-analysis-
cd src
```


3. Create a python virtual environment

Windows:

```
python -m venv venv
```

MacOs/Linux:

```
python3 -m venv venv

```

4. Activate the virtual environment

Windows: 

```
venv\Scripts\activate.bat

```

MacOs/Linux:

```
source venv/bin/activate
```

5. Download dependencies

```
pip install -r aity_desktop/requirements.txt

```

6. Run the program

```
python -m aity_desktop

```

7. Enjoy!

## Code layout: database layer (web service)

Services never talk to SQL directly. They open a unit of work, call repository methods, and get domain models (`aity_web.models`) back. Design decisions are in ADR-004.

```
service (e.g. run_analysis, later the W5 routes)
   │  async with uow_factory() as uow: ... await uow.commit()
   ▼
db/uow.py                 one AsyncSession = one transaction
   │  uow.users / uow.sessions / uow.documents / uow.analyses
   ▼
repositories/sqlalchemy/  SQL queries, user_id filtering, errors
   │  builds queries from
   ▼
db/tables.py              the schema (source of truth for Alembic too)
   │  executed through
   ▼
db/engine.py              connection pool to PostgreSQL (asyncpg)
```

| File | Role |
|---|---|
| `src/aity_web/db/engine.py` | `database_url()` reads `AITY_DATABASE_URL` and switches it to `postgresql+asyncpg://` (other databases are refused); `make_engine()` builds the connection pool once at startup; `make_session_factory()` gives one `AsyncSession` per request or job. |
| `src/aity_web/db/tables.py` | The four tables (`users`, `sessions`, `documents`, `analysis_results`) with columns, keys, CHECK rules and indexes. Every schema change starts here; Alembic autogenerate compares it with the database, and `tests/db` fails if the migrations and this file differ. Top-level fields of the Pydantic models must be mirrored here by hand. |
| `src/aity_web/db/uow.py` | `SqlAlchemyUnitOfWork`: wraps one `AsyncSession` and exposes the repositories. `commit()` saves; leaving the `async with` block without committing (or with an error) rolls back. Supplies the clock for timestamps. `uow.credentials` raises `NotImplementedError` until W4. `make_uow_factory()` returns the zero-argument factory services receive. |
| `src/aity_web/db/__init__.py` | Re-exports what other code needs (`make_engine`, `make_session_factory`, `make_uow_factory`, `database_url`, `metadata`, `SqlAlchemyUnitOfWork`). |
| `src/aity_web/db/migrations/`, `db/migrate.py` | Alembic environment and migrations; `python -m aity_web.db.migrate` runs them without `alembic.ini`. |
| `src/aity_web/repositories/base.py` | The repository interfaces (ABCs), errors and the `UnitOfWork` contract that both implementations follow. |
| `src/aity_web/repositories/sqlalchemy/` | PostgreSQL implementations, one file per repository. Every query filters by `user_id`, so another user's row looks like a missing one (`NotFoundError`); results are domain models, never database rows; inserts run in a savepoint so a duplicate becomes `ConflictError` without breaking the transaction. `analyses.py` converts the JSONB blobs (`payload`, `token_usage`, `sustainability`) to and from their Pydantic models; `_common.py` holds the shared base class and error mapping. |
| `src/aity_web/repositories/memory.py` | In-memory fakes of the same interfaces, used only in tests. |

Wiring (what the W5 app startup will do):

```python
from aity_web.db import make_engine, make_session_factory, make_uow_factory

engine = make_engine()                                   # AITY_DATABASE_URL
uow_factory = make_uow_factory(make_session_factory(engine))

async with uow_factory() as uow:
    session = await uow.sessions.add(user_id=user_id, name="SMS pilot")
    await uow.commit()
```

Tests pass `InMemoryStore().uow_factory` instead, so services cannot tell which one they use. `tests/aity_web/test_repository_contract.py` runs the same tests against both implementations (PostgreSQL opt-in with `AITY_DB_TESTS=1`), which proves they behave alike.

## Development database (web service)

The web service stores users, sessions, documents and analysis results in PostgreSQL 16 (ADR-004).

1. Set `AITY_DATABASE_URL` in your shell (values for the dev compose file are in `.env.example`; Alembic and the app read the environment, not `.env`). The tests default to the compose `aity_test` database, so `AITY_TEST_DATABASE_URL` is only needed for another database.
2. Start Postgres: `docker compose -f compose.dev.yml up -d`
3. Create or update the schema from the repo root: `alembic upgrade head` (or `python -m aity_web.db.migrate` with `src` on the path).
4. Database tests are opt-in and empty the `aity_test` database:
   ```
   # bash
   AITY_DB_TESTS=1 python -m pytest tests/aity_web/test_repository_contract.py tests/db -v
   # PowerShell
   $env:AITY_DB_TESTS = "1"; python -m pytest tests/aity_web/test_repository_contract.py tests/db -v
   ```

### Changing the schema

Requires Docker Desktop (Windows) or Docker Engine; `compose.dev.yml` pulls `postgres:16-alpine` on first start.

1. Edit `src/aity_web/db/tables.py`.
2. Bring the dev database to the current head: `alembic upgrade head`.
3. Draft the migration: `alembic revision --autogenerate -m "add user_api_keys" --rev-id 0002` (written to `src/aity_web/db/migrations/versions/0002_add_user_api_keys.py`; use the next number in sequence).
4. Review the draft with the checklist at the top of the file. Autogenerate misses CHECK changes on existing tables, turns renames into drop + add, and never writes data changes.
5. Apply it (`alembic upgrade head`) and run `AITY_DB_TESTS=1 python -m pytest tests/db`, which checks upgrade/downgrade and fails if the migrated database and `tables.py` differ.

Running `alembic revision --autogenerate` with no `tables.py` changes produces an empty migration (`pass`); delete it. Autogenerate compares `tables.py` with the database and does not read the Pydantic models, so a new top-level model field must be added to `tables.py` first.
