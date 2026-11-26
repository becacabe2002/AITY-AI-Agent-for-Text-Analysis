# Database layer

Services never talk to SQL directly. They open a unit of work, call repository methods and get domain models back.

```
service (for example run_analysis)
   |  opens a unit of work, calls repositories, commits
   v
unit of work         one database session = one transaction
   |
   v
repositories         queries, per-user filtering, error mapping
   |
   v
table definitions    the schema (also the source for migrations)
   |
   v
engine               connection pool to PostgreSQL
```

| Part | Role |
|---|---|
| Engine | Reads the database address from `AITY_DATABASE_URL`, accepts PostgreSQL only, and hands out one session per request or job. |
| Table definitions | The tables with their keys, checks and indexes. Every schema change starts here. See [database schema](database-schema.md). |
| Unit of work | One transaction that gives access to all repositories. See [unit of work](unit-of-work.md). |
| Repositories | Interfaces plus PostgreSQL implementations, and in-memory fakes used only in tests. See [repository contract](repository-contract.md). |
| Migrations | Alembic environment and versions. See [database migrations](database-migrations.md). |

Because services only know the interfaces, tests pass the in-memory unit of work instead of the PostgreSQL one. A shared test suite runs against both to show they behave alike.

To run a local database, see [development database](development-database.md).
