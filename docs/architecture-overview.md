# Architecture overview

The web service lives in `src/aity_web` and is built in layers. Each layer only talks to the one below it.

| Layer | Purpose | Notes |
|---|---|---|
| `services` | Business logic: sessions, documents, analyses, API keys, sustainability | [analysis flow](analysis-flow.md) |
| `repositories` | Interfaces for reading and writing stored data, plus in-memory fakes for tests | [repository contract](repository-contract.md) |
| `db` | PostgreSQL engine, table definitions, unit of work, migrations | [database schema](database-schema.md) |
| `llm` | Provider-independent access to language models | [provider adapters](llm-provider-adapters.md) |
| `models` | Shared data shapes (users, sessions, documents, results) | [domain models](domain-models.md) |

Two ideas shape the design:

- Services never talk to SQL or to a model SDK directly. They use repository interfaces and the provider registry, so storage and providers can be swapped or faked in tests.
- The desktop app (`aity_desktop`) is separate and is never imported by the web service.
