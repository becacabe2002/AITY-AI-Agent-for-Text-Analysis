# Database schema

PostgreSQL holds four tables: `users`, `sessions`, `documents` and `analysis_results`. The table definitions in `db/tables.py` are the source of truth; migrations are drafted from them. See [database migrations](database-migrations.md).

## Ownership

Data is separated per user in two ways:

- every repository query filters by the user;
- foreign keys are composite, so a document cannot be stored in another user's session, and an analysis cannot point at a document of a different session or user.

## Deleting

Deleting cascades: a session removes its documents and analyses, and a document removes its analyses. Text files on disk are removed by the service after the commit.

## Stored blobs

Payload, token usage and sustainability are stored as JSON and read whole. Changes inside them need no table change, but a new top-level field on a domain model must be mirrored in the table definition by hand.

## Still to come

API keys and prompt variants (week 4), comparisons and syntheses (week 8).
