# Repository contract

Repositories are the only way services read or write stored data. Every implementation (PostgreSQL and the in-memory test fakes) follows the same rules, and one shared test suite checks that they behave alike.

- **Per-user access.** Every call about a user-owned row takes the user and filters by it. A missing row and another user's row look the same (`NotFoundError`), so callers cannot probe for other users' data. Only the user repository looks users up.
- **Domain models only.** Repositories return domain models, never database rows.
- **No commits.** Repositories never commit; the service owns the transaction. See [unit of work](unit-of-work.md).
- **Who sets ids and times.** Users, sessions and credentials get their ids and timestamps from the repository. Documents and results arrive fully built.
- **Cascading deletes.** Deleting a session or document removes what belongs to it. Deleting a result detaches copies that were served from it. Files on disk are not touched here.
- **Errors.** `NotFoundError` for missing or foreign rows, `ConflictError` for duplicates or rows whose parent disappeared, `ValueError` for programming mistakes.
- **Partial updates.** A marker value means "leave unchanged"; `None` is a real value that clears a field.
