# Unit of work

A unit of work is one database transaction that gives access to all repositories. Services receive a factory that creates a new one per operation.

- The transaction opens when the `async with` block is entered and closes when it ends.
- Nothing is saved unless the service calls `commit()`. Leaving the block without a commit, or with an error, rolls everything back.
- After a commit the same unit of work can start a new transaction.
- Timestamps set by repositories come from an injectable clock, so tests can control time.
- Tests pass an in-memory unit of work instead; services cannot tell the difference.

Services never import the database package directly; they only know the unit-of-work interface.
