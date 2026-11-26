# API models

Request and response bodies for the web API are separate from the stored domain models.

## Endpoints covered

| Action | Request | Response |
|---|---|---|
| Create session | name, optional description | the session |
| Update session | any of name and description | the session |
| Delete session | none | no content |
| Upload document | multipart file | the document |
| Start analysis | provider and model | accepted (runs as a job) |
| Read analysis | none | the analysis |

## Principles

- Request bodies refuse unknown fields, so a client cannot sneak in extras such as a prompt variant.
- Responses never include internal fields: user ids, stored text paths and hashes, the cache key and server-side sustainability warnings.
- A cache-hit flag appears in responses only; it is never stored.
- In a session update, an omitted field stays unchanged, a null or blank description clears it, and the name cannot be null.

See [domain models](domain-models.md).
