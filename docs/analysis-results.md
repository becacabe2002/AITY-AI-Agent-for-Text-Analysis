# Analysis results

A stored analysis result has two layers:

- the **payload**: what the model generated (summary, keywords, topics);
- everything the service adds: provider and model used, prompt variant, token usage, sustainability estimate, cache key and audit timestamps.

## Status

| Status | Meaning | Payload | Error text |
|---|---|---|---|
| `ok` | Valid payload, no problems | present | none |
| `partial` | Valid payload with semantic shortfalls (for example too few keywords) | present | the shortfalls |
| `failed` | No usable payload | none | short, sanitised reason |

Problems that make a result `partial` never trigger a retry.

## Payload clean-up

Before storing, text is stripped and keywords are lowercased and deduplicated. Topics are kept as written, because merging free-text labels would be guesswork.

## What the API shows

The API hides internal fields (user id, cache key, server-side warnings). See [API models](api-models.md).
