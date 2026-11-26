# Provider errors

Every failed provider call is reported as a single error type, whichever provider it came from. The kind tells the user what they can do.

| Kind | Meaning | Retry later? |
|---|---|---|
| `auth` | Key invalid, revoked or without permission | no |
| `rate_limit` | Too many requests, or no credit left | yes, unless credit is exhausted |
| `unavailable` | Provider down, overloaded, timed out or unreachable | yes |
| `bad_request` | Any other rejection, such as a bad model id | no |

## Principles

- The conversion happens in one place, the model handle that wraps every call. Other exceptions (programming or configuration mistakes) pass through unchanged.
- The user-facing message never contains SDK text, because provider messages can echo request details. The original exception stays chained in the server log.
- A retry hint from the provider is kept when sent.

See also [analysis flow](analysis-flow.md).
