# Analysis flow

Analysing a stored document happens in three steps. No database transaction is open during the model call, which can take minutes on AITY Built-in.

1. **Read.** Load the document, check the provider and model against the registry, read and clean the text, compute the cache key and fetch the user's API key if the provider needs one.
2. **Call.** Send the document to the provider with the prompt variant chosen by the operator.
3. **Write.** Turn the outcome into a stored result (failed ones included), update the session's last-changed time and commit.

Principles:

- Configuration mistakes (unknown provider or model, missing key, empty text) fail before any network call.
- Problems after the call starts never raise to the caller; they become a stored result with status `failed` and a short, sanitised error. Raw model output and exception text go to the server log only.
- Sustainability estimation never fails an analysis.
- If the document is deleted while the call runs, the result is dropped and the caller gets an error.
- Requests never choose a prompt variant.

Related notes: [analysis results](analysis-results.md), [cache key](cache-key.md), [provider adapters](llm-provider-adapters.md), [provider errors](provider-errors.md), [sustainability estimation](sustainability-estimation.md).
