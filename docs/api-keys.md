# API keys (bring your own key)

Users supply their own keys for Gemini, OpenAI and Anthropic. AITY Built-in needs none and is refused here.

## Storage

- A key is stored encrypted, together with the provider it belongs to. The provider is checked on every read, so a ciphertext copied onto another provider's row is rejected.
- Each key is stored as one Fernet token that encrypts a small JSON payload: the key, its provider, and optionally a base URL and extra settings. Fernet authenticates the token, so a tampered or foreign ciphertext is rejected rather than misread.
- One master key encrypts every user's payload (flat Fernet). It is loaded from a file at startup; see [secrets](secrets.md). Per-user data keys and key rotation are future work.
- The cipher sits behind a small interface, so tests can plug in another one; the service itself never falls back to a fake.
- Keys are validated: printable characters only, no spaces, limited length.

## Handling rules

- Plaintext keys exist only inside the service and in the caller's local variable, which passes them straight to the adapter for one call.
- Keys are never logged, never put in error messages and never returned by the API; listing shows only which providers have a key and when it was saved.
- If a stored key cannot be decrypted, the user has to enter it again.
