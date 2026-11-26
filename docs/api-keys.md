# API keys (bring your own key)

Users supply their own keys for Gemini, OpenAI and Anthropic. AITY Built-in needs none and is refused here.

## Storage

- A key is stored encrypted, together with the provider it belongs to. The provider is checked on every read, so a ciphertext copied onto another provider's row is rejected.
- Encryption is pluggable. A fake, reversible encoding is used in tests and development only. Production uses a real cipher whose master key is supplied as a secret.
- Keys are validated: printable characters only, no spaces, limited length.

## Handling rules

- Plaintext keys exist only inside the service and in the caller's local variable, which passes them straight to the adapter for one call.
- Keys are never logged, never put in error messages and never returned by the API; listing shows only which providers have a key and when it was saved.
- If a stored key cannot be decrypted, the user has to enter it again.
