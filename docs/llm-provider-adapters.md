# LLM provider adapters

AITY talks to language models through one small contract, so the rest of the code does not depend on any provider SDK.

## Providers

Four adapters exist: Gemini, OpenAI and Anthropic (cloud, with the user's own key) and Ollama (AITY Built-in, no key).

## The contract

- One non-streaming call that takes messages, a system prompt and an optional response schema, and returns a response object.
- The response carries the raw text, the parsed structured output, token usage, a normalised finish reason and, for cloud providers, the environmental impact data.
- Adapters keep no user state. The API key arrives with each call and is never stored.
- Unknown options are rejected rather than ignored.

## Structured output

Each adapter uses its provider's native way to enforce the JSON schema. AITY Built-in can enforce it by grammar or, as a fallback, by adding the schema to the prompt.

## Token counts

Output tokens always include reasoning tokens, the same way providers bill them. Counts a provider does not report stay empty rather than guessed.

## Errors

Adapters let SDK errors through. They are converted in one place; see [provider errors](provider-errors.md).

Related: [model catalogue](model-catalog.md), [prompt variants](prompt-variants.md).
