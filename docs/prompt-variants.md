# Prompt variants

A prompt variant is a named set of prompt texts used for one kind of task. Today only `default_analysis` is active.

## Rules

- **Defined in code.** Variants live in `src/aity_web/llm/prompts/`. There is no database table for them; analysis rows store only the variant name.
- **Names are immutable.** Once a name has produced results, its text never changes. A change ships under a new name, such as `default_analysis_v2`. A test pins a fingerprint of each variant to catch accidental edits.
- **Profiles.** A variant has a standard text and optional overrides. The `small` profile is shorter and more directive, for small local models. The profile comes from the model's catalogue entry, so the same variant and model always give the same text.
- **Placeholders.** Reserved names with no text (`extraction_v1`, `qa_v1`) keep future names from being reused.
- **Document placement.** The document is always in the user message, never in the system prompt, and is treated as data rather than instructions.
- **Parameters.** Word and item limits are template parameters with defaults.

See also [model catalogue](model-catalog.md) and [cache key](cache-key.md), where the variant name is part of the key.
