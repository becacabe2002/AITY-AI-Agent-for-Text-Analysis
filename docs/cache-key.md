# Cache key

Every analysis result stores a content hash that identifies what was asked: the cleaned document text, the provider, the model and the prompt variant. A per-user lookup that reuses earlier results is planned for week 6.

## Text normalisation

Before hashing and before sending to a model, text is normalised: Unicode normalisation, consistent line endings, trailing spaces removed and runs of blank lines collapsed.

## Key version

A version number is stored next to the hash. Lookups only match the current version. Bump it, in the same change, when something that decides what a model is asked changes:

- text normalisation or the hash formula;
- the shipped model catalogue;
- the provider-facing payload schema;
- adapter defaults or rules;
- model behaviour changed by the operator outside the repository (bump by hand).

A pinned test fails on the first three until its pins are updated. A prompt text change needs no bump, because new text ships under a new variant name that is part of the hash. See [prompt variants](prompt-variants.md).
