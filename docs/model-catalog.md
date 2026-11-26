# Model catalogue

The catalogue (`config/models.json`) lists which models AITY offers for each provider. It is written by hand.

- **Cloud entries** are maintained in the repository.
- **AITY Built-in entries** are controlled by the operator. The operator can point `AITY_MODELS_JSON` at their own copy of the file.
- Each provider has exactly one default model, and the default must be selectable.
- A model marked not selectable is hidden from new sessions but still resolves, so old results keep working.
- Each model can carry default options and a prompt profile. See [prompt variants](prompt-variants.md).

## One Built-in model at a time

AITY Built-in runs a single model on the shared server. The operator picks it with `AITY_OLLAMA_MODEL` (default: the catalogue's default). Other Built-in models stay viewable but cannot be run. Changing the model means restarting the service.

## Loading

The catalogue is loaded and validated at startup, so a broken file fails early rather than at the first analysis. To see what providers currently offer, see [model list fetching](model-list-fetching.md).
