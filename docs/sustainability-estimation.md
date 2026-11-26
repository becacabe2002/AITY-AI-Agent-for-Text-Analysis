# Sustainability estimation

Every analysis stores an estimate of its energy use and CO2e. All values are estimates; the record says how each one was made.

## Methods

| Source | Used for | Based on |
|---|---|---|
| EcoLogits | Cloud providers | Model class and token counts |
| AITY CPU model | AITY Built-in | CPU time of the Ollama container, measured by the [estimation shim](ollama-shim.md) |
| none | When no estimate is possible | status `unavailable` |

## Status

- `ok`: the primary method worked without degradation.
- `partial`: numbers exist but the source flagged a degradation (for example the CPU counter was unreadable and wall-clock time was used as an upper bound).
- `unavailable`: no numbers; the reason is recorded.

## Principles

- The collector never fails an analysis; any problem becomes `unavailable`.
- Latency covers only the model call itself.
- The AITY formula and its constants live in the backend (`config/sustainability_constants.toml`), so changing them never needs a new Ollama image. The file can be replaced with `AITY_SUSTAINABILITY_CONSTANTS`.
- Warnings stay on the server and are not shown through the API.
- EcoLogits is initialised once per process.
