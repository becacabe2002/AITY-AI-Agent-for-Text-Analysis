# Ollama estimation shim

The shim is a small service that runs inside the Ollama container. It only **measures**; it does not compute energy.

## What it measures

Between a start and a stop call it records the container's CPU time, the wall-clock duration, the CPU count it may use and the size of the loaded model.

## Why a shim

The backend turns the readings into energy and CO2e. Keeping the formula in the backend means a formula change never requires a new Ollama image. See [sustainability estimation](sustainability-estimation.md).

## Behaviour

- Reachable only on the internal Docker network.
- The analysis service starts a track before a Built-in model call and always stops it afterwards, outside the latency timer.
- Tracks never stopped are forgotten after an hour.
- If the shim is unreachable, the estimate becomes `unavailable`; the analysis itself is unaffected.
- CPU time is attributed to a single call only because AITY Built-in runs one job at a time.
