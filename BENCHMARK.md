# Neuron Benchmark MVP

This project compares simple spiking neuron implementations inside the same
small PyTorch classifier. The first MVP is intentionally dependency-light and
uses a deterministic synthetic two-class dataset, so it can run without a
dataset download.

## Models

- `if`: integrate-and-fire with hard reset
- `lif`: leaky integrate-and-fire
- `adaptive-lif`: LIF with a decaying post-spike threshold boost

All models receive the same encoded currents and run for the same number of
simulation steps. The CLI reports accuracy, wall-clock inference time, and
spike rate as JSON.

## Run

```bash
python -m pip install -e .
python -m neuron_benchmark --neuron lif
python -m neuron_benchmark --neuron adaptive-lif --samples 512 --steps 20 --seed 7
```

Run the test suite with:

```bash
python -m pytest
```

The current accuracy is an untrained smoke-test metric, not a claim about
model quality. The next experiment should add a shared training loop and then
run multiple seeds per neuron before drawing conclusions.

## Result schema

Each invocation emits one JSON object with:

- `neuron`
- `samples`
- `steps`
- `accuracy`
- `runtime_seconds`
- `spike_rate`

This provides a stable starting point for collecting CSV/JSONL experiment
results and plotting accuracy versus simulation cost.
