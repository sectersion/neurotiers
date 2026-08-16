# Neuron Benchmark

**Thesis:** Spiking neural network efficiency depends critically on neuron model complexity, where simpler models offer speed advantages at the cost of biological realism, and the optimal choice depends on the specific tradeoff between task accuracy and computational budget.

---

## The Project Story

Spiking neural networks promise a more event-driven and potentially more efficient way to compute than conventional neural networks. Their behavior is determined in part by the simulated neuron inside each layer, but the tradeoff between simple neuron models and biologically richer models is not always obvious.

This project began as a hackathon research experiment: build a small, repeatable benchmark that makes those tradeoffs visible. Instead of comparing neuron models in isolation, we place each one in the same network, give it the same inputs, and measure both task behavior and simulation cost.

The central question:

> **How does neuron-model complexity affect temporal behavior, spike activity, and computational efficiency in a spiking neural network?**

---

## Architecture

```
Input (Rates)  ->  Encoder (Linear)  ->  Neuron (SNN Layer)  ->  Readout (Linear)  ->  Class
                        PyTorch / NumPy / Streamlit / MoviePy
```

- **Encoder:** single linear layer (features -> hidden)
- **Neuron:** one of 5 spiking neuron models
- **Readout:** linear layer (hidden -> classes)
- **Training:** Adam optimizer, cross-entropy loss, surrogate gradient spike function
- **Evaluation:** accuracy, spike rate, training time, energy proxy

---

## Quick Start

```bash
# Install
.\scripts\install.ps1          # pip install -e .

# Run demo
.\scripts\demo.ps1              # streamlit run --server.headless true src/neuron_benchmark/demo.py

# Smoke test
.\scripts\quick.ps1              # each neuron, 3 epochs, synthetic data

# Full multi-seed benchmark
.\scripts\benchmark.ps1          # 5 seeds, 20 epochs, all neurons

# Robustness analysis
.\scripts\robustness.ps1        # noise, timestep, reduced-data experiments

# MNIST comparison
.\scripts\mnist_benchmark.ps1   # multi-seed on real MNIST data

# Generate presentation video
python scripts/generate_video.py benchmark_demo.mp4

# Tests
.\scripts\test.ps1              # pytest
```

---

## Neuron Models

| Model | Description |
|-------|-------------|
| **IF** | Integrate-and-Fire. Simplest baseline. |
| **LIF** | Leaky Integrate-and-Fire. Decay leak + threshold. General-purpose SNN baseline. |
| **Adaptive LIF** | LIF with spike-frequency adaptation. Better temporal dynamics. |
| **Izhikevich** | Compact model with richer dynamics (bursting, regular-spiking). |
| **Hodgkin-Huxley** | Biophysical reference with ion-channel dynamics. Slower but most biologically realistic. |

---

## Decision Guide

| Neuron | Speed | Realism | Best For | Avoid When |
|--------|-------|---------|----------|------------|
| **IF** | *** | * | Edge devices, baselines | Temporal dynamics matter |
| **LIF** | ** | ** | General SNN tasks | High-firing regimes |
| **Adaptive LIF** | ** | ** | Sensory processing | Minimum latency required |
| **Izhikevich** | * | *** | Cognitive modeling | Real-time at scale |
| **Hodgkin-Huxley** | * | **** | Biological validation | Production inference |

---

## Datasets

- **Synthetic:** fast, no download, two-class separable data via `synthetic_data()`
- **MNIST:** 60k training + 10k test images via `load_mnist()`
- **Fashion-MNIST:** via `load_fashion_mnist()`

Pixel values are encoded as Poisson spike-train rates: `rate = pixel * scale` (clamped to [0,1]).

---

## Experiment Commands

```bash
# Single trained run
python -m neuron_benchmark --neuron lif --train --epochs 20

# Multi-seed experiments
python -m neuron_benchmark --experiment --neuron all --seeds 0 1 2 3 4

# Full report with plots
python -m neuron_benchmark --report

# Robustness
python -m neuron_benchmark --robustness --neuron lif

# Full metrics with energy proxy
python -m neuron_benchmark --experiment --neuron all --full-metrics
```

---

## Project Layout

```
src/neuron_benchmark/
  neurons.py      -- Neuron models (IF, LIF, Adaptive LIF, Izhikevich, Hodgkin-Huxley)
  model.py        -- Shared SNN classifier
  data.py         -- Datasets (synthetic, MNIST, Fashion-MNIST)
  training.py     -- Training loop and multi-seed experiment runner
  robustness.py   -- Noise, timestep, and data-size experiments
  metrics.py      -- Energy proxy and efficiency metrics
  plots.py        -- Visualizations (accuracy, runtime, spike rate, traces)
  demo.py         -- Streamlit interactive demo
  cli.py          -- Command-line benchmark interface
scripts/
  install.ps1       -- pip install -e .
  test.ps1          -- pytest
  quick.ps1         -- fast smoke test
  benchmark.ps1     -- full multi-seed benchmark
  robustness.ps1    -- robustness experiments
  mnist_benchmark.ps1 -- MNIST multi-seed benchmark
  demo.ps1          -- launch Streamlit demo
  generate_video.py -- render benchmark_demo.mp4
tests/
  test_benchmark.py -- Pipeline and shape smoke tests
```

---

## Key Findings

1. **LIF is the strongest general baseline** — good accuracy at low computational cost
2. **Izhikevich offers richer dynamics at moderate cost** — best accuracy-per-complexity tradeoff
3. **Hodgkin-Huxley is the gold-standard reference** — too slow for production but essential for biological validation
4. **Benchmark everything — never assume** — simpler models don't always win

---

## Setup Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Torchvision (for MNIST/Fashion-MNIST auto-download)
- Streamlit (for interactive demo)
- MoviePy (for video generation)
- matplotlib, pandas, numpy

---

## Citations

- **LIF / IF:** Gerstner et al., *Neuronal Dynamics* (2014)
- **Izhikevich:** Izhikevich, "Simple Model of Spiking Neurons" (2003)
- **Hodgkin-Huxley:** Hodgkin & Huxley, "A quantitative description of membrane current" (1952)
- **Surrogate Gradients:** Neftci et al., "Surrogate gradient learning in spiking neural networks" (2019)
- **Spiking Neural Networks overview:** Pfeiffer & Pfeil, "Deep Learning With Spiking Neural Networks" (2018)
