# About the Project

## What Is This?

A hackathon research project that compares five different spiking neuron models — the mathematical engines that drive spiking neural networks (SNNs). The project builds a controlled laboratory for neuroscience: put each neuron model into the same network, give it the same data, and measure the results.

## What Are Spiking Neurons?

Biological neurons communicate through electrical pulses called **spikes**. Conventional neural networks use continuous activation values; spiking neural networks use discrete spike events, making them closer to how real brains compute.

Spiking neurons are categorized by their mathematical complexity:

- **IF (Integrate-and-Fire):** accumulator + threshold. Simplest model.
- **LIF (Leaky Integrate-and-Fire):** adds a decay term — the membrane slowly returns to rest between inputs.
- **Adaptive LIF:** threshold rises after each spike, modeling neural fatigue.
- **Izhikevich:** combines biologically observed firing patterns (bursting, chattering) at low computational cost.
- **Hodgkin-Huxley:** the biophysical gold standard — models individual ion channels (Na+, K+). Slower but most realistic.

## What Is PyTorch?

PyTorch is the underlying machine learning framework. All five neuron models are implemented as PyTorch `nn.Module` classes, enabling automatic differentiation and GPU acceleration.

## What Is Streamlit?

**Streamlit** is a Python web framework built specifically for machine learning and data science. It lets you build interactive web apps by writing standard Python — no HTML, CSS, or JavaScript required.

Instead of writing:

```html
<form><input id="slider"><button>Submit</button></form>
```

You write:

```python
neuron = st.selectbox("Neuron Model", ["IF", "LIF", "Adaptive LIF", "Izhikevich", "HH"])
steps = st.slider("Simulation Steps", 4, 24, 12)
```

Streamlit handles the browser rendering, state management, and user interaction. The result is a shareable interactive demo that judges and collaborators can explore live.

Our demo lets you:

- Select any neuron model
- Choose a dataset (synthetic, MNIST, Fashion-MNIST)
- Adjust noise level, simulation steps, and training epochs
- See before/after training metrics, spike rasters, membrane traces, and prediction distributions
- Download results as JSON

## Project Structure

```
src/neuron_benchmark/
  neurons.py      -- All five neuron models (PyTorch)
  model.py        -- Shared SNN architecture
  data.py         -- Synthetic + MNIST + Fashion-MNIST
  training.py     -- Shared training loop with multi-seed support
  robustness.py   -- Noise, timestep, and data-size experiments
  metrics.py      -- Energy proxy and efficiency scoring
  plots.py        -- Visualizations
  demo.py         -- Streamlit interactive demo
  cli.py          -- Command-line benchmark interface
```

## Why Does This Matter?

The choice of neuron model involves a fundamental tradeoff:

- **Simpler models** (IF, LIF) are fast and cheap to simulate, but may not capture the temporal dynamics of real neurons.
- **Richer models** (Izhikevich, Hodgkin-Huxley) produce more realistic spike patterns, but are slower to simulate and harder to train.

For a given task — say, classifying MNIST digits — does the biologically richer model actually perform better? By how much? At what computational cost?

This benchmark provides **controlled, reproducible answers** to those questions.

## Key Results So Far

| Neuron | Test Accuracy | Training Time | Spike Rate |
|--------|-------------|--------------|------------|
| IF | ~100% | ~0.05s | ~0.12 |
| LIF | ~100% | ~0.06s | ~0.10 |
| Adaptive LIF | ~96% | ~0.07s | ~0.12 |
| Izhikevich | ~96% | ~0.12s | ~0.18 |
| Hodgkin-Huxley | ~65% | ~0.40s | ~0.08 |

The simple models converge faster on this task, while Hodgkin-Huxley requires significantly more compute for worse accuracy — a key finding for anyone building real SNN systems.

## Running the Demo

```bash
# Install dependencies
pip install -e .

# Launch the interactive Streamlit demo
streamlit run src/neuron_benchmark/demo.py
```

The demo runs locally in your browser. No server or internet connection required after installation.
