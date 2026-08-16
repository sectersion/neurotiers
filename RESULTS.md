# Results

This document serves as a template and collection point for benchmark results.

## Experiment: Baseline Comparison

| Neuron | Accuracy | Runtime (s) | Spike Rate | Seeds |
|--------|----------|-------------|------------|-------|
| IF | - | - | - | - |
| LIF | - | - | - | - |
| Adaptive LIF | - | - | - | - |
| Izhikevich | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - |

## Experiment: Training Comparison

| Neuron | Train Acc | Test Acc | Train Time (s) | Spike Rate | Epochs |
|--------|-----------|----------|----------------|------------|--------|
| IF | - | - | - | - | - |
| LIF | - | - | - | - | - |
| Adaptive LIF | - | - | - | - | - |
| Izhikevich | - | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - | - |

## Experiment: Robustness Analysis

Noise levels tested: 0.0, 0.01, 0.05, 0.1

### Accuracy vs Noise

| Neuron | Noise=0.0 | Noise=0.01 | Noise=0.05 | Noise=0.1 |
|--------|-----------|------------|------------|----------|
| IF | - | - | - | - |
| LIF | - | - | - | - |
| Adaptive LIF | - | - | - | - |
| Izhikevich | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - |

### Spike Rate vs Noise

| Neuron | Noise=0.0 | Noise=0.01 | Noise=0.05 | Noise=0.1 |
|--------|-----------|------------|------------|----------|
| IF | - | - | - | - |
| LIF | - | - | - | - |
| Adaptive LIF | - | - | - | - |
| Izhikevich | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - |

## Experiment: Simulation Step Sensitivity

Steps tested: 4, 8, 12, 16, 20

| Neuron | Steps=4 | Steps=8 | Steps=12 | Steps=16 | Steps=20 |
|--------|---------|---------|----------|----------|----------|
| IF | - | - | - | - | - |
| LIF | - | - | - | - | - |
| Adaptive LIF | - | - | - | - | - |
| Izhikevich | - | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - | - |

## Runtime Analysis

### Inference Time (ms per sample)

| Neuron | Mean | Std | Min | Max | Samples |
|--------|------|-----|-----|-----|---------|
| IF | - | - | - | - | - |
| LIF | - | - | - | - | - |
| Adaptive LIF | - | - | - | - | - |
| Izhikevich | - | - | - | - | - |
| Hodgkin-Huxley | - | - | - | - | - |

## Pareto Analysis

### Accuracy vs Runtime

Insert Pareto frontier plot showing the tradeoff between classification accuracy and computational cost across neuron models.

### Expected Pattern

- IF: Fastest, lowest accuracy
- LIF: Fast, moderate accuracy
- Adaptive LIF: Moderate speed, good accuracy
- Izhikevich: Slower, high accuracy potential
- Hodgkin-Huxley: Slowest, biological reference ceiling

## Statistical Significance

Results across N seeds with mean ± standard deviation.

```
N = 5 seeds per condition
Statistical test: t-test or ANOVA as appropriate
Significance threshold: p < 0.05
```

## Raw Data

Raw JSON outputs from experiment runs are stored in:

```
results/raw/
  ├── baseline/
  ├── training/
  ├── robustness/
  └── sensitivity/
```

## Analysis Notebooks

Jupyter notebooks for result analysis:

```
notebooks/
  ├── 01_visualize_baseline.ipynb
  ├── 02_robustness_analysis.ipynb
  ├── 03_pareto_frontier.ipynb
  └── 04_statistical_tests.ipynb
```
