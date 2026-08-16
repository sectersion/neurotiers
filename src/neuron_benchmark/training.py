import statistics
import time

import torch
from torch import nn

from .data import synthetic_data, load_mnist, load_fashion_mnist
from .metrics import compute_all_metrics_for_experiment
from .model import SNNClassifier

NEURONS = ["if", "lif", "adaptive-lif", "izhikevich", "hodgkin-huxley"]


def train_model(neuron="lif", samples=256, steps=12, epochs=20, seed=0, seeds=None, dataset="synthetic"):
    """Train one neuron model with a shared, deterministic procedure.

    If seeds is provided, trains across all seeds and returns aggregated results.
    Otherwise trains with single seed and returns individual results.
    """
    if seeds is not None:
        results = [train_model(neuron, samples, steps, epochs, seed=s, dataset=dataset) for s in seeds]
        return _aggregate_results(results, neuron, samples, steps, epochs, seeds)

    torch.manual_seed(seed)
    if dataset == "mnist":
        x, y = load_mnist(train=True, samples=1000, seed=seed)
    elif dataset == "fashion_mnist":
        x, y = load_fashion_mnist(train=True, samples=1000, seed=seed)
    else:
        x, y = synthetic_data(samples=samples, seed=seed)
    split = max(1, int(samples * 0.8))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    model = SNNClassifier(neuron=neuron, steps=steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(train_x)
        loss_fn(logits, train_y).backward()
        optimizer.step()
    train_seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        logits, spikes = model(test_x)
        test_accuracy = (logits.argmax(1) == test_y).float().mean().item()
        train_logits, _ = model(train_x)
        train_accuracy = (train_logits.argmax(1) == train_y).float().mean().item()
    return {
        "neuron": neuron,
        "samples": samples,
        "steps": steps,
        "epochs": epochs,
        "seed": seed,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "training_seconds": train_seconds,
        "spike_rate": spikes.float().mean().item(),
    }


def _aggregate_results(results, neuron, samples, steps, epochs, seeds):
    train_accuracies = [r["train_accuracy"] for r in results]
    test_accuracies = [r["test_accuracy"] for r in results]
    training_seconds = [r["training_seconds"] for r in results]
    spike_rates = [r["spike_rate"] for r in results]
    
    return {
        "neuron": neuron,
        "samples": samples,
        "steps": steps,
        "epochs": epochs,
        "seeds": seeds,
        "mean_train_accuracy": statistics.mean(train_accuracies),
        "std_train_accuracy": statistics.stdev(train_accuracies) if len(seeds) > 1 else 0.0,
        "mean_test_accuracy": statistics.mean(test_accuracies),
        "std_test_accuracy": statistics.stdev(test_accuracies) if len(seeds) > 1 else 0.0,
        "mean_training_seconds": statistics.mean(training_seconds),
        "std_training_seconds": statistics.stdev(training_seconds) if len(seeds) > 1 else 0.0,
        "mean_spike_rate": statistics.mean(spike_rates),
        "std_spike_rate": statistics.stdev(spike_rates) if len(seeds) > 1 else 0.0,
        "per_seed": results,
    }


def run_experiments(neuron, seeds, samples=256, steps=12, epochs=20, dataset="synthetic", full_metrics=False):
    """Train one model across all seeds and return aggregated statistics."""
    result = train_model(neuron=neuron, samples=samples, steps=steps, epochs=epochs, seeds=seeds, dataset=dataset)
    return compute_all_metrics_for_experiment(result, full_metrics=full_metrics)
