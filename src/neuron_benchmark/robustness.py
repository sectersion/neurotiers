import time

import torch
from torch import nn

from .data import synthetic_data
from .model import SNNClassifier


def _train_model_on_data(train_x, train_y, test_x, test_y, neuron, steps, epochs):
    torch.manual_seed(0)
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
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "training_seconds": train_seconds,
        "spike_rate": spikes.float().mean().item(),
    }


def _inject_noise(x, std):
    return x + torch.randn_like(x) * std


def _prepare_data(samples, seed):
    torch.manual_seed(seed)
    x, y = synthetic_data(samples=samples, seed=seed)
    split = max(1, int(samples * 0.8))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    return train_x, test_x, train_y, test_y


def noise_injection_experiment(neuron="lif", samples=256, steps=12, epochs=20, seed=0, noise_stds=[0.0, 0.05, 0.1, 0.2, 0.3]):
    """Vary Gaussian noise added to inputs and measure impact on accuracy."""
    train_x, test_x, train_y, test_y = _prepare_data(samples, seed)
    results = []
    for std in noise_stds:
        noisy_train_x = _inject_noise(train_x, std)
        noisy_test_x = _inject_noise(test_x, std)
        metrics = _train_model_on_data(noisy_train_x, train_y, noisy_test_x, test_y, neuron, steps, epochs)
        results.append({**metrics, "noise_std": std})
    return {"experiment": "noise_injection", "neuron": neuron, "samples": samples, "steps": steps, "results": results}


def timestep_sensitivity_experiment(neuron="lif", samples=256, steps_list=[4, 6, 8, 10, 12, 16, 20], epochs=20, seed=0):
    """Vary simulation timesteps and measure impact on accuracy and speed."""
    train_x, test_x, train_y, test_y = _prepare_data(samples, seed)
    results = []
    for steps in steps_list:
        metrics = _train_model_on_data(train_x, train_y, test_x, test_y, neuron, steps, epochs)
        results.append({**metrics, "steps": steps})
    return {"experiment": "timestep_sensitivity", "neuron": neuron, "samples": samples, "results": results}


def reduced_training_data_experiment(neuron="lif", samples=256, steps=12, epochs=20, seed=0, train_fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    """Vary training set size and measure impact on accuracy."""
    torch.manual_seed(seed)
    x, y = synthetic_data(samples=samples, seed=seed)
    split = max(1, int(samples * 0.8))
    full_train_x, test_x = x[:split], x[split:]
    full_train_y, test_y = y[:split], y[split:]
    results = []
    for frac in train_fractions:
        n_train = max(1, int(len(full_train_x) * frac))
        train_x = full_train_x[:n_train]
        train_y = full_train_y[:n_train]
        metrics = _train_model_on_data(train_x, train_y, test_x, test_y, neuron, steps, epochs)
        results.append({**metrics, "train_fraction": frac, "train_samples": n_train})
    return {"experiment": "reduced_training_data", "neuron": neuron, "samples": samples, "steps": steps, "results": results}


def run_robustness(neuron="lif", samples=256, steps=12, epochs=20, seed=0,
                   noise_stds=None, steps_list=None, train_fractions=None):
    """Run all robustness experiments and return combined results."""
    if noise_stds is None:
        noise_stds = [0.0, 0.05, 0.1, 0.2, 0.3]
    if steps_list is None:
        steps_list = [4, 6, 8, 10, 12, 16, 20]
    if train_fractions is None:
        train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    return {
        "neuron": neuron,
        "samples": samples,
        "steps": steps,
        "epochs": epochs,
        "seed": seed,
        "noise_injection": noise_injection_experiment(neuron, samples, steps, epochs, seed, noise_stds),
        "timestep_sensitivity": timestep_sensitivity_experiment(neuron, samples, steps_list, epochs, seed),
        "reduced_training_data": reduced_training_data_experiment(neuron, samples, steps, epochs, seed, train_fractions),
    }
