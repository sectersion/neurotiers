import math

import torch

from .model import SNNClassifier


def energy_proxy(synaptic_ops, avg_spike_rate):
    return synaptic_ops * avg_spike_rate


def compute_all_metrics(result, full_metrics=False):
    model = SNNClassifier(
        features=8, hidden=16, classes=2,
        neuron=result["neuron"], steps=result["steps"]
    )
    model_params = sum(p.numel() for p in model.parameters())
    training_seconds = result.get("training_seconds", result.get("mean_training_seconds", 0))
    spike_rate = result.get("spike_rate", result.get("mean_spike_rate", 0))
    test_accuracy = result.get("test_accuracy", result.get("mean_test_accuracy", 0))

    synaptic_ops = model_params * training_seconds
    result["synaptic_ops"] = synaptic_ops
    result["avg_spike_rate"] = spike_rate
    result["energy_proxy"] = energy_proxy(synaptic_ops, spike_rate)

    if spike_rate > 0:
        result["ops_per_spike"] = training_seconds / spike_rate
    else:
        result["ops_per_spike"] = math.inf

    if training_seconds > 0:
        result["efficiency_score"] = test_accuracy / training_seconds
    else:
        result["efficiency_score"] = 0.0

    return result


def compute_all_metrics_for_experiment(result, full_metrics=False):
    result = compute_all_metrics(result, full_metrics)
    if "per_seed" in result:
        result["per_seed"] = [
            compute_all_metrics(r, full_metrics) for r in result["per_seed"]
        ]
    return result