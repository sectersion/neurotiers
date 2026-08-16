import json

import torch

from neuron_benchmark.cli import run
from neuron_benchmark.model import SNNClassifier
from neuron_benchmark.robustness import (
    noise_injection_experiment,
    reduced_training_data_experiment,
    run_robustness,
    timestep_sensitivity_experiment,
)
from neuron_benchmark.training import train_model, run_experiments, NEURONS


def test_all_neurons_produce_expected_shapes():
    x = torch.rand(5, 8)
    for name in ("if", "lif", "adaptive-lif", "izhikevich", "hodgkin-huxley"):
        logits, spikes = SNNClassifier(neuron=name, steps=4)(x)
        assert logits.shape == (5, 2)
        assert spikes.shape == (4, 5, 16)


def test_run_returns_json_metrics():
    result = run(samples=10, steps=3)
    json.dumps(result)
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["spike_rate"] <= 1
    assert result["runtime_seconds"] >= 0


def test_training_loop_updates_all_neurons():
    for name in ("if", "lif", "adaptive-lif", "izhikevich", "hodgkin-huxley"):
        result = train_model(name, samples=20, steps=3, epochs=2, seed=1)
        assert 0 <= result["test_accuracy"] <= 1
        assert result["training_seconds"] >= 0


def test_noise_injection_experiment():
    result = noise_injection_experiment(neuron="lif", samples=20, steps=3, epochs=2, seed=1, noise_stds=[0.0, 0.1])
    assert result["experiment"] == "noise_injection"
    assert len(result["results"]) == 2
    noise_stds_found = [r["noise_std"] for r in result["results"]]
    assert 0.0 in noise_stds_found
    assert 0.1 in noise_stds_found
    for r in result["results"]:
        assert 0 <= r["test_accuracy"] <= 1


def test_timestep_sensitivity_experiment():
    result = timestep_sensitivity_experiment(neuron="lif", samples=20, steps_list=[4, 6], epochs=2, seed=1)
    assert result["experiment"] == "timestep_sensitivity"
    assert len(result["results"]) == 2
    for r in result["results"]:
        assert r["steps"] in [4, 6]
        assert 0 <= r["test_accuracy"] <= 1


def test_reduced_training_data_experiment():
    result = reduced_training_data_experiment(neuron="lif", samples=20, steps=3, epochs=2, seed=1, train_fractions=[0.5, 1.0])
    assert result["experiment"] == "reduced_training_data"
    assert len(result["results"]) == 2
    for r in result["results"]:
        assert r["train_fraction"] in [0.5, 1.0]
        assert 0 <= r["test_accuracy"] <= 1


def test_run_robustness_returns_all_experiments():
    result = run_robustness(neuron="lif", samples=20, steps=3, epochs=2, seed=1,
                            noise_stds=[0.0], steps_list=[4], train_fractions=[1.0])
    assert "noise_injection" in result
    assert "timestep_sensitivity" in result
    assert "reduced_training_data" in result
    assert result["neuron"] == "lif"


def test_train_model_single_seed_returns_all_fields():
    result = train_model("lif", samples=20, steps=3, epochs=2, seed=42)
    assert "neuron" in result
    assert "seed" in result
    assert "train_accuracy" in result
    assert "test_accuracy" in result
    assert "training_seconds" in result
    assert "spike_rate" in result


def test_train_model_multiple_seeds_returns_aggregated_results():
    seeds = [0, 1, 2]
    result = train_model("lif", samples=20, steps=3, epochs=2, seeds=seeds)
    assert result["neuron"] == "lif"
    assert result["seeds"] == seeds
    assert "mean_train_accuracy" in result
    assert "std_train_accuracy" in result
    assert "mean_test_accuracy" in result
    assert "std_test_accuracy" in result
    assert "mean_training_seconds" in result
    assert "std_training_seconds" in result
    assert "mean_spike_rate" in result
    assert "std_spike_rate" in result
    assert "per_seed" in result
    assert len(result["per_seed"]) == len(seeds)


def test_train_model_multiple_seeds_single_seed_stdev_is_zero():
    seeds = [42]
    result = train_model("lif", samples=20, steps=3, epochs=2, seeds=seeds)
    assert result["std_train_accuracy"] == 0.0
    assert result["std_test_accuracy"] == 0.0
    assert result["std_training_seconds"] == 0.0
    assert result["std_spike_rate"] == 0.0


def test_run_experiments_returns_aggregated_stats():
    seeds = [0, 1]
    result = run_experiments("lif", seeds=seeds, samples=20, steps=3, epochs=2)
    assert result["neuron"] == "lif"
    assert "mean_train_accuracy" in result
    assert "per_seed" in result
    assert len(result["per_seed"]) == 2


def test_run_experiments_all_neurons():
    seeds = [0, 1]
    for neuron in NEURONS:
        result = run_experiments(neuron, seeds=seeds, samples=20, steps=3, epochs=2)
        assert result["neuron"] == neuron
        assert len(result["per_seed"]) == len(seeds)
