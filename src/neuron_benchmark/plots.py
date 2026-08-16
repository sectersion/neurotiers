import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .neurons import IF, LIF, AdaptiveLIF, Izhikevich, HodgkinHuxley
from .data import synthetic_data, load_mnist, load_fashion_mnist
from .model import SNNClassifier


def plot_accuracy_vs_runtime(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for neuron in results_df["neuron"].unique():
        subset = results_df[results_df["neuron"] == neuron]
        ax.scatter(subset["runtime_seconds"], subset["accuracy"], label=neuron, alpha=0.7, s=60)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Runtime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_accuracy_vs_spike_rate(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for neuron in results_df["neuron"].unique():
        subset = results_df[results_df["neuron"] == neuron]
        ax.scatter(subset["spike_rate"], subset["accuracy"], label=neuron, alpha=0.7, s=60)
    ax.set_xlabel("Spike Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Spike Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def _record_membrane_trace(neuron_model, inputs):
    traces = {}
    if isinstance(neuron_model, IF):
        membrane = torch.zeros_like(inputs[0])
        for t, current in enumerate(inputs):
            membrane = membrane + current
            spike = (membrane >= neuron_model.threshold).to(membrane.dtype)
            membrane = membrane * (1 - spike)
            traces[t] = membrane.clone()
    elif isinstance(neuron_model, LIF):
        membrane = torch.zeros_like(inputs[0])
        for t, current in enumerate(inputs):
            membrane = neuron_model.decay * membrane + current
            spike = (membrane >= neuron_model.threshold).to(membrane.dtype)
            membrane = membrane * (1 - spike)
            traces[t] = membrane.clone()
    elif isinstance(neuron_model, AdaptiveLIF):
        membrane = torch.zeros_like(inputs[0])
        extra = torch.zeros_like(membrane)
        for t, current in enumerate(inputs):
            membrane = neuron_model.decay * membrane + current
            spike = (membrane >= neuron_model.threshold + extra).to(membrane.dtype)
            membrane = membrane * (1 - spike)
            extra = neuron_model.decay * extra + neuron_model.adaptation * spike
            traces[t] = membrane.clone()
    elif isinstance(neuron_model, Izhikevich):
        voltage = torch.full_like(inputs[0], neuron_model.c)
        recovery = neuron_model.b * voltage
        for t, current in enumerate(inputs):
            dv = 0.04 * voltage.square() + 5 * voltage + 140 - recovery + neuron_model.gain * current
            voltage = voltage + neuron_model.dt * dv
            recovery = recovery + neuron_model.dt * neuron_model.a * (neuron_model.b * voltage - recovery)
            spike = (voltage >= neuron_model.threshold).to(voltage.dtype)
            voltage = torch.where(spike.bool(), torch.as_tensor(neuron_model.c, device=voltage.device, dtype=voltage.dtype), voltage)
            recovery = recovery + neuron_model.d * spike
            traces[t] = voltage.clone()
    elif isinstance(neuron_model, HodgkinHuxley):
        voltage = torch.zeros_like(inputs[0])
        m = torch.full_like(voltage, 0.0529)
        h = torch.full_like(voltage, 0.596)
        n = torch.full_like(voltage, 0.317)
        for t, current in enumerate(inputs):
            absolute_v = voltage - 65.0
            am = 0.1 * (25 - absolute_v) / (torch.exp((25 - absolute_v) / 10) - 1 + 1e-8)
            bm = 4 * torch.exp(-absolute_v / 18)
            ah = 0.07 * torch.exp(-absolute_v / 20)
            bh = 1 / (torch.exp((30 - absolute_v) / 10) + 1)
            an = 0.01 * (10 - absolute_v) / (torch.exp((10 - absolute_v) / 10) - 1 + 1e-8)
            bn = 0.125 * torch.exp(-absolute_v / 80)
            m = (m + neuron_model.dt * (am * (1 - m) - bm * m)).clamp(0, 1)
            h = (h + neuron_model.dt * (ah * (1 - h) - bh * h)).clamp(0, 1)
            n = (n + neuron_model.dt * (an * (1 - n) - bn * n)).clamp(0, 1)
            ionic = (120 * m**3 * h * (voltage - 50) +
                     36 * n**4 * (voltage + 77) + 0.3 * (voltage + 54.4))
            voltage = (voltage + neuron_model.dt * (neuron_model.gain * current - ionic)).clamp(-100, 100)
            traces[t] = voltage.clone()
    return traces


def plot_membrane_traces(model, inputs):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    neurons_map = {"if": IF, "lif": LIF, "adaptive-lif": AdaptiveLIF,
                   "izhikevich": Izhikevich, "hodgkin-huxley": HodgkinHuxley}
    for idx, (name, neuron_cls) in enumerate(neurons_map.items()):
        if idx >= 5:
            break
        temp_model = SNNClassifier(neuron=name, steps=model.steps)
        temp_model.eval()
        with torch.no_grad():
            _, spikes = temp_model(inputs[:1]) if inputs.size(0) > 1 else temp_model(inputs)
        neuron_instance = neurons_map[name]()
        inputs_for_neuron = temp_model.encoder(inputs[:1]).relu().unsqueeze(0).expand(temp_model.steps, -1, -1)
        traces = _record_membrane_trace(neuron_instance, inputs_for_neuron[:, 0, :])
        trace_array = torch.stack([traces[t] for t in sorted(traces.keys())])
        time_steps = list(range(trace_array.shape[0]))
        neuron_idx = 0
        axes[idx].plot(time_steps, trace_array[:, neuron_idx].cpu().numpy(), linewidth=1.5)
        axes[idx].set_title(f"{name} - Neuron {neuron_idx}")
        axes[idx].set_xlabel("Time Step")
        axes[idx].set_ylabel("Membrane Potential (mV)")
        axes[idx].grid(True, alpha=0.3)
    for idx in range(len(neurons_map), 6):
        axes[idx].axis("off")
    plt.tight_layout()
    return fig, axes


def plot_raster(spikes):
    fig, ax = plt.subplots(figsize=(10, 5))
    spike_tensor = spikes.detach().cpu()
    if spike_tensor.ndim == 3:
        spike_tensor = spike_tensor.mean(1)
    n_steps, n_neurons = spike_tensor.shape
    spike_positions = torch.where(spike_tensor > 0.5)
    ax.scatter(spike_positions[0], spike_positions[1], s=3, c="black", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Spike Raster Plot")
    ax.set_xlim(-0.5, n_steps + 0.5)
    ax.set_ylim(-0.5, n_neurons + 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_robustness_results(robustness_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for neuron_name, results in robustness_results.items():
        noise_levels = results.get("noise_levels", [])
        accuracies = results.get("accuracies", [])
        if noise_levels and accuracies:
            ax.plot(noise_levels, accuracies, marker="o", label=neuron_name, linewidth=2)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to Input Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_pareto(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    if "mean_test_accuracy" in results_df.columns:
        x_col, y_col = "mean_training_seconds", "mean_test_accuracy"
        x_err = results_df.get("std_training_seconds", None)
        y_err = results_df.get("std_test_accuracy", None)
    else:
        x_col, y_col = "training_seconds", "test_accuracy"
        x_err, y_err = None, None

    neuron_types = results_df["neuron"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_types)))
    color_map = dict(zip(neuron_types, colors))

    for neuron in neuron_types:
        subset = results_df[results_df["neuron"] == neuron]
        ax.errorbar(subset[x_col], subset[y_col],
                    xerr=subset["std_training_seconds"] if "std_training_seconds" in subset.columns else None,
                    yerr=subset["std_test_accuracy"] if "std_test_accuracy" in subset.columns else None,
                    fmt="o", label=neuron, color=color_map[neuron], alpha=0.7, s=80, capsize=3)

    sorted_df = results_df.sort_values(x_col)
    pareto_frontier = []
    max_acc = 0
    for _, row in sorted_df.iterrows():
        acc = row[y_col]
        if acc >= max_acc:
            pareto_frontier.append(row)
            max_acc = acc
    if pareto_frontier:
        pf_df = pd.DataFrame(pareto_frontier).sort_values(x_col)
        ax.plot(pf_df[x_col], pf_df[y_col], "k--", linewidth=1.5, label="Pareto Frontier")

    for _, row in results_df.iterrows():
        ax.annotate(row["neuron"], (row[x_col], row[y_col]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs Training Time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_convergence(neuron, seeds, samples=256, steps=12, epochs=20, dataset="synthetic"):
    fig, ax = plt.subplots(figsize=(10, 6))
    dataset_loaders = {
        "synthetic": lambda: synthetic_data(samples=samples, seed=0),
        "mnist": lambda: load_mnist(train=True, samples=samples, seed=0),
        "fashion_mnist": lambda: load_fashion_mnist(train=True, samples=samples, seed=0),
    }
    load_fn = dataset_loaders.get(dataset, dataset_loaders["synthetic"])

    all_epoch_losses = []
    for seed in seeds:
        torch.manual_seed(seed)
        x, y = load_fn()
        split = max(1, int(len(x) * 0.8))
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]

        model = SNNClassifier(neuron=neuron, steps=steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        epoch_losses = []
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits, _ = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        all_epoch_losses.append(epoch_losses)

    all_epoch_losses = np.array(all_epoch_losses)
    mean_losses = all_epoch_losses.mean(axis=0)
    std_losses = all_epoch_losses.std(axis=0)

    epochs_range = np.arange(1, epochs + 1)
    ax.fill_between(epochs_range, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2)
    ax.plot(epochs_range, mean_losses, linewidth=2, label=f"{neuron} (n={len(seeds)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Convergence: {neuron} across {len(seeds)} seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_leaderboard(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    if "mean_test_accuracy" in results_df.columns:
        acc_col = "mean_test_accuracy"
        err_col = "std_test_accuracy"
    else:
        acc_col = "test_accuracy"
        err_col = None

    sorted_df = results_df.sort_values(acc_col, ascending=True)
    neuron_types = sorted_df["neuron"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_types)))
    color_map = dict(zip(neuron_types, colors))

    y_positions = np.arange(len(sorted_df))
    bars = ax.barh(y_positions, sorted_df[acc_col],
                   xerr=sorted_df[err_col] if err_col and err_col in sorted_df.columns else None,
                   color=[color_map[n] for n in sorted_df["neuron"]], alpha=0.8, capsize=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_df["neuron"])
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Neuron Leaderboard by Test Accuracy")
    for i, (neuron, acc) in enumerate(zip(sorted_df["neuron"], sorted_df[acc_col])):
        ax.text(acc + 0.01, i, f"{acc:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig, ax


def generate_report(results_df, output_dir, convergence_neuron=None, convergence_seeds=None,
                    convergence_kwargs=None, dataset="synthetic"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = {}
    fig1, _ = plot_accuracy_vs_runtime(results_df)
    figures["accuracy_vs_runtime"] = output_path / "accuracy_vs_runtime.png"
    fig1.savefig(figures["accuracy_vs_runtime"], dpi=150)
    plt.close(fig1)
    fig2, _ = plot_accuracy_vs_spike_rate(results_df)
    figures["accuracy_vs_spike_rate"] = output_path / "accuracy_vs_spike_rate.png"
    fig2.savefig(figures["accuracy_vs_spike_rate"], dpi=150)
    plt.close(fig2)

    if all(col in results_df.columns for col in ["test_accuracy", "training_seconds"]) or \
       all(col in results_df.columns for col in ["mean_test_accuracy", "mean_training_seconds"]):
        fig3, _ = plot_pareto(results_df)
        figures["pareto"] = output_path / "pareto.png"
        fig3.savefig(figures["pareto"], dpi=150)
        plt.close(fig3)

        fig4, _ = plot_leaderboard(results_df)
        figures["leaderboard"] = output_path / "leaderboard.png"
        fig4.savefig(figures["leaderboard"], dpi=150)
        plt.close(fig4)

    if convergence_neuron and convergence_seeds:
        ck = convergence_kwargs or {}
        fig5, _ = plot_convergence(convergence_neuron, convergence_seeds,
                                   dataset=dataset, **ck)
        figures["convergence"] = output_path / "convergence.png"
        fig5.savefig(figures["convergence"], dpi=150)
        plt.close(fig5)

    markdown_path = output_path / "summary.md"
    with open(markdown_path, "w") as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write("## Accuracy vs Runtime\n\n")
        f.write(f"![Accuracy vs Runtime](accuracy_vs_runtime.png)\n\n")
        f.write("## Accuracy vs Spike Rate\n\n")
        f.write(f"![Accuracy vs Spike Rate](accuracy_vs_spike_rate.png)\n\n")
        if "pareto" in figures:
            f.write("## Pareto Frontier\n\n")
            f.write(f"![Pareto Frontier](pareto.png)\n\n")
        if "leaderboard" in figures:
            f.write("## Leaderboard\n\n")
            f.write(f"![Leaderboard](leaderboard.png)\n\n")
        if "convergence" in figures:
            f.write("## Training Convergence\n\n")
            f.write(f"![Convergence](convergence.png)\n\n")
        f.write("## Results Table\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")
    return figures, markdown_path
