"""Interactive demo for neuron benchmark using Streamlit."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from neuron_benchmark.data import load_fashion_mnist, load_mnist, synthetic_data
from neuron_benchmark.model import SNNClassifier

NEURONS = ["if", "lif", "adaptive-lif", "izhikevich", "hodgkin-huxley"]
DATASETS = {"Synthetic": "synthetic", "MNIST": "mnist", "Fashion-MNIST": "fashion_mnist"}


def get_data(dataset_name, samples, seed):
    if dataset_name == "synthetic":
        return synthetic_data(samples=samples, seed=seed)
    elif dataset_name == "mnist":
        return load_mnist(train=True, samples=samples, seed=seed)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(train=True, samples=samples, seed=seed)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(neuron, features, classes, steps):
    return SNNClassifier(features=features, hidden=32, classes=classes, neuron=neuron, steps=steps)


def run_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, spikes = model(x)
        accuracy = (logits.argmax(1) == y).float().mean().item()
        spike_rate = spikes.float().mean().item()
    return logits, spikes, accuracy, spike_rate


def train_model(model, train_x, train_y, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(train_x)
        loss_fn(logits, train_y).backward()
        optimizer.step()
    return model


def run_simulation(neuron, dataset_name, samples, steps, noise_level, seed, epochs):
    torch.manual_seed(seed)
    x, y = get_data(dataset_name, samples, seed)

    if noise_level > 0:
        x = x + noise_level * torch.randn_like(x)
        x = x.clamp(0, 1)

    split = max(1, int(samples * 0.8))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]

    # Untrained model
    untrained_model = build_model(neuron, x.shape[1], len(y.unique()), steps)
    untrained_logits, untrained_spikes, untrained_acc, untrained_sr = run_model(untrained_model, test_x, test_y)

    # Trained model
    trained_model = build_model(neuron, x.shape[1], len(y.unique()), steps)
    trained_model = train_model(trained_model, train_x, train_y, epochs)
    trained_logits, trained_spikes, trained_acc, trained_sr = run_model(trained_model, test_x, test_y)

    return {
        "untrained": {"logits": untrained_logits, "spikes": untrained_spikes,
                      "accuracy": untrained_acc, "spike_rate": untrained_sr},
        "trained":  {"logits": trained_logits,  "spikes": trained_spikes,
                      "accuracy": trained_acc,  "spike_rate": trained_sr},
        "test_x": test_x, "test_y": test_y,
        "neuron": neuron, "dataset": dataset_name,
        "steps": steps, "samples": samples,
        "noise_level": noise_level, "seed": seed,
        "epochs": epochs,
    }


def plot_raster(spikes, title="Spike Raster"):
    spike_tensor = spikes.detach().cpu()
    if spike_tensor.ndim == 3:
        spike_tensor = spike_tensor.mean(1)
    n_steps, n_neurons = spike_tensor.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    sp = np.where(spike_tensor.numpy() > 0.5)
    ax.scatter(sp[0], sp[1], s=3, c="black", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)
    ax.set_xlim(-0.5, n_steps + 0.5)
    ax.set_ylim(-0.5, n_neurons + 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction_histogram(logits, labels, title):
    preds = logits.argmax(1).cpu().numpy()
    true = labels.cpu().numpy()
    n_classes = int(true.max()) + 2
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist([true, preds], bins=range(n_classes), label=["True", "Predicted"], alpha=0.7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_comparison_bar(before_acc, after_acc, before_sr, after_sr, neuron):
    labels = ["Accuracy", "Spike Rate"]
    before_vals = [before_acc, before_sr]
    after_vals = [after_acc, after_sr]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, before_vals, width, label="Before Training", color="#ff7f7f", alpha=0.8)
    ax.bar(x + width/2, after_vals, width, label="After Training", color="#7f7fff", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"{neuron} — Before vs After Training")
    ax.legend()
    ax.set_ylim(0, max(max(before_vals), max(after_vals)) * 1.2)
    for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
        ax.text(i - width/2, bv + 0.01, f"{bv:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, av + 0.01, f"{av:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    return fig


def plot_membrane(model, test_x, neuron_type, steps):
    from neuron_benchmark.neurons import IF, LIF, AdaptiveLIF, Izhikevich, HodgkinHuxley
    neurons_map = {"if": IF, "lif": LIF, "adaptive-lif": AdaptiveLIF,
                   "izhikevich": Izhikevich, "hodgkin-huxley": HodgkinHuxley}
    neuron_instance = neurons_map.get(neuron_type, LIF)()

    # Work with numpy throughout to avoid autograd graph issues
    inputs_np = model.encoder(test_x[:1]).relu().detach()
    inputs_np = inputs_np.unsqueeze(0).expand(steps, -1, -1)[:, 0, :].cpu().numpy()

    if isinstance(neuron_instance, IF):
        membrane = np.zeros_like(inputs_np[0])
        traces = []
        for current in inputs_np:
            membrane = membrane + current
            spike = (membrane >= neuron_instance.threshold).astype(np.float32)
            membrane = membrane * (1 - spike)
            traces.append(membrane.copy())

    elif isinstance(neuron_instance, LIF):
        membrane = np.zeros_like(inputs_np[0])
        traces = []
        for current in inputs_np:
            membrane = neuron_instance.decay * membrane + current
            spike = (membrane >= neuron_instance.threshold).astype(np.float32)
            membrane = membrane * (1 - spike)
            traces.append(membrane.copy())

    elif isinstance(neuron_instance, AdaptiveLIF):
        membrane = np.zeros_like(inputs_np[0])
        extra = np.zeros_like(membrane)
        traces = []
        for current in inputs_np:
            membrane = neuron_instance.decay * membrane + current
            spike = (membrane >= neuron_instance.threshold + extra).astype(np.float32)
            membrane = membrane * (1 - spike)
            extra = neuron_instance.decay * extra + neuron_instance.adaptation * spike
            traces.append(membrane.copy())

    elif isinstance(neuron_instance, Izhikevich):
        voltage = np.full_like(inputs_np[0], neuron_instance.c)
        recovery = neuron_instance.b * voltage
        traces = []
        for current in inputs_np:
            dv = 0.04 * voltage**2 + 5 * voltage + 140 - recovery + neuron_instance.gain * current
            voltage = voltage + neuron_instance.dt * dv
            recovery = recovery + neuron_instance.dt * neuron_instance.a * (neuron_instance.b * voltage - recovery)
            spike = (voltage >= neuron_instance.threshold).astype(np.float32)
            voltage = np.where(spike > 0.5, neuron_instance.c, voltage)
            recovery = recovery + neuron_instance.d * spike
            traces.append(voltage.copy())

    elif isinstance(neuron_instance, HodgkinHuxley):
        voltage = np.zeros_like(inputs_np[0])
        m = np.full_like(voltage, 0.0529)
        h = np.full_like(voltage, 0.596)
        n = np.full_like(voltage, 0.317)
        traces = []
        for current in inputs_np:
            abs_v = voltage - 65.0
            am = 0.1 * (25 - abs_v) / (np.exp((25 - abs_v) / 10) - 1 + 1e-8)
            bm = 4 * np.exp(-abs_v / 18)
            ah = 0.07 * np.exp(-abs_v / 20)
            bh = 1.0 / (np.exp((30 - abs_v) / 10) + 1)
            an = 0.01 * (10 - abs_v) / (np.exp((10 - abs_v) / 10) - 1 + 1e-8)
            bn = 0.125 * np.exp(-abs_v / 80)
            m = np.clip(m + neuron_instance.dt * (am * (1 - m) - bm * m), 0, 1)
            h = np.clip(h + neuron_instance.dt * (ah * (1 - h) - bh * h), 0, 1)
            n = np.clip(n + neuron_instance.dt * (an * (1 - n) - bn * n), 0, 1)
            ionic = (120.0 * m**3 * h * (voltage - 50) +
                     36.0 * n**4 * (voltage + 77) + 0.3 * (voltage + 54.4))
            voltage = np.clip(voltage + neuron_instance.dt * (neuron_instance.gain * current - ionic), -100, 100)
            traces.append(voltage.copy())

    trace_array = np.stack(traces)  # shape: (steps, neurons)
    n_show = min(5, trace_array.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_show):
        ax.plot(trace_array[:, i], label=f"Neuron {i}", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title(f"Membrane Trace -- {neuron_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_shareable_json(results):
    return {
        "neuron": results["neuron"],
        "dataset": results["dataset"],
        "steps": results["steps"],
        "samples": results["samples"],
        "epochs": results["epochs"],
        "noise_level": results["noise_level"],
        "seed": results["seed"],
        "before_training": {
            "test_accuracy": round(results["untrained"]["accuracy"], 4),
            "spike_rate": round(results["untrained"]["spike_rate"], 4),
        },
        "after_training": {
            "test_accuracy": round(results["trained"]["accuracy"], 4),
            "spike_rate": round(results["trained"]["spike_rate"], 4),
        },
        "improvement": {
            "accuracy_delta": round(results["trained"]["accuracy"] - results["untrained"]["accuracy"], 4),
            "spike_rate_delta": round(results["trained"]["spike_rate"] - results["untrained"]["spike_rate"], 4),
        },
    }


def main():
    import streamlit as st

    st.set_page_config(page_title="Neuron Benchmark Demo", page_icon="🧠", layout="wide")
    st.title("🧠 Neuron Benchmark — Live Comparison")
    st.markdown("**Research Question:** How does neuron-model complexity affect spiking behavior and classification accuracy?")
    st.markdown("---")

    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.header("⚙️ Configuration")
        neuron = st.selectbox("Neuron Model", NEURONS, index=1,
                              help="IF = baseline, LIF = leaky, Adaptive LIF = threshold adaptation, Izhikevich = rich dynamics, Hodgkin-Huxley = biophysical reference")
        dataset = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
        st.markdown("---")
        st.subheader("Simulation")
        steps = st.slider("Simulation Steps", 4, 24, 12)
        samples = st.slider("Samples", 32, 512, 128)
        noise_level = st.slider("Input Noise", 0.0, 0.5, 0.0, 0.05)
        seed = st.number_input("Random Seed", 0, 999, 42)
        st.markdown("---")
        st.subheader("Training")
        epochs = st.slider("Training Epochs", 0, 50, 5)
        show_membrane = st.checkbox("Show Membrane Trace", value=True)

        run_button = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    if run_button or "results" in st.session_state:
        if run_button:
            dataset_name = DATASETS[dataset]
            with st.spinner("Training and evaluating..."):
                res = run_simulation(neuron, dataset_name, samples, steps,
                                     noise_level, seed, epochs)
            st.session_state["results"] = res

        results = st.session_state.get("results", {})
        if results:
            u = results["untrained"]
            t = results["trained"]

            with col2:
                st.header("📊 Before vs After Training")
                left, mid, right = st.columns(3)
                left.metric("**Before: Accuracy**", f"{u['accuracy']:.1%}",
                            delta=f"{(t['accuracy'] - u['accuracy']):.1%}",
                            delta_color="normal")
                mid.metric("**After: Accuracy**", f"{t['accuracy']:.1%}")
                right.metric("**Spike Rate (before→after)**",
                             f"{u['spike_rate']:.3f} → {t['spike_rate']:.3f}",
                             delta=f"{t['spike_rate'] - u['spike_rate']:.3f}",
                             delta_color="normal")

                st.markdown("---")
                st.subheader("📈 Comparison")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.pyplot(plot_comparison_bar(
                        u["accuracy"], t["accuracy"],
                        u["spike_rate"], t["spike_rate"],
                        results["neuron"]))
                with comp_col2:
                    st.pyplot(plot_prediction_histogram(
                        t["logits"], results["test_y"],
                        "Prediction Distribution (After Training)"))

                st.markdown("---")
                st.subheader("⚡ Spike Rasters")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.pyplot(plot_raster(u["spikes"], "Before Training"))
                with r_col2:
                    st.pyplot(plot_raster(t["spikes"], "After Training"))

                if show_membrane:
                    st.markdown("---")
                    st.subheader("🧬 Membrane Dynamics")
                    trained_model = build_model(results["neuron"],
                                               results["test_x"].shape[1],
                                               len(results["test_y"].unique()),
                                               results["steps"])
                    trained_model = train_model(trained_model,
                                                results["test_x"][:results["samples"]//5],
                                                results["test_y"][:results["samples"]//5],
                                                results["epochs"])
                    st.pyplot(plot_membrane(trained_model,
                                           results["test_x"],
                                           results["neuron"],
                                           results["steps"]))

                st.markdown("---")
                st.subheader("📤 Share Results")
                share_data = make_shareable_json(results)
                share_str = json.dumps(share_data, indent=2)
                st.download_button(
                    label="⬇ Download Results JSON",
                    data=share_str,
                    file_name=f"neuron_benchmark_{results['neuron']}_{results['dataset']}.json",
                    mime="application/json",
                    use_container_width=True)
                with st.expander("📄 JSON Preview"):
                    st.code(share_str, language="json")

                st.markdown("---")
                st.caption(
                    f"Model: **{results['neuron']}** | Dataset: **{results['dataset']}** "
                    f"| Steps: **{results['steps']}** | Epochs: **{results['epochs']}** "
                    f"| Noise: **{results['noise_level']}** | Seed: **{results['seed']}**"
                )


if __name__ == "__main__":
    main()
