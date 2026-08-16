"""
Generate a polished benchmark demo video using MoviePy.
9 slides: title -> question -> architecture -> models -> traces -> demo
        -> results -> decision -> conclusion
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy import ImageClip, concatenate_videoclips
from PIL import Image

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = "#0B2447"
TEAL   = "#19376D"
SKY    = "#576CBC"
GOLD   = "#F0A500"
WHITE  = "#FFFFFF"
LGREY  = "#E8E8E8"
DGREY  = "#555555"
RED    = "#E63946"
GREEN  = "#2DC653"

NC = {
    "IF":             "#76C7F4",
    "LIF":            "#4ECDC4",
    "Adaptive LIF":   "#45B7D1",
    "Izhikevich":     "#F7DC6F",
    "Hodgkin-Huxley": "#E74C3C",
}


# ── Figure helpers ─────────────────────────────────────────────────────────────
def fig2np(fig):
    from io import BytesIO
    Image.MAX_IMAGE_PIXELS = None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def slide(duration):
    def decorator(makefig):
        def wrapper(*a, **kw):
            fig = makefig(*a, **kw)
            arr = fig2np(fig)
            return ImageClip(arr).with_duration(duration)
        return wrapper
    return decorator


# ── 1. Title ─────────────────────────────────────────────────────────────────
@slide(4.0)
def make_title():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.axis("off")
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, a in [(0.55, 0.07), (0.45, 0.10), (0.35, 0.14)]:
        ax.fill(0.5 + r*np.cos(theta), 0.5 + r*np.sin(theta), color=TEAL, alpha=a)
    ax.text(0.5, 0.62, "Neuron Benchmark", ha="center", va="center",
            transform=ax.transAxes, color=WHITE, fontsize=48, fontweight="bold")
    ax.text(0.5, 0.42, "Comparing Spiking Neuron Models for SNN Classification",
            ha="center", va="center", transform=ax.transAxes, color=SKY, fontsize=18)
    ax.text(0.5, 0.26, "Hackathon Research Project -- 2026",
            ha="center", va="center", transform=ax.transAxes, color=DGREY, fontsize=14)
    ax.plot(np.linspace(0, 1, 300), [0.19]*300, color=GOLD, lw=2, alpha=0.6, transform=ax.transAxes)
    return fig


# ── 2. Research Question ───────────────────────────────────────────────────────
@slide(5.0)
def make_question():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(TEAL); ax.set_facecolor(TEAL); ax.axis("off")
    ax.text(0.5, 0.78, "THE RESEARCH QUESTION", ha="center", va="center",
            transform=ax.transAxes, color=GOLD, fontsize=16, fontweight="bold")
    ax.text(0.5, 0.52,
            '"How does neuron-model complexity affect\ntemporal behavior, spike activity,\n'
            'and computational efficiency in SNNs?"',
            ha="center", va="center", transform=ax.transAxes,
            color=WHITE, fontsize=22, style="italic", linespacing=1.7)
    for i, (tag, q) in enumerate([
        ("[SPIKE]  ", "Spike dynamics & firing patterns"),
        ("[CHART]  ", "Classification accuracy"),
        ("[TIME]   ", "Simulation speed & efficiency"),
    ]):
        ax.text(0.5, 0.22 - i*0.08, f"{tag}  {q}", ha="center", va="center",
                transform=ax.transAxes, color=WHITE, fontsize=14, alpha=0.85)
    return fig


# ── 3. Architecture ────────────────────────────────────────────────────────────
@slide(6.0)
def make_arch():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    boxes = [
        (2.0, 4.5, "Input\nRates"),
        (5.5, 4.5, "Encoder\n(Linear)"),
        (9.0, 4.5, "Neuron\n(SNN Layer)"),
        (12.5, 4.5, "Readout\n(Linear)"),
        (15.0, 4.5, "Class"),
    ]
    kw = dict(boxstyle="round,pad=0.5", facecolor=TEAL, edgecolor=SKY, linewidth=2)
    for x, y, lbl in boxes:
        ax.text(x, y, lbl, ha="center", va="center", fontsize=14,
                color=WHITE, fontweight="bold", bbox=kw)
    for (x1,y1,_), (x2,y2,_) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2-0.9, y2), xytext=(x1+0.9, y1),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5))
    ax.text(8, 2.2, "Train -> Evaluate -> Compare -> Visualize",
            ha="center", va="center", fontsize=14, color=GOLD)
    ax.text(8, 1.5, "Python | PyTorch | Streamlit | MoviePy",
            ha="center", va="center", fontsize=12, color=DGREY)
    ax.text(8, 0.9, "Reproducible | Multi-seed | Open-source",
            ha="center", va="center", fontsize=12, color=DGREY)
    return fig


# ── 4. Neuron Models ──────────────────────────────────────────────────────────
@slide(6.0)
def make_models():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.axis("off")
    ax.text(8, 8.4, "NEURON MODELS UNDER TEST", ha="center", va="top",
            fontsize=18, fontweight="bold", color=GOLD)
    rows = [
        ("IF",             "Integrate-and-Fire",          "Simplest baseline. Hard reset.",           NC["IF"]),
        ("LIF",            "Leaky Integrate-and-Fire",    "Decay leak + threshold spiking.",        NC["LIF"]),
        ("Adaptive LIF",   "Adaptive LIF",                "Spike-frequency adaptation.",             NC["Adaptive LIF"]),
        ("Izhikevich",     "Izhikevich Model",            "Rich dynamics, fast simulation.",        NC["Izhikevich"]),
        ("Hodgkin-Huxley","Hodgkin-Huxley",              "Biophysical reference. Slower.",          NC["Hodgkin-Huxley"]),
    ]
    for i, (s, n, d, c) in enumerate(rows):
        y = 7.0 - i*1.3
        ax.add_patch(plt.Rectangle((0.5, y-0.35), 0.5, 0.7, facecolor=c))
        ax.text(1.3,  y+0.12, s, ha="left", va="center",
                fontsize=16, fontweight="bold", color=WHITE)
        ax.text(1.3,  y-0.22, f"{n}  --  {d}", ha="left", va="center",
                fontsize=12, color=LGREY)
    return fig


# ── 5. Membrane Traces (static) ──────────────────────────────────────────────
@slide(8.0)
def make_traces():
    n_list = ["IF", "LIF", "Adaptive LIF", "Izhikevich"]
    ncols, nrows = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY)
    axes = axes.flatten()
    t = np.arange(40)
    for idx, name in enumerate(n_list):
        ax = axes[idx]
        ax.set_facecolor(TEAL)
        # Generate representative voltage trace
        np.random.seed(42 + idx)
        if name == "IF":
            v = np.cumsum(np.random.randn(40)*0.1 + 0.15).clip(-2, 3)
        elif name == "LIF":
            v = np.zeros(40)
            for i in range(1, 40):
                v[i] = 0.8*v[i-1] + np.random.randn()*0.1 + 0.15
            v = v.clip(-2, 3)
        elif name == "Adaptive LIF":
            v = np.zeros(40); th = 0
            for i in range(1, 40):
                v[i] = 0.8*v[i-1] + np.random.randn()*0.1 + 0.15
                th += 0.05
                v[i] = max(v[i], th)
            v = v.clip(-2, 3)
        else:  # Izhikevich
            v = -65 + np.cumsum(np.random.randn(40)*3 + 0.5)
            spikes = v > 30
            v[spikes] = -65
            v = v - v.min()
        ax.plot(t, v, color=NC[name], lw=1.5)
        ax.axhline(0, color=DGREY, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(name, color=WHITE, fontsize=12, fontweight="bold", pad=6)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.spines[:].set_visible(False)
        ax.set_xlim(0, 40)
    for idx in range(len(n_list), ncols*nrows):
        axes[idx].axis("off")
    fig.suptitle("Membrane Dynamics -- Voltage Traces",
                 color=GOLD, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── 6. Before vs After ────────────────────────────────────────────────────────
@slide(7.0)
def make_demo(before_acc, after_acc, before_sr, after_sr):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY)
    titles  = ["BEFORE TRAINING", "AFTER TRAINING", "IMPROVEMENT"]
    accs   = [before_acc, after_acc, after_acc - before_acc]
    srs    = [before_sr, after_sr, after_sr - before_sr]
    colors = [RED, GREEN, SKY]
    for col, (ax, title, acc, sr, color) in enumerate(zip(axes, titles, accs, srs, colors)):
        ax.set_facecolor(NAVY); ax.axis("off")
        ax.text(0.5, 0.82, title, ha="center", va="top", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=color)
        ax.text(0.5, 0.55, f"{acc:.0%}", ha="center", va="center",
                transform=ax.transAxes, fontsize=48, fontweight="bold", color=WHITE)
        ax.text(0.5, 0.42, "Accuracy", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=DGREY)
        ax.text(0.5, 0.25, f"{sr:.3f}", ha="center", va="center",
                transform=ax.transAxes, fontsize=28, fontweight="bold", color=color)
        ax.text(0.5, 0.12, "Spike Rate", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=DGREY)
    fig.suptitle("Live Demo -- LIF Neuron", color=GOLD, fontsize=16,
                 fontweight="bold", y=0.97)
    fig.tight_layout()
    return fig


# ── 7. Results ────────────────────────────────────────────────────────────────
@slide(8.0)
def make_results(results):
    names  = [r["name"]       for r in results]
    accs   = [r["accuracy"]   for r in results]
    times  = [r["time_s"]    for r in results]
    srs    = [r["spike_rate"] for r in results]
    colors = [NC.get(n, SKY) for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY)
    kw = dict(width=0.6, zorder=3)
    for ax in axes:
        ax.set_facecolor(NAVY)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", alpha=0.2, color=LGREY)
        ax.yaxis.set_tick_params(labelcolor=WHITE)

    # Accuracy
    bars = axes[0].bar(range(5), accs, color=colors, **kw)
    axes[0].set_title("Test Accuracy", color=GOLD, fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    for b, v in zip(bars, accs):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}",
                     ha="center", color=WHITE, fontsize=9)
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(names, color=WHITE, rotation=20, ha="right")

    # Runtime
    bars = axes[1].bar(range(5), times, color=colors, **kw)
    axes[1].set_title("Training Time (s)", color=GOLD, fontsize=13, fontweight="bold")
    for b, v in zip(bars, times):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}",
                     ha="center", color=WHITE, fontsize=9)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(names, color=WHITE, rotation=20, ha="right")

    # Spike rate
    bars = axes[2].bar(range(5), srs, color=colors, **kw)
    axes[2].set_title("Spike Rate", color=GOLD, fontsize=13, fontweight="bold")
    for b, v in zip(bars, srs):
        axes[2].text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}",
                     ha="center", color=WHITE, fontsize=9)
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels(names, color=WHITE, rotation=20, ha="right")

    fig.suptitle("Benchmark Results -- All Neuron Models",
                 color=GOLD, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── 8. Decision Guide ─────────────────────────────────────────────────────────
@slide(6.0)
def make_decision():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.axis("off")
    ax.text(8, 8.6, "DECISION GUIDE", ha="center", va="top",
            fontsize=18, fontweight="bold", color=GOLD)
    hdrs = ["Neuron", "Speed", "Realism", "Best For", "Avoid When"]
    xs   = [0.5, 2.8, 4.3, 6.0, 9.0]
    ws   = [1.8, 1.2, 1.4, 2.5, 2.0]
    for ci, (h, x) in enumerate(zip(hdrs, xs)):
        ax.text(x+ws[ci]/2, 7.8, h, ha="center", va="center",
                fontsize=10, fontweight="bold", color=GOLD)
    rows = [
        ("IF",             "***", "*",    "Edge devices, baselines",      "Temporal dynamics"),
        ("LIF",            "**",   "**",   "General SNN tasks",            "High-firing regimes"),
        ("Adaptive LIF",   "**",   "**",   "Sensory processing",          "Min. latency required"),
        ("Izhikevich",     "*",    "***",  "Cognitive modeling",           "Real-time at scale"),
        ("Hodgkin-Huxley", "*",    "****", "Biological validation",        "Production inference"),
    ]
    row_c = [NC[n] for n in ["IF","LIF","Adaptive LIF","Izhikevich","Hodgkin-Huxley"]]
    for ri, (row, color) in enumerate(zip(rows, row_c)):
        y = 6.8 - ri*1.2
        bg = TEAL if ri%2==0 else NAVY
        ax.add_patch(plt.Rectangle((0.2, y-0.4), 15.6, 0.9, facecolor=bg))
        for ci, (cell, x, w) in enumerate(zip(row, xs, ws)):
            ax.text(x+w/2, y+0.06, cell, ha="center", va="center",
                    fontsize=10, color=WHITE,
                    fontweight="bold" if ci==0 else "normal")
    return fig


# ── 9. Conclusion ────────────────────────────────────────────────────────────
@slide(5.0)
def make_conclusion():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(TEAL); ax.set_facecolor(TEAL); ax.axis("off")
    ax.text(0.5, 0.80, "KEY FINDING", ha="center", va="center",
            transform=ax.transAxes, color=GOLD, fontsize=14, fontweight="bold")
    ax.text(0.5, 0.55,
            "No single neuron model dominates.\n"
            "The optimal choice depends on your\n"
            "task, hardware, and accuracy budget.",
            ha="center", va="center", transform=ax.transAxes,
            color=WHITE, fontsize=18, linespacing=1.7)
    for i, t in enumerate([
        "[BRAIN]  LIF is the strongest general baseline",
        "[BOLT]   Izhikevich offers richer dynamics at moderate cost",
        "[FLASK]  Hodgkin-Huxley is the gold-standard reference",
        "[CHART]  Benchmark everything -- never assume",
    ]):
        ax.text(0.5, 0.26 - i*0.09, t, ha="center", va="center",
                transform=ax.transAxes, color=WHITE, fontsize=13, alpha=0.9)
    ax.text(8, 0.08, "github.com/your-repo  |  Open-source + Reproducible",
            ha="center", va="center", color=DGREY, fontsize=11)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def generate(out="benchmark_demo.mp4"):
    print("[1/4] Training models...")
    from neuron_benchmark.training import train_model
    benchmarks = [
        ("IF",             "if"),
        ("LIF",            "lif"),
        ("Adaptive LIF",   "adaptive-lif"),
        ("Izhikevich",     "izhikevich"),
        ("Hodgkin-Huxley", "hodgkin-huxley"),
    ]
    results = []
    for name, key in benchmarks:
        r = train_model(key, samples=128, steps=12, epochs=20, seed=42)
        print(f"  {name}: acc={r['test_accuracy']:.3f}  sr={r['spike_rate']:.4f}  t={r['training_seconds']:.3f}s")
        results.append({"name": name, "accuracy": r["test_accuracy"],
                        "time_s": r["training_seconds"], "spike_rate": r["spike_rate"]})

    print("[2/4] Computing demo stats...")
    ru = train_model("lif", samples=128, steps=12, epochs=0,  seed=7)
    rt = train_model("lif", samples=128, steps=12, epochs=15, seed=7)
    print(f"  Untrained: acc={ru['test_accuracy']:.3f}  sr={ru['spike_rate']:.4f}")
    print(f"  Trained:   acc={rt['test_accuracy']:.3f}  sr={rt['spike_rate']:.4f}")

    print("[3/4] Building slides...")
    clips = [
        make_title(),
        make_question(),
        make_arch(),
        make_models(),
        make_traces(),
        make_demo(ru["test_accuracy"], rt["test_accuracy"],
                  ru["spike_rate"],   rt["spike_rate"]),
        make_results(results),
        make_decision(),
        make_conclusion(),
    ]

    print("[4/4] Rendering video...")
    joined = concatenate_videoclips(clips, method="compose")
    joined.write_videofile(
        out,
        fps=30,
        codec="h264_nvenc",
        preset="fast",
        bitrate="5000k",
        audio=False,
    )
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmark_demo.mp4"
    generate(out)
