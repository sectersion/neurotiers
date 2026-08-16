import argparse
import json
import time

import pandas as pd
import torch

from .data import synthetic_data
from .model import SNNClassifier
from .plots import generate_report, plot_accuracy_vs_runtime, plot_accuracy_vs_spike_rate, plot_membrane_traces, plot_raster, plot_pareto, plot_convergence, plot_leaderboard
from .robustness import run_robustness
from .training import train_model, run_experiments, NEURONS


def run(neuron="lif", samples=128, steps=12, seed=0):
    torch.manual_seed(seed)
    x, y = synthetic_data(samples=samples, seed=seed)
    model = SNNClassifier(neuron=neuron, steps=steps)
    start = time.perf_counter()
    with torch.no_grad():
        logits, spikes = model(x)
    elapsed = time.perf_counter() - start
    accuracy = (logits.argmax(1) == y).float().mean().item()
    return {"neuron": neuron, "samples": samples, "steps": steps,
            "accuracy": accuracy, "runtime_seconds": elapsed,
            "spike_rate": spikes.float().mean().item()}


def run_full_benchmark(samples=128, steps=12, seed=0):
    results = []
    for neuron in NEURONS:
        result = run(neuron=neuron, samples=samples, steps=steps, seed=seed)
        results.append(result)
    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a minimal SNN neuron benchmark")
    parser.add_argument("--neuron", choices=NEURONS, default="lif")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", action="store_true", help="train before measuring")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--experiment", action="store_true", help="run all neurons across all seeds")
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "fashion_mnist"], default="synthetic", help="dataset to use")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds for experiment mode")
    parser.add_argument("--robustness", action="store_true", help="run robustness experiments")
    parser.add_argument("--report", action="store_true", help="run full benchmark and generate plots")
    parser.add_argument("--output-dir", type=str, default="benchmark_report", help="output directory for report")
    parser.add_argument("--full-metrics", action="store_true", help="include energy estimates and efficiency metrics")
    args = parser.parse_args(argv)

    if args.report:
        experiment_results = []
        for neuron in NEURONS:
            result = run_experiments(neuron, seeds=args.seeds, samples=args.samples,
                                     steps=args.steps, epochs=args.epochs)
            experiment_results.append({
                "neuron": result["neuron"],
                "mean_test_accuracy": result["mean_test_accuracy"],
                "std_test_accuracy": result["std_test_accuracy"],
                "mean_training_seconds": result["mean_training_seconds"],
                "std_training_seconds": result["std_training_seconds"],
                "mean_spike_rate": result["mean_spike_rate"],
            })
        results_df = pd.DataFrame(experiment_results)
        convergence_neuron = args.neuron if args.neuron in NEURONS else NEURONS[0]
        figures, markdown_path = generate_report(
            results_df, args.output_dir,
            convergence_neuron=convergence_neuron,
            convergence_seeds=args.seeds,
            convergence_kwargs={"samples": args.samples, "steps": args.steps, "epochs": args.epochs},
            dataset=args.dataset
        )
        print(f"Report generated in: {args.output_dir}")
        for key, path in figures.items():
            print(f"  - {path}")
        print(f"  - {markdown_path}")
    elif args.robustness:
        result = run_robustness(args.neuron, args.samples, args.steps, args.epochs, args.seed)
        print(json.dumps(result, sort_keys=True))
    elif args.experiment:
        results = []
        for neuron in NEURONS:
            result = run_experiments(neuron, seeds=args.seeds, samples=args.samples,
                                     steps=args.steps, epochs=args.epochs, full_metrics=args.full_metrics)
            results.append(result)
        print(json.dumps(results, sort_keys=True))
    elif args.train:
        result = train_model(args.neuron, args.samples, args.steps, args.epochs, args.seed)
        print(json.dumps(result, sort_keys=True))
    else:
        result = run(args.neuron, args.samples, args.steps, args.seed)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
