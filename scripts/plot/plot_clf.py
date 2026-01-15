import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from clfextract.utils import (LAYER_MAP, MARKER_MAP, METRIC_MAP, MODELS,
                              MODELS_MAP, savefig)

sns.set_style("whitegrid")
colors = sns.color_palette("tab10", len(MODELS))
color_map = {model: color for model, color in zip(MODELS, colors)}


def create_lineplot(
    data_by_model, metric_name, output_file, y_label, y_min=0.5, y_max=1.05
):
    """
    Generic function for creating lineplots for any metric with confidence intervals

    Parameters:
    - data_by_model: Dictionary mapping models to lists of (layer, value) tuples
    - metric_name: Name of the metric for the filename
    - output_file: Full path to the output file
    - y_label: Label for the y-axis
    - y_min: Minimum value for y-axis
    - y_max: Maximum value for y-axis
    """
    plt.figure()

    for model, values in data_by_model.items():
        if not values:
            continue

        values.sort(key=lambda x: x[0])  # Sort by layer
        layers, test_values = zip(*values)
        fractions = [layer / LAYER_MAP[model] for layer in layers]

        # Group values by layer for confidence interval calculation
        layer_groups = {}
        for layer, value in values:
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(value)

        medians = []
        lower_bounds = []
        upper_bounds = []
        for layer in sorted(layer_groups.keys()):
            group_values = layer_groups[layer]
            median = np.median(group_values)
            # Calculate confidence interval for the median using bootstrapping
            bootstrapped_medians = [
                np.median(
                    np.random.choice(group_values, size=len(group_values), replace=True)
                )
                for _ in range(1000)
            ]
            lower_bound = np.percentile(bootstrapped_medians, 2.5)  # 2.5th percentile
            upper_bound = np.percentile(bootstrapped_medians, 97.5)  # 97.5th percentile

            medians.append(median)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)

        fractions = [layer / LAYER_MAP[model] for layer in sorted(layer_groups.keys())]
        # Filter out points at x=0 if needed
        if metric_name.startswith("attack"):
            filtered = [
                (fraction, median, lower_bound, upper_bound)
                for fraction, median, lower_bound, upper_bound in zip(
                    fractions, medians, lower_bounds, upper_bounds
                )
                if fraction != 0
            ]
            fractions, medians, lower_bounds, upper_bounds = zip(*filtered)

        plt.plot(
            fractions,
            medians,
            linestyle="solid",
            color=color_map[model],
            marker=MARKER_MAP[model],
            label=MODELS_MAP[model],
            markersize=4,
            linewidth=1,
        )

        plt.fill_between(
            fractions,
            lower_bounds,
            upper_bounds,
            color=color_map[model],
            alpha=0.2,
        )

    plt.xlabel("Normalized Candidate Size")
    plt.ylabel(y_label)
    plt.xlim(0, 1.05)
    plt.ylim(y_min, y_max)
    plt.legend()

    savefig(output_file)
    plt.close()


def extract_metric_data(results, metric, data_type="test"):
    """
    Extract data for a specific metric from results

    Parameters:
    - results: List of result dictionaries
    - metric: Metric to extract
    - data_type: 'test' or 'train'

    Returns:
    - Dictionary mapping models to dictionaries mapping sources to lists of (layer, value) tuples
    """
    model_source_data = {}

    for result in results:
        if result["model"] not in MODELS:
            continue

        model = result["model"]
        source = result["source"]
        layer = result["layer"]

        if model not in model_source_data:
            model_source_data[model] = {}

        if source not in model_source_data[model]:
            model_source_data[model][source] = []

        # Extract the appropriate value based on the metric and result structure

        if "cross_source" in result["layer_classifier"] and data_type == "cross_source":
            cross_source = result["layer_classifier"]["cross_source"]
            if True:
                fold_results = [
                    cross_source["fold_results"][i][metric]
                    for i in range(len(cross_source["fold_results"]))
                ]
                for fold_value in fold_results:
                    model_source_data[model][source].append((layer, fold_value))
        elif "attacks" in result["layer_classifier"] and data_type == "attack":
            if len(result["attacks"]) <= 1:
                continue
            attack_data = result["layer_classifier"]["attacks"]["gcg"]

            fold_results = [
                attack_data["fold_results"][i]["transfer"]["harmful"][metric]
                for i in range(len(attack_data["fold_results"]))
            ]
            for fold_value in fold_results:
                model_source_data[model][source].append((layer, fold_value))
        else:
            # Standard metric in train/test data
            values = result["layer_classifier"][data_type][metric]
            for fold_value in values:
                model_source_data[model][source].append((layer, fold_value))

    return model_source_data


def process_results(results, output_base_dir):
    """Main function to process all results and generate visualizations"""
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "cross_source"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "attacks"), exist_ok=True)

    metric = "agreement_llm_f1"
    sources = set(result["source"] for result in results)

    # Extract metric data
    metric_data = extract_metric_data(results, metric)

    # Generate plots for each source
    for source in sources:
        data_for_plot = {
            model: data[source] for model, data in metric_data.items() if source in data
        }

        if not any(data_for_plot.values()):
            continue

        metric_label = f"Test {METRIC_MAP[metric]}"
        output_file = os.path.join(output_base_dir, "benign", f"{metric}_{source}.pdf")

        create_lineplot(data_for_plot, metric, output_file, metric_label)

    # Process cross-source results
    cross_source_data = extract_metric_data(
        results, "agreement_llm_f1", data_type="cross_source"
    )
    sources = set(
        source for model_data in cross_source_data.values() for source in model_data
    )

    for source in sources:
        data_for_plot = {
            model: data[source]
            for model, data in cross_source_data.items()
            if source in data
        }
        output_file = os.path.join(
            output_base_dir, "cross_source", f"cross_source_{metric}_{source}.pdf"
        )
        create_lineplot(
            data_for_plot,
            "cross_source",
            output_file,
            f"Test {METRIC_MAP['agreement_llm_f1']}",
        )

    # Process attack results
    attack_data = extract_metric_data(results, "agreement_llm", data_type="attack")
    sources = set(
        source for model_data in attack_data.values() for source in model_data
    )

    for source in sources:
        data_for_plot = {
            model: data[source] for model, data in attack_data.items() if source in data
        }
        output_file = os.path.join(
            output_base_dir, "attacks", f"transfer_llm_clf_{source}.pdf"
        )
        create_lineplot(
            data_for_plot,
            "attack",
            output_file,
            "Transferability Rate",
            y_min=0,
            y_max=1.05,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for model results."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON file containing all results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="visualizations",
        help="Base directory for output visualizations",
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        default=["llama2", "qwen2", "gemma1", "granite"],
        help="Models to include in the visualizations",
    )
    args = parser.parse_args()

    # Filter models if needed
    for i in range(len(MODELS)):
        if MODELS[i] not in args.models:
            MODELS[i] = None

    # Load results
    all_results = json.load(open(args.input, "r"))

    # Process all results
    process_results(all_results, args.output)
