import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clfextract.utils import MODELS, MODELS_MAP, REVERSE_MODELS_MAP, savefig

# Set up color palette and seaborn style
palette = sns.color_palette("tab10", len(MODELS_MAP.values()))
model_palette = {model: palette[i] for i, model in enumerate(MODELS_MAP.values())}
sns.set_style("whitegrid")


def get_args():
    """
    Parses command-line arguments for input file, output directory,
    models to include, and dataset name.
    """
    parser = argparse.ArgumentParser(description="Visualize model performance metrics")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the JSON file containing the performance results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Directory to save output plots and statistics",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["llama2"],
        help="Models to include in the visualization",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        default="",
        help="Name of the dataset (for filtering data and plot titles)",
    )
    return parser.parse_args()


def load_json_data(json_file, subset=""):
    """
    Loads performance data from a JSON file, filtering by dataset subset.
    Extracts time, vram_low, vram_high, and relevant metadata.
    Filters out records where vram_low or vram_high are 0.
    Separates baseline GCG data (no 'clf' key) from CLF-specific data.
    """
    all_records = []
    baseline_time = []
    baseline_vram_high = []

    with open(json_file, "r") as f:
        all_data = json.load(f)

    for data in all_data:
        metadata = data["metadata"]
        # Skip records if a subset is specified and doesn't match
        if subset and subset not in metadata["dataset"]:
            continue

        # Check if 'clf' key exists in metadata for CLF-specific data
        if "clf" in metadata:
            clf_layer = metadata["clf"]["layer"]
            for record in data["records"]:
                time = record.get("time")
                vram_low = record.get("vram_low")
                vram_high = record.get("vram_high")

                # Remove any entry that is 0
                if time is None or vram_high is None or time == 0 or vram_high == 0:
                    continue

                record_with_metadata = {
                    "model": (
                        metadata["model"]
                        if "granite" not in metadata["model"]
                        else "granite"
                    ),
                    "dataset": metadata["dataset"],
                    "layer": clf_layer,
                    "time": time,
                    "vram_low": vram_low,
                    "vram_high": vram_high,
                }
                all_records.append(record_with_metadata)
        else:
            # This is a baseline GCG record (no 'clf' key)
            for record in data["records"]:
                time = record.get("time")
                vram_high = record.get("vram_high")

                # Remove any entry that is 0
                if time is None or vram_high is None or time == 0 or vram_high == 0:
                    continue

                baseline_time.append(time)
                baseline_vram_high.append(vram_high)

    return pd.DataFrame(all_records), baseline_time, baseline_vram_high


def visualize_performance_metrics(
    df, baseline_time, baseline_vram, output_dir="results", dataset_name=""
):
    """
    Generates a dual-axis line plot for average runtime and VRAM usage per model,
    showing confidence intervals based on layer/candidate size.
    Also adds horizontal lines for the baseline GCG (time and VRAM) with std.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert model names using MODELS_MAP for consistent plotting
    df["model_mapped"] = df["model"].map(MODELS_MAP)
    # Convert VRAM to GB
    df["vram_high_gb"] = df["vram_high"] / (1024**3)  # Convert B to GB

    # Average models and datasets
    df_avg = (
        df.groupby(["layer"])
        .agg(  # Group by layer only to average across models and datasets
            avg_time=("time", "mean"),
            std_time=("time", "std"),
            avg_vram=("vram_high_gb", "mean"),
            std_vram=("vram_high_gb", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(14, 8))
    ax1 = plt.gca()  # Get current axes to align grids

    # Plot Time
    (line_time,) = ax1.plot(  # <--- ADD A COMMA HERE TO UNPACK THE LIST
        df_avg["layer"],
        df_avg["avg_time"],
        color="black",
        marker="o",  # Use circles
        linewidth=2,
        markersize=6,
        label="Average Runtime",
    )
    ax1.fill_between(
        df_avg["layer"],
        df_avg["avg_time"] - df_avg["std_time"],
        df_avg["avg_time"] + df_avg["std_time"],
        color="black",
        alpha=0.15,
    )

    ax1.set_xlabel("Candidate Size")
    ax1.set_ylabel("Runtime (s)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Set xticks every 5
    max_layer = df_avg["layer"].max()
    ax1.set_xticks(np.arange(5, max_layer + 1, 5))

    # Add baseline horizontal line for time
    baseline_time_line = None  # Initialize outside the if block
    if baseline_time:
        time_mean = np.mean(baseline_time)
        baseline_time_line = ax1.axhline(
            time_mean,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Baseline GCG (Time) Mean",
        )  # Label for the legend

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot VRAM
    (line_vram,) = ax2.plot(
        df_avg["layer"],
        df_avg["avg_vram"],
        color="gray",
        marker="o",  # Use circles
        linewidth=2,
        markersize=6,
        label="Average VRAM Usage",
    )
    ax2.fill_between(
        df_avg["layer"],
        df_avg["avg_vram"] - df_avg["std_vram"],
        df_avg["avg_vram"] + df_avg["std_vram"],
        color="gray",
        alpha=0.15,
    )

    ax2.set_ylabel("VRAM Usage (GB)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(bottom=0)  # Set y-axis to start at 0 for VRAM

    # Add baseline horizontal line for vram_high
    baseline_vram_line = None  # Initialize outside the if block
    if baseline_vram:
        vram_mean = np.mean(baseline_vram) / (1024**3)  # Convert baseline VRAM to GB
        baseline_vram_line = ax2.axhline(
            vram_mean,
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Baseline GCG (VRAM) Mean",
        )  # Label for the legend

    # Combine legends from both axes
    handles = []
    labels = []

    # Add the averaged lines to the legend
    handles.append(line_time)
    labels.append("Runtime (Avg.)")  # Use the correct label
    handles.append(line_vram)
    labels.append("VRAM Usage (Avg.)")  # Use the correct label

    # Add baseline lines to legend if they exist
    if baseline_time_line:
        handles.append(baseline_time_line)
        labels.append("Baseline GCG")  # Use the correct label

    # Add the legend to ax1
    ax1.legend(
        handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(0.0, 0.95)
    )  # Position top left

    # Remove the legend
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.xticks(rotation=45, ha="right")

    # Align the grids
    ax1.grid(True, which="both", linestyle="-", linewidth=0.5, color="lightgray")
    ax2.grid(False)  # Grid removed as requested earlier

    plt.tight_layout()
    savefig(os.path.join(output_dir, f"efficiency_average.pdf"))


if __name__ == "__main__":
    args = get_args()

    # Filter MODELS list based on command-line arguments
    allowed_models = [model for model in MODELS if model in args.models]
    if not allowed_models:
        print("No valid models selected. Please check your --models argument.")
        exit()

    df, baseline_time, baseline_vram_high = load_json_data(
        args.input, subset=args.dataset_name
    )

    # Filter DataFrame to include only specified models
    df = df[df["model"].isin(allowed_models)].copy()
    # Handle the case where 'granite' might be in model names but not directly in MODELS
    df["model"] = df["model"].apply(lambda x: "granite" if "granite" in x else x)

    # Check if DataFrame is empty after filtering
    if df.empty:
        print(
            "No valid CLF data found after filtering. Only baseline data might be available."
        )

    # Visualize the performance metrics with the dual-axis plot, passing the baseline data
    visualize_performance_metrics(
        df,
        baseline_time,
        baseline_vram_high,
        args.output,
        dataset_name=args.dataset_name,
    )
