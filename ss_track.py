from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Server-safe backend
plt.switch_backend('Agg')

BASE_DIR = Path(".")
VARIANTS = ["WT", "DM", "G600D", "P601L", "T553K"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

SS_LABELS = [
    "Loops", "Breaks", "Bends", "Turns", "PP-II Helices",
    "pi-Helices", "3-10 Helices", "B-Strands", "B-Bridges", "Alpha-Helices"
]

def load_xvg_ss(filepath: Path) -> np.ndarray:
    rows = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "@")):
                rows.append([float(x) for x in line.split()])
    return np.array(rows)

def main():
    parser = argparse.ArgumentParser(description="Analyze Secondary Structure Evolution.")
    parser.add_argument("-f", "--file", required=True, help="SS count filename")
    args = parser.parse_args()

    out_prefix = Path(args.file).stem
    all_data: Dict[str, np.ndarray] = {}
    mean_fractions: Dict[str, np.ndarray] = {}

    # 1. Load Data and Individual Plots
    for variant in VARIANTS:
        filepath = BASE_DIR / variant / args.file
        if not filepath.exists():
            continue

        data = load_xvg_ss(filepath)
        time_ns = data[:, 0] / 1000.0
        counts = data[:, 1:]
        all_data[variant] = counts # Store raw counts for histograms
        
        total_res = np.sum(counts[0, :])
        fractions = counts / total_res
        mean_fractions[variant] = np.mean(fractions, axis=0)

        # Individual Stacked Area Plots
        plt.figure(figsize=(10, 5))
        plt.stackplot(time_ns, fractions.T, labels=SS_LABELS, alpha=0.8)
        plt.title(f"SS Composition: {variant} ({out_prefix})")
        plt.xlabel("Time (ns)")
        plt.ylabel("Fraction of Residues")
        plt.ylim(0, 1)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        plt.tight_layout()
        plt.savefig(f"{variant}_{out_prefix}_evolution.png", dpi=300)
        plt.close()

    if not all_data:
        print("No data found.")
        return

    # 2. Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    variants_found = list(mean_fractions.keys())
    bottom = np.zeros(len(variants_found))
    for i, label in enumerate(SS_LABELS):
        values = np.array([mean_fractions[v][i] for v in variants_found])
        plt.bar(variants_found, values, bottom=bottom, label=label)
        bottom += values
    plt.title(f"Average SS Comparison: {out_prefix}")
    plt.ylabel("Fractional Occupancy")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_comparison_bar.png", dpi=300)
    plt.close()

    # 3. GRID OF HISTOGRAMS (New Section)
    # Creating a 2x5 grid for the 10 SS types
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=False)
    axes = axes.flatten()

    for i, ss_type in enumerate(SS_LABELS):
        ax = axes[i]
        for j, variant in enumerate(variants_found):
            # Plotting the count distribution for this specific SS type
            ss_counts = all_data[variant][:, i]
            
            # Use 'step' to avoid visual clutter in overlaps
            ax.hist(ss_counts, bins=15, histtype='step', linewidth=2, 
                    label=variant, color=COLORS[j], density=True)
        
        ax.set_title(ss_type, fontweight='bold')
        ax.set_xlabel("Residue Count")
        ax.set_ylabel("Probability Density")
        if i == 0: # Only one legend needed for the grid
            ax.legend(fontsize='x-small')

    plt.suptitle(f"Secondary Structure Distributions: {out_prefix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{out_prefix}_dist_grid.png", dpi=300)
    plt.close()
    
    print(f"Analysis complete. Generated evolution, bar, and grid distribution plots.")

if __name__ == "__main__":
    main()
