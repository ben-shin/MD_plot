import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Force non-interactive backend for server environments
plt.switch_backend('Agg')

BASE_DIR = Path(".")
VARIANTS = ["WT", "DM", "G600D", "P601L", "T553K"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# will turn these into flags

def load_xvg(filepath: Path) -> np.ndarray:
    """Standard GROMACS XVG parser skipping headers."""
    rows = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=float)

def moving_average(data, window_size=50):
    """Smooths data for the time-series plot."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    parser = argparse.ArgumentParser(description="Professional RMSD Analysis Suite")
    parser.add_argument("-f", "--file", required=True, help="RMSD filename (e.g., rmsd_beta_548_554.xvg)")
    args = parser.parse_args()

    out_prefix = Path(args.file).stem
    data_dict = {}

    # 1. Load Data
    for variant in VARIANTS:
        path = BASE_DIR / variant / args.file
        if path.exists():
            data_dict[variant] = load_xvg(path)
        else:
            print(f"Skipping {variant}: File not found.")

    if not data_dict:
        print("No data loaded. Check filenames.")
        return

    # --- PLOT 1: TIME SERIES (With Smoothing) ---
    plt.figure(figsize=(12, 6))
    for i, (variant, data) in enumerate(data_dict.items()):
        time_ns = data[:, 0] / 1000.0
        rmsd_nm = data[:, 1]
        
        # Plot raw data faintly
        plt.plot(time_ns, rmsd_nm, color=COLORS[i], alpha=0.2, linewidth=0.5)
        # Plot moving average for clarity
        smoothed = moving_average(rmsd_nm)
        plt.plot(time_ns[:len(smoothed)], smoothed, color=COLORS[i], label=variant, linewidth=2)

  # will add flags for these too
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("RMSD (nm)", fontsize=12)
    plt.title(f"RMSD Trajectory: {out_prefix} (Smoothed)", fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_timeseries.png", dpi=300)

    # --- PLOT 2: CLEAR HISTOGRAMS (Step + Density) ---
    plt.figure(figsize=(10, 6))
    for i, (variant, data) in enumerate(data_dict.items()):
        vals = data[:, 1]
        
        # Clean Outline (Step)
        plt.hist(vals, bins=60, density=True, histtype='step', 
                 color=COLORS[i], linewidth=2.5, label=variant)
        
        # Optional: Smooth KDE Line
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(min(vals), max(vals), 200)
            plt.plot(x_range, kde(x_range), color=COLORS[i], linestyle='--', alpha=0.5)
        except:
            pass # Fallback if data is too sparse for KDE

  # will add flags
    plt.xlabel("Interdomain Distance (nm)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"Interdomain Distance Distribution Overlap: {out_prefix}", fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_histogram_clear.png", dpi=300)

    # --- PLOT 3: BAR CHART (Comparison) ---
    variants_present = list(data_dict.keys())
    means = [np.mean(data_dict[v][:, 1]) for v in variants_present]
    stds = [np.std(data_dict[v][:, 1]) for v in variants_present]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(variants_present, means, yerr=stds, capsize=7, 
                   color=COLORS[:len(variants_present)], edgecolor='black', alpha=0.8)
    
    plt.ylabel("Mean Interdomain Distance (nm)", fontsize=12)
    plt.title(f"Average Stability Comparison: {out_prefix}", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_barplot.png", dpi=300)

    # 4. Export Stats CSV
    with open(f"{out_prefix}_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Mean", "StdDev", "Min", "Max"])
        for v in variants_present:
            d = data_dict[v][:, 1]
            writer.writerow([v, np.mean(d), np.std(d), np.min(d), np.max(d)])

    print(f"Success! Generated: \n - {out_prefix}_timeseries.png \n - {out_prefix}_histogram_clear.png \n - {out_prefix}_barplot.png \n - {out_prefix}_stats.csv")

if __name__ == "__main__":
    main()
