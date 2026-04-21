from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Server-safe backend
plt.switch_backend('Agg')

DEFAULT_VARIANTS = ["WT", "DM", "G600D", "P601L", "T553K"]
# Consistent color palette for all your plots
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Calculates mean of angles accounting for periodic boundary (-180 to 180)."""
    if len(angles_deg) == 0: return float("nan")
    radians = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return ((angle + 180.0) % 360.0) - 180.0

def parse_residue_token(token: str) -> int | None:
    """Parses 'RES-ID' string from GROMACS rama.xvg."""
    if "-" not in token: return None
    tail = token.split("-")[-1]
    return int(tail) if tail.isdigit() else None

def load_rama_for_residues(filepath: Path, residues_of_interest: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """Loads and organizes phi/psi data by residue number."""
    data = {r: {"frame": [], "phi": [], "psi": []} for r in residues_of_interest}
    current_frame, prev_res = 0, None

    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "@")): continue
            parts = line.split()
            if len(parts) < 3: continue
            
            try:
                phi, psi = float(parts[0]), float(parts[1])
                residue = parse_residue_token(parts[2])
                if residue is None: continue
                
                # Detect new frame by residue number reset
                if prev_res is not None and residue < prev_res:
                    current_frame += 1
                prev_res = residue

                if residue in data:
                    data[residue]["frame"].append(current_frame)
                    data[residue]["phi"].append(phi)
                    data[residue]["psi"].append(psi)
            except ValueError: continue

    return {r: {k: np.array(v) for k, v in fields.items()} for r, fields in data.items()}

def plot_histogram_step(residue, angle_name, variant_data, outpath):
    """Generates clean overlapping histograms using 'step' outlines."""
    plt.figure(figsize=(10, 6))
    edges = np.linspace(-180, 180, 72) # 5-degree bins
    
    for i, (variant, values) in enumerate(variant_data.items()):
        if len(values) == 0: continue
        color = COLORS[i % len(COLORS)]
        plt.hist(values, bins=edges, density=True, histtype='step', 
                 linewidth=2, label=variant, color=color)
        plt.hist(values, bins=edges, density=True, histtype='stepfilled', 
                 alpha=0.1, color=color)

    plt.xlabel(f"{angle_name} angle (degrees)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"Distribution of {angle_name}: Residue {residue}", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_mean_bar_with_err(residue, angle_name, variant_data, outpath):
    """Generates bar plots with circular means and standard deviation error bars."""
    plt.figure(figsize=(8, 6))
    variants = list(variant_data.keys())
    # Note: Using circular mean for the height, standard deviation for the error
    means = [circular_mean_deg(variant_data[v]) for v in variants]
    stds = [np.std(variant_data[v]) for v in variants]

    plt.bar(variants, means, yerr=stds, capsize=7, color=COLORS[:len(variants)], 
            edgecolor='black', alpha=0.8)
    plt.ylabel(f"Mean {angle_name} (degrees)", fontsize=12)
    plt.title(f"Average {angle_name} Stability: Residue {residue}", fontsize=14)
    plt.ylim(-180, 180)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Professional Rama Analysis Suite")
    parser.add_argument("--residues", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--time-step-ps", type=float, default=1.0)
    parser.add_argument("--time-unit", choices=["frame", "ps", "ns"], default="frame")
    args = parser.parse_args()
    
    outdir = Path("rama_analysis_pro")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    all_data = {}
    for variant in args.variants:
        path = Path(".") / variant / "rama.xvg"
        if path.exists():
            all_data[variant] = load_rama_for_residues(path, args.residues)
        else:
            print(f"Skipping {variant}: File not found at {path}")

    # 2. Process each residue
    for res in args.residues:
        res_dir = outdir / f"residue_{res}"
        res_dir.mkdir(exist_ok=True)
        
        for angle in ["phi", "psi"]:
            var_angles = {v: all_data[v][res][angle] for v in all_data if res in all_data[v]}
            
            # Histogram Plots
            plot_histogram_step(res, angle, var_angles, res_dir / f"{angle}_histogram_clear.png")
            
            # Bar Plots with Error Bars
            plot_mean_bar_with_err(res, angle, var_angles, res_dir / f"{angle}_barplot.png")

            # Time Series Plots
            plt.figure(figsize=(10, 5))
            for i, (v, y) in enumerate(var_angles.items()):
                frames = all_data[v][res]["frame"]
                # Time conversion
                time_val = frames * args.time_step_ps
                if args.time_unit == "ns": time_val /= 1000.0
                
                plt.plot(time_val, y, color=COLORS[i % len(COLORS)], label=v, alpha=0.7, linewidth=1)
            
            plt.title(f"Evolution of {angle}: Residue {res}")
            plt.ylabel(f"{angle} (degrees)")
            plt.xlabel(f"Time ({args.time_unit})")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(res_dir / f"{angle}_timeseries.png", dpi=300)
            plt.close()

        # Scatter Plots (Classic Ramachandran)
        for i, variant in enumerate(args.variants):
            if variant in all_data and res in all_data[variant]:
                d = all_data[variant][res]
                plt.figure(figsize=(6, 6))
                plt.scatter(d["phi"], d["psi"], s=5, alpha=0.3, color=COLORS[i % len(COLORS)])
                plt.xlim(-180, 180); plt.ylim(-180, 180)
                plt.axhline(0, color='black', lw=0.5); plt.axvline(0, color='black', lw=0.5)
                plt.title(f"Rama Plot: {variant} | Res {res}")
                plt.xlabel("Phi (degrees)"); plt.ylabel("Psi (degrees)")
                plt.savefig(res_dir / f"rama_scatter_{variant}.png", dpi=300)
                plt.close()

    print(f"\nSuccess! Results written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
