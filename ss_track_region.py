from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Headless server support
plt.switch_backend('Agg')

SS_MAP = {
    'H': 'Alpha-Helix', 'G': '3-10-Helix', 'I': 'Pi-Helix',
    'E': 'Beta-Strand', 'B': 'Beta-Bridge',
    'T': 'Turn', 'S': 'Bend', '~': 'Coil'
}

SS_COLORS = {
    'H': '#2ca02c', 'G': '#98df8a', 'I': '#bcbd22',
    'E': '#d62728', 'B': '#ff9896',
    'T': '#1f77b4', 'S': '#aec7e8', '~': '#7f7f7f'
}

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def parse_region_spec(spec: str) -> Tuple[str, Tuple[int, int]]:
    name, rng = spec.split(":", 1)
    start, end = map(int, rng.split("-"))
    return name, (start, end)

def process_variant(args_tuple):
    variant, filepath, regions = args_tuple
    if not filepath.exists(): return variant, None
    
    frames = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as h:
        for line in h:
            if not line or line[0] in ("#", "@"): continue
            frames.append(line.strip())
    
    if not frames: return variant, None
    actual_len = len(frames[0])
    
    res = {}
    for name, (start, end) in regions.items():
        s_idx, e_idx = start - 1, min(end, actual_len)
        storage = {code: [] for code in SS_MAP}
        for f in frames:
            reg = f[s_idx:e_idx]
            for code in SS_MAP:
                storage[code].append(reg.count(code)) # RAW RESIDUE COUNT
        res[name] = {c: np.array(v) for c, v in storage.items()}
    return variant, res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["WT", "DM", "G600D", "P601L", "T553K"])
    parser.add_argument("--region", action="append", type=parse_region_spec, required=True)
    parser.add_argument("--time-step-ps", type=float, default=10.0)
    parser.add_argument("--time-unit", default="ns")
    parser.add_argument("--cpus", type=int, default=5)
    args = parser.parse_args()

    outdir = Path("ss_residue_analysis")
    outdir.mkdir(exist_ok=True)
    regions = dict(args.region)

    tasks = [(v, Path(".") / v / "ss_bend_368_375.dat", regions) for v in args.variants]
    with Pool(args.cpus) as pool:
        all_data = {v: d for v, d in pool.map(process_variant, tasks) if d}

    v_list = [v for v in args.variants if v in all_data]

    for reg_name, (start, end) in regions.items():
        reg_len = end - start + 1
        reg_dir = outdir / reg_name
        reg_dir.mkdir(parents=True, exist_ok=True)

        for code, full_name in SS_MAP.items():
            # Only plot if there's significant data
            if not any(np.sum(all_data[v][reg_name][code]) > 0 for v in v_list):
                continue
            
            # --- 1. HISTOGRAM: Pure Residue Count vs Probability Density ---
            plt.figure(figsize=(8, 5))
            # Create integer bins centered on counts (e.g., -0.5 to 0.5 for count 0)
            bins = np.arange(-0.5, reg_len + 1.5, 1)
            for i, v in enumerate(v_list):
                plt.hist(all_data[v][reg_name][code], bins=bins, density=True, 
                         histtype='step', label=v, color=COLORS[i%len(COLORS)], lw=2)
            plt.title(f"{reg_name}: {full_name} ({code}) Count Distribution")
            plt.xlabel("Number of Residues"); plt.ylabel("Probability Density")
            plt.xticks(np.arange(0, reg_len + 1))
            plt.legend(); plt.tight_layout()
            plt.savefig(reg_dir / f"hist_count_{code}.png", dpi=300); plt.close()

            # --- 2. TIME SERIES: Residue Count ---
            plt.figure(figsize=(10, 4))
            for i, v in enumerate(v_list):
                y = all_data[v][reg_name][code]
                x = np.arange(len(y)) * args.time_step_ps / (1000.0 if args.time_unit == "ns" else 1.0)
                plt.plot(x, y, label=v, color=COLORS[i%len(COLORS)], alpha=0.6, lw=1)
            plt.title(f"{reg_name}: {full_name} Count vs Time")
            plt.xlabel(f"Time ({args.time_unit})"); plt.ylabel("Number of Residues")
            plt.ylim(-0.2, reg_len + 0.5); plt.legend(loc='upper right'); plt.tight_layout()
            plt.savefig(reg_dir / f"timeseries_count_{code}.png", dpi=300); plt.close()

            # --- 3. MEAN FRACTION: (Normalized for comparison) ---
            plt.figure(figsize=(7, 5))
            means = [np.mean(all_data[v][reg_name][code]) / reg_len for v in v_list]
            stds = [np.std(all_data[v][reg_name][code]) / reg_len for v in v_list]
            plt.bar(v_list, means, yerr=stds, capsize=5, color=COLORS[:len(v_list)], edgecolor='black', alpha=0.8)
            plt.title(f"{reg_name}: Mean {full_name} Fraction"); plt.ylabel("Fraction of Region"); plt.ylim(0, 1.1)
            plt.tight_layout(); plt.savefig(reg_dir / f"mean_fraction_{code}.png", dpi=300); plt.close()

        # --- 4. STACKED COMPOSITION: (Normalized) ---
        plt.figure(figsize=(9, 6))
        bottom = np.zeros(len(v_list))
        for code in ['H', 'G', 'I', 'E', 'B', 'T', 'S', '~']:
            m = np.array([np.mean(all_data[v][reg_name][code]) / reg_len for v in v_list])
            plt.bar(v_list, m, bottom=bottom, label=f"{code}: {SS_MAP[code]}", color=SS_COLORS[code])
            bottom += m
        plt.title(f"{reg_name}: Total Composition"); plt.ylabel("Fraction")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.ylim(0, 1); plt.tight_layout()
        plt.savefig(reg_dir / "stacked_composition.png", dpi=300); plt.close()

    print(f"Exhaustive residue-count analysis complete in {outdir.resolve()}")

if __name__ == "__main__":
    main()
