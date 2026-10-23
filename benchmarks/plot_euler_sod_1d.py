"""Plot density, velocity, and pressure from the Sod benchmark CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    required = ("x", "density", "velocity", "pressure")
    if any(key not in rows[0] for key in required):
        raise ValueError(f"CSV must contain columns: {', '.join(required)}")
    return {key: [float(row[key]) for row in rows] for key in required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="sod_results.csv")
    parser.add_argument("--output", default="sod_profiles.png")
    args = parser.parse_args()

    data = load_csv(args.input)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for ax, field, label in zip(
        axes,
        ("density", "velocity", "pressure"),
        (r"Density $\\rho$", r"Velocity $u$", r"Pressure $p$"),
    ):
        ax.plot(data["x"], data[field], linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x")
    fig.suptitle("1D Euler Sod shock tube — P0 / first-order finite-volume baseline")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Wrote {output} ({len(data['x'])} cells)")


if __name__ == "__main__":
    main()
