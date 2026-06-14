"""
plot_thresholds.py

Visualiza las distribuciones de EAR y MAR a partir del CSV generado
por calibratethresholds.py, para ayudar a elegir EAR_THRESH y MAR_THRESH.

Genera:
  - Histograma de EAR con línea de umbral actual
  - Histograma de MAR con línea de umbral actual
  - Serie temporal de EAR y MAR a lo largo del video

Uso:
    python plot_thresholds.py <csv_path> [--ear_thresh 0.21] [--mar_thresh 0.60]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Umbrales actuales definidos en roi_extraction.py
DEFAULT_EAR_THRESH = 0.21
DEFAULT_MAR_THRESH = 0.60


def load_csv(csv_path: Path) -> dict:
    frames, ears, mars = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            ears.append(float(row["ear"]))
            mars.append(float(row["mar"]))
    return {
        "frames": np.array(frames),
        "ears": np.array(ears),
        "mars": np.array(mars),
    }


def plot_all(data: dict, ear_thresh: float, mar_thresh: float, csv_name: str, output_dir: Path):
    frames = data["frames"]
    ears = data["ears"]
    mars = data["mars"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Calibración de umbrales — {csv_name}", fontsize=14, fontweight="bold")

    # --- Histograma EAR ---
    ax = axes[0, 0]
    ax.hist(ears, bins=60, color="#3b82f6", edgecolor="white", alpha=0.85)
    ax.axvline(ear_thresh, color="#ef4444", linewidth=2, linestyle="--",
               label=f"EAR_THRESH = {ear_thresh}")
    ax.set_title("Distribución de EAR")
    ax.set_xlabel("EAR")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # --- Histograma MAR ---
    ax = axes[0, 1]
    ax.hist(mars, bins=60, color="#8b5cf6", edgecolor="white", alpha=0.85)
    ax.axvline(mar_thresh, color="#ef4444", linewidth=2, linestyle="--",
               label=f"MAR_THRESH = {mar_thresh}")
    ax.set_title("Distribución de MAR")
    ax.set_xlabel("MAR")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # --- Serie temporal EAR ---
    ax = axes[1, 0]
    ax.plot(frames, ears, color="#3b82f6", linewidth=0.6, alpha=0.8)
    ax.axhline(ear_thresh, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"EAR_THRESH = {ear_thresh}")
    ax.fill_between(frames, 0, ears, where=(ears < ear_thresh),
                     color="#ef4444", alpha=0.2, label="Microsueño (EAR < umbral)")
    ax.set_title("EAR a lo largo del video")
    ax.set_xlabel("Frame")
    ax.set_ylabel("EAR")
    ax.legend(fontsize=8)

    # --- Serie temporal MAR ---
    ax = axes[1, 1]
    ax.plot(frames, mars, color="#8b5cf6", linewidth=0.6, alpha=0.8)
    ax.axhline(mar_thresh, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"MAR_THRESH = {mar_thresh}")
    ax.fill_between(frames, 0, mars, where=(mars > mar_thresh),
                     color="#ef4444", alpha=0.2, label="Bostezo (MAR > umbral)")
    ax.set_title("MAR a lo largo del video")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MAR")
    ax.legend(fontsize=8)

    plt.tight_layout()

    # Guardar imagen
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_name.replace('.csv', '')}_plots.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"Gráficas guardadas en: {out_path}")

    # Estadísticas resumen
    print(f"\n{'='*45}")
    print(f"  Estadísticas — {csv_name}")
    print(f"{'='*45}")
    print(f"  Total frames analizados: {len(ears)}")
    print(f"")
    print(f"  EAR  →  min: {ears.min():.4f}  max: {ears.max():.4f}")
    print(f"          media: {ears.mean():.4f}  mediana: {np.median(ears):.4f}")
    print(f"          std: {ears.std():.4f}")
    print(f"          Frames < EAR_THRESH ({ear_thresh}): "
          f"{(ears < ear_thresh).sum()} ({(ears < ear_thresh).mean()*100:.1f}%)")
    print(f"")
    print(f"  MAR  →  min: {mars.min():.4f}  max: {mars.max():.4f}")
    print(f"          media: {mars.mean():.4f}  mediana: {np.median(mars):.4f}")
    print(f"          std: {mars.std():.4f}")
    print(f"          Frames > MAR_THRESH ({mar_thresh}): "
          f"{(mars > mar_thresh).sum()} ({(mars > mar_thresh).mean()*100:.1f}%)")
    print(f"{'='*45}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizar distribuciones de EAR y MAR")
    parser.add_argument("csv_path", type=str, help="Ruta al CSV generado por calibratethresholds.py")
    parser.add_argument("--ear_thresh", type=float, default=DEFAULT_EAR_THRESH,
                        help=f"Umbral EAR a marcar (default: {DEFAULT_EAR_THRESH})")
    parser.add_argument("--mar_thresh", type=float, default=DEFAULT_MAR_THRESH,
                        help=f"Umbral MAR a marcar (default: {DEFAULT_MAR_THRESH})")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: No se encontró el archivo {csv_path}")
        sys.exit(1)

    data = load_csv(csv_path)
    output_dir = csv_path.parent
    plot_all(data, args.ear_thresh, args.mar_thresh, csv_path.name, output_dir)
