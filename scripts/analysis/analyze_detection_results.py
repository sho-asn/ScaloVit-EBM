import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot_energy_distribution(df, output_filename):
    """
    Plots the distribution of energy scores for normal and anomalous data using KDE.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    plt.figure(figsize=(10, 6))
    
    # Define the color palette
    palette = {0: 'steelblue', 1: 'red'}

    # sns.kdeplot with 'hue' creates the plot.
    ax = sns.kdeplot(data=df, x='score', hue='ground_truth', fill=False, common_norm=False, palette=palette, legend=False)
    
    plt.title('Energy Score Distribution')
    plt.xlabel('Energy Score')
    plt.ylabel('Density')
    plt.grid(True)

    plt.savefig(output_filename)
    print(f"Saved refined energy distribution plot to {output_filename}")
    plt.close()

def plot_energy_over_time(df, output_filename):
    """
    Plots energy scores over time, highlighting anomaly regions.
    """
    with plt.rc_context({'font.size': 14,
                         'axes.labelsize': 16,
                         'axes.titlesize': 18,
                         'xtick.labelsize': 14,
                         'ytick.labelsize': 14,
                         'legend.fontsize': 14}):
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot energy score
        ax.plot(df.index, df['score'], label='Energy Score', color='steelblue', linewidth=1.5)

        # Get y-limits to span the full plot height
        ymin, ymax = ax.get_ylim()

        # Highlight ground truth anomalies
        ax.fill_between(df.index, ymin, ymax, 
                         where=(df['ground_truth'] == 1), 
                         color='red', alpha=0.3, label='Ground Truth Anomaly', interpolate=True)

        # Add vertical lines for ground truth anomaly boundaries
        gt_changes = df['ground_truth'].diff()
        for idx in df.index[gt_changes == 1]:
            ax.axvline(idx, color='red', linestyle='--', linewidth=1, label='_nolegend_')
        for idx in df.index[gt_changes == -1]:
            ax.axvline(idx, color='red', linestyle='--', linewidth=1, label='_nolegend_')

        ax.set_title('Energy Score Over Time')
        ax.set_xlabel('Index')
        ax.set_ylabel('Energy Score')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Reset y-limits to maintain the original plot range
        ax.set_ylim(ymin, ymax)
        
        plt.savefig(output_filename)
        print(f"Saved refined energy over time plot to {output_filename}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze and plot detection results.')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: File not found at {args.csv_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    df.index.name = 'index'

    output_dir = os.path.dirname(args.csv_path)
    base_filename = os.path.splitext(os.path.basename(args.csv_path))[0]
    dist_plot_filename = os.path.join(output_dir, f"{base_filename}_energy_distribution.png")
    time_plot_filename = os.path.join(output_dir, f"{base_filename}_energy_over_time.png")

    plot_energy_distribution(df, dist_plot_filename)
    plot_energy_over_time(df, time_plot_filename)

if __name__ == '__main__':
    main()