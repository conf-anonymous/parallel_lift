#!/usr/bin/env python3
"""
Parallel Lift: Benchmark Result Visualization
Generates publication-ready figures from CSV benchmark data.

Usage:
    python plot_results.py

Requires: matplotlib, pandas
    pip install matplotlib pandas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style configuration for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors
CPU_COLOR = '#E24A33'  # Red-orange
GPU_COLOR = '#348ABD'  # Blue
SPEEDUP_COLOR = '#467821'  # Green


def plot_size_scaling(df: pd.DataFrame, output_path: str = 'figure_a_size_scaling.png'):
    """
    Figure A: Time vs Matrix Size
    Shows CPU exact vs GPU+CRT exact as matrix size increases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Execution time (log scale)
    ax1.plot(df['matrix_size'], df['cpu_time_s'], 'o-',
             color=CPU_COLOR, linewidth=2, markersize=8, label='CPU (Bareiss)')
    ax1.plot(df['matrix_size'], df['gpu_time_s'], 's-',
             color=GPU_COLOR, linewidth=2, markersize=8, label='GPU+CRT')

    ax1.set_yscale('log')
    ax1.set_xlabel('Matrix Size (n × n)')
    ax1.set_ylabel('Time (seconds, log scale)')
    ax1.set_title('Execution Time vs Matrix Size')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add crossover annotation
    crossover_idx = (df['speedup'] > 1).idxmax()
    if crossover_idx > 0:
        ax1.axvline(x=df.loc[crossover_idx, 'matrix_size'],
                    color='gray', linestyle='--', alpha=0.5)
        ax1.annotate('Crossover',
                     xy=(df.loc[crossover_idx, 'matrix_size'],
                         df.loc[crossover_idx, 'cpu_time_s']),
                     xytext=(10, 20), textcoords='offset points',
                     fontsize=10, color='gray')

    # Right panel: Speedup
    ax2.bar(df['matrix_size'], df['speedup'], color=SPEEDUP_COLOR, alpha=0.7, width=12)
    ax2.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Matrix Size (n × n)')
    ax2.set_ylabel('Speedup (CPU time / GPU time)')
    ax2.set_title('GPU Speedup vs Matrix Size')

    # Add speedup labels on bars
    for i, (size, speedup) in enumerate(zip(df['matrix_size'], df['speedup'])):
        ax2.annotate(f'{speedup:.2f}×',
                     xy=(size, speedup),
                     ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_entry_magnitude(df: pd.DataFrame, output_path: str = 'figure_b_entry_magnitude.png'):
    """
    Figure B: Time vs Entry Magnitude (bits)
    Shows how CPU time scales with integer size while GPU stays nearly flat.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Execution time
    ax1.plot(df['entry_bits'], df['cpu_time_s'], 'o-',
             color=CPU_COLOR, linewidth=2, markersize=10, label='CPU (Bareiss)')
    ax1.plot(df['entry_bits'], df['gpu_time_s'], 's-',
             color=GPU_COLOR, linewidth=2, markersize=10, label='GPU+CRT')

    ax1.set_xlabel('Entry Magnitude (bits)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time vs Entry Bit-Width (n=96)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add annotation explaining the key insight
    ax1.annotate('GPU time nearly constant\nas BigInt pressure grows',
                 xy=(27, df['gpu_time_s'].mean()),
                 xytext=(22, 1.5),
                 fontsize=10, style='italic',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # Right panel: Speedup with determinant bits annotation
    ax2_main = ax2
    ax2_main.bar(df['entry_bits'], df['speedup'], color=SPEEDUP_COLOR, alpha=0.7, width=2.5)
    ax2_main.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    ax2_main.set_xlabel('Entry Magnitude (bits)')
    ax2_main.set_ylabel('Speedup (CPU time / GPU time)', color=SPEEDUP_COLOR)
    ax2_main.tick_params(axis='y', labelcolor=SPEEDUP_COLOR)
    ax2_main.set_title('Speedup vs Entry Magnitude')

    # Add speedup labels
    for bits, speedup in zip(df['entry_bits'], df['speedup']):
        ax2_main.annotate(f'{speedup:.2f}×',
                          xy=(bits, speedup),
                          ha='center', va='bottom',
                          fontsize=10, fontweight='bold')

    # Secondary y-axis for determinant bits
    ax2_det = ax2_main.twinx()
    ax2_det.plot(df['entry_bits'], df['determinant_bits'], 'D--',
                 color='purple', alpha=0.7, markersize=6, label='Det bits')
    ax2_det.set_ylabel('Determinant Size (bits)', color='purple')
    ax2_det.tick_params(axis='y', labelcolor='purple')

    ax2_main.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_stability(df: pd.DataFrame, output_path: str = 'figure_c_stability.png'):
    """
    Figure C: Stability Analysis
    Shows variance across multiple runs with error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    # CPU bars with error bars
    cpu_bars = ax.bar(x - width/2, df['cpu_mean'], width,
                      yerr=df['cpu_stddev'], capsize=5,
                      label='CPU (Bareiss)', color=CPU_COLOR, alpha=0.8)

    # GPU bars with error bars
    gpu_bars = ax.bar(x + width/2, df['gpu_mean'], width,
                      yerr=df['gpu_stddev'], capsize=5,
                      label='GPU+CRT', color=GPU_COLOR, alpha=0.8)

    ax.set_xlabel('Matrix Size (n × n)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time with Variance (5 runs each)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in df['matrix_size']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, (cpu, gpu, speedup) in enumerate(zip(df['cpu_mean'], df['gpu_mean'], df['speedup_mean'])):
        ax.annotate(f'{speedup:.2f}×',
                    xy=(i, max(cpu, gpu) + df['cpu_stddev'].iloc[i] + 0.2),
                    ha='center', fontsize=11, fontweight='bold', color=SPEEDUP_COLOR)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_combined_summary(scaling_df: pd.DataFrame, entry_df: pd.DataFrame,
                          output_path: str = 'figure_summary.png'):
    """
    Combined summary figure for presentations/papers.
    """
    fig = plt.figure(figsize=(16, 10))

    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Panel 1: Size scaling - time
    ax1.plot(scaling_df['matrix_size'], scaling_df['cpu_time_s'], 'o-',
             color=CPU_COLOR, linewidth=2, markersize=6, label='CPU')
    ax1.plot(scaling_df['matrix_size'], scaling_df['gpu_time_s'], 's-',
             color=GPU_COLOR, linewidth=2, markersize=6, label='GPU+CRT')
    ax1.set_yscale('log')
    ax1.set_xlabel('Matrix Size (n)')
    ax1.set_ylabel('Time (s, log)')
    ax1.set_title('(A) Time vs Size')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Size scaling - speedup
    ax2.plot(scaling_df['matrix_size'], scaling_df['speedup'], 'o-',
             color=SPEEDUP_COLOR, linewidth=2, markersize=8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(scaling_df['matrix_size'], 1, scaling_df['speedup'],
                     where=(scaling_df['speedup'] > 1), alpha=0.3, color=SPEEDUP_COLOR)
    ax2.set_xlabel('Matrix Size (n)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('(B) Speedup vs Size')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Entry magnitude - time
    ax3.plot(entry_df['entry_bits'], entry_df['cpu_time_s'], 'o-',
             color=CPU_COLOR, linewidth=2, markersize=8, label='CPU')
    ax3.plot(entry_df['entry_bits'], entry_df['gpu_time_s'], 's-',
             color=GPU_COLOR, linewidth=2, markersize=8, label='GPU+CRT')
    ax3.set_xlabel('Entry Bits')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('(C) Time vs Entry Magnitude (n=96)')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Entry magnitude - speedup
    ax4.plot(entry_df['entry_bits'], entry_df['speedup'], 'o-',
             color=SPEEDUP_COLOR, linewidth=2, markersize=8)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(entry_df['entry_bits'], 1, entry_df['speedup'],
                     alpha=0.3, color=SPEEDUP_COLOR)
    ax4.set_xlabel('Entry Bits')
    ax4.set_ylabel('Speedup')
    ax4.set_title('(D) Speedup vs Entry Magnitude')
    ax4.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('Parallel Lift: Exact Determinant via CRT on GPU\n'
                 'Speedup scales with both matrix size and integer magnitude',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    print(f'✓ Saved: {output_path}')
    plt.close()


def main():
    print('Parallel Lift: Generating Benchmark Figures')
    print('=' * 50)

    # Load data
    script_dir = Path(__file__).parent

    # Load scaling results
    scaling_file = script_dir / 'scaling_results.csv'
    if scaling_file.exists():
        scaling_df = pd.read_csv(scaling_file)
        print(f'Loaded: {scaling_file} ({len(scaling_df)} rows)')
        plot_size_scaling(scaling_df)
    else:
        print(f'Warning: {scaling_file} not found')
        scaling_df = None

    # Load entry magnitude results
    entry_file = script_dir / 'entry_magnitude_results.csv'
    if entry_file.exists():
        entry_df = pd.read_csv(entry_file)
        print(f'Loaded: {entry_file} ({len(entry_df)} rows)')
        plot_entry_magnitude(entry_df)
    else:
        print(f'Warning: {entry_file} not found')
        entry_df = None

    # Load stability results
    stability_file = script_dir / 'stability_results.csv'
    if stability_file.exists():
        stability_df = pd.read_csv(stability_file)
        print(f'Loaded: {stability_file} ({len(stability_df)} rows)')
        plot_stability(stability_df)
    else:
        print(f'Warning: {stability_file} not found')

    # Combined summary
    if scaling_df is not None and entry_df is not None:
        plot_combined_summary(scaling_df, entry_df)

    print('=' * 50)
    print('Done! Figures saved to current directory.')


if __name__ == '__main__':
    main()
