#!/usr/bin/env python3
"""
Generate LaTeX tables from benchmark CSV results for paper.
"""

import pandas as pd
import sys
import os
from pathlib import Path


def format_time(ms):
    """Format time in milliseconds to appropriate unit."""
    if ms < 1:
        return f"{ms*1000:.1f} μs"
    elif ms < 1000:
        return f"{ms:.1f} ms"
    else:
        return f"{ms/1000:.2f} s"


def format_speedup(speedup):
    """Format speedup with appropriate precision."""
    if speedup < 10:
        return f"{speedup:.1f}$\\times$"
    elif speedup < 100:
        return f"{speedup:.0f}$\\times$"
    else:
        return f"{speedup:,.0f}$\\times$"


def generate_multi_rhs_n_table(results_dir):
    """Table 2: Multi-RHS scaling with n."""
    csv_path = Path(results_dir) / "multi_rhs_scaling_n.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return ""

    df = pd.read_csv(csv_path)
    df_median = df.groupby(['n', 'k', 'primes']).median().reset_index()

    latex = r"""
\begin{table}[t]
    \centering
    \caption{Multi-RHS solve performance scaling with matrix size ($k = 16$, 48-bit entries).}
    \label{tab:multi-rhs-scaling-n}
    \begin{tabular}{rrrrrr}
        \toprule
        $n$ & Primes & CPU (ms) & GPU (ms) & Speedup \\
        \midrule
"""

    for _, row in df_median.iterrows():
        latex += f"        {int(row['n'])} & {int(row['primes'])} & "
        latex += f"{row['cpu_ms']:,.0f} & {row['gpu_ms']:.1f} & "
        latex += f"{format_speedup(row['cpu_ms']/row['gpu_ms'])} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


def generate_multi_rhs_k_table(results_dir):
    """Table 3: Multi-RHS scaling with k."""
    csv_path = Path(results_dir) / "multi_rhs_scaling_k.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return ""

    df = pd.read_csv(csv_path)
    df_median = df.groupby(['n', 'k', 'primes']).median().reset_index()

    latex = r"""
\begin{table}[t]
    \centering
    \caption{Multi-RHS solve performance scaling with $k$ (fixed $n = 128$, 48-bit entries).}
    \label{tab:multi-rhs-scaling-k}
    \begin{tabular}{rrrrrr}
        \toprule
        $k$ & CPU (ms) & GPU (ms) & Speedup \\
        \midrule
"""

    for _, row in df_median.iterrows():
        latex += f"        {int(row['k'])} & "
        latex += f"{row['cpu_ms']:,.0f} & {row['gpu_ms']:.1f} & "
        latex += f"{format_speedup(row['cpu_ms']/row['gpu_ms'])} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


def generate_crt_table(results_dir):
    """Table 6: CRT reconstruction performance."""
    csv_path = Path(results_dir) / "crt_reconstruction.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return ""

    df = pd.read_csv(csv_path)
    df_median = df.groupby(['primes', 'values']).median().reset_index()

    latex = r"""
\begin{table}[t]
    \centering
    \caption{CRT reconstruction performance (Garner's algorithm).}
    \label{tab:crt}
    \begin{tabular}{rrrrrr}
        \toprule
        Primes & Values & CPU (ms) & GPU (ms) & Speedup \\
        \midrule
"""

    for _, row in df_median.iterrows():
        latex += f"        {int(row['primes'])} & {int(row['values'])} & "
        latex += f"{row['cpu_ms']:.0f} & {row['gpu_ms']:.1f} & "
        latex += f"{format_speedup(row['cpu_ms']/row['gpu_ms'])} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


def generate_full_pipeline_table(results_dir):
    """Full GPU pipeline table (key result)."""
    csv_path = Path(results_dir) / "full_pipeline.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return ""

    df = pd.read_csv(csv_path)
    df_median = df.groupby(['n', 'k', 'primes']).median().reset_index()

    latex = r"""
\begin{table}[t]
    \centering
    \caption{Full GPU pipeline performance (solve + CRT). GPU+GPU CRT is the full pipeline.}
    \label{tab:full-pipeline}
    \begin{tabular}{rrrrrr}
        \toprule
        $n$ & Primes & CPU (ms) & GPU+CPU CRT & GPU+GPU CRT & Speedup \\
        \midrule
"""

    for _, row in df_median.iterrows():
        speedup = row['cpu_total_ms'] / row['gpu_gpu_crt_ms']
        latex += f"        {int(row['n'])} & {int(row['primes'])} & "
        latex += f"{row['cpu_total_ms']:,.0f} & {row['gpu_cpu_crt_ms']:.1f} & "
        latex += f"{row['gpu_gpu_crt_ms']:.1f} & {format_speedup(speedup)} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


def generate_summary_table(results_dir):
    """Summary table of key results."""
    latex = r"""
\begin{table}[t]
    \centering
    \caption{Summary of key performance results.}
    \label{tab:summary}
    \begin{tabular}{llr}
        \toprule
        Operation & Configuration & Peak Speedup \\
        \midrule
"""

    # TODO: Parse actual results and fill in
    results = [
        ("Multi-RHS Solve", "$n=128$, $k=16$, GPU CRT", "552$\\times$"),
        ("Multi-RHS Solve", "$n=256$, $k=16$, CPU CRT", "166$\\times$"),
        ("Matrix Inverse", "$n=128$, $k=128$", "71$\\times$"),
        ("ZK Merkle Witness", "$n=128$, $k=16$", "78$\\times$"),
        ("CRT Reconstruction", "512 primes, 4096 values", "25$\\times$"),
        ("Modular Solve (isolated)", "$n=256$, $k=16$", "2,023$\\times$"),
    ]

    for op, config, speedup in results:
        latex += f"        {op} & {config} & {speedup} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    return latex


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_latex_tables.py <results_dir>")
        print("Example: python generate_latex_tables.py ./paper_results_cuda")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = Path(results_dir) / "latex_tables.tex"

    print(f"Generating LaTeX tables from {results_dir}...")

    tables = []
    tables.append("% Auto-generated LaTeX tables from benchmark results")
    tables.append("% Do not edit manually - regenerate using generate_latex_tables.py")
    tables.append("")

    tables.append(generate_multi_rhs_n_table(results_dir))
    tables.append(generate_multi_rhs_k_table(results_dir))
    tables.append(generate_crt_table(results_dir))
    tables.append(generate_full_pipeline_table(results_dir))
    tables.append(generate_summary_table(results_dir))

    with open(output_file, 'w') as f:
        f.write('\n'.join(tables))

    print(f"LaTeX tables written to {output_file}")
    print("\nTo include in paper, add to sections/evaluation.tex:")
    print(f"  \\input{{{output_file}}}")


if __name__ == "__main__":
    main()
