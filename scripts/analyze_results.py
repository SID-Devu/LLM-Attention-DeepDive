#!/usr/bin/env python3
"""
Analyze attention benchmark results and generate visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark CSV results."""
    df = pd.read_csv(csv_path)
    return df

def plot_scaling(df: pd.DataFrame, output_dir: Path):
    """Plot performance scaling with sequence length."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Filter for single batch, 8 heads, 64 dim
    subset = df[(df['batch_size'] == 1) & (df['num_heads'] == 8) & (df['head_dim'] == 64)]
    
    for impl in ['naive', 'shared', 'flash']:
        impl_data = subset[subset['attention_type'] == impl]
        
        # Latency plot
        axes[0].plot(impl_data['seq_len'], impl_data['time_ms'], 
                    marker='o', label=impl.capitalize())
        
        # TFLOPS plot
        axes[1].plot(impl_data['seq_len'], impl_data['tflops'],
                    marker='o', label=impl.capitalize())
        
        # Bandwidth plot
        axes[2].plot(impl_data['seq_len'], impl_data['bandwidth_gbps'],
                    marker='o', label=impl.capitalize())
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Latency Scaling')
    axes[0].legend()
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('TFLOPS')
    axes[1].set_title('Throughput Scaling')
    axes[1].legend()
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Bandwidth (GB/s)')
    axes[2].set_title('Memory Bandwidth')
    axes[2].legend()
    axes[2].set_xscale('log', base=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=150)
    plt.close()

def plot_speedup(df: pd.DataFrame, output_dir: Path):
    """Plot speedup comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df[(df['batch_size'] == 1) & (df['num_heads'] == 8) & (df['head_dim'] == 64)]
    
    seq_lens = sorted(subset['seq_len'].unique())
    
    naive_times = []
    shared_speedups = []
    flash_speedups = []
    
    for seq in seq_lens:
        naive = subset[(subset['seq_len'] == seq) & (subset['attention_type'] == 'naive')]
        shared = subset[(subset['seq_len'] == seq) & (subset['attention_type'] == 'shared')]
        flash = subset[(subset['seq_len'] == seq) & (subset['attention_type'] == 'flash')]
        
        if len(naive) > 0 and len(shared) > 0:
            naive_t = naive['time_ms'].values[0]
            shared_speedups.append(naive_t / shared['time_ms'].values[0])
        else:
            shared_speedups.append(0)
            
        if len(naive) > 0 and len(flash) > 0:
            naive_t = naive['time_ms'].values[0]
            flash_speedups.append(naive_t / flash['time_ms'].values[0])
        else:
            flash_speedups.append(0)
    
    x = np.arange(len(seq_lens))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, shared_speedups, width, label='Shared Memory', color='steelblue')
    bars2 = ax.bar(x + width/2, flash_speedups, width, label='Flash Attention', color='coral')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Speedup vs Naive')
    ax.set_title('Attention Kernel Speedup Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=150)
    plt.close()

def plot_memory_efficiency(df: pd.DataFrame, output_dir: Path):
    """Plot memory efficiency analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Calculate theoretical memory for standard vs flash attention
    # Standard: O(N²) for attention matrix
    # Flash: O(N) - no full attention matrix
    
    B, H, D = 1, 8, 64
    
    standard_mem = []
    flash_mem = []
    
    for S in seq_lens:
        # Q, K, V, O + attention matrix
        qkv_mem = 4 * B * H * S * D * 4  # float32
        attn_mem = B * H * S * S * 4
        standard_mem.append((qkv_mem + attn_mem) / (1024 * 1024))  # MB
        
        # Flash: just Q, K, V, O
        flash_mem.append(qkv_mem / (1024 * 1024))
    
    ax.plot(seq_lens, standard_mem, 'o-', label='Standard Attention', color='red', linewidth=2)
    ax.plot(seq_lens, flash_mem, 's-', label='Flash Attention', color='green', linewidth=2)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Scaling: Standard vs Flash Attention')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Annotate key points
    ax.annotate(f'{standard_mem[-1]:.0f} MB', 
                xy=(seq_lens[-1], standard_mem[-1]),
                xytext=(seq_lens[-1]*0.7, standard_mem[-1]*1.5))
    ax.annotate(f'{flash_mem[-1]:.0f} MB',
                xy=(seq_lens[-1], flash_mem[-1]),
                xytext=(seq_lens[-1]*0.7, flash_mem[-1]*0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_scaling.png', dpi=150)
    plt.close()

def generate_summary(df: pd.DataFrame, output_dir: Path):
    """Generate summary report."""
    summary = {
        'configurations_tested': len(df),
        'implementations': df['attention_type'].unique().tolist(),
        'best_performance': {},
        'scaling_analysis': {}
    }
    
    for impl in df['attention_type'].unique():
        impl_data = df[df['attention_type'] == impl]
        summary['best_performance'][impl] = {
            'max_tflops': float(impl_data['tflops'].max()),
            'max_bandwidth_gbps': float(impl_data['bandwidth_gbps'].max()),
            'min_latency_ms': float(impl_data['time_ms'].min())
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Text report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write("# Attention Benchmark Results\n\n")
        f.write(f"Total configurations tested: {len(df)}\n\n")
        
        f.write("## Best Performance by Implementation\n\n")
        for impl, perf in summary['best_performance'].items():
            f.write(f"### {impl.capitalize()}\n")
            f.write(f"- Max TFLOPS: {perf['max_tflops']:.3f}\n")
            f.write(f"- Max Bandwidth: {perf['max_bandwidth_gbps']:.2f} GB/s\n")
            f.write(f"- Min Latency: {perf['min_latency_ms']:.3f} ms\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Flash Attention** shows superior memory efficiency at long sequences\n")
        f.write("2. **Shared Memory** optimization provides consistent speedup over naive\n")
        f.write("3. Scaling is O(N²) for standard attention, O(N) for Flash Attention\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze attention benchmarks')
    parser.add_argument('--csv', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from {args.csv}")
    df = load_results(args.csv)
    
    print("Generating scaling analysis plot...")
    plot_scaling(df, output_dir)
    
    print("Generating speedup comparison plot...")
    plot_speedup(df, output_dir)
    
    print("Generating memory efficiency plot...")
    plot_memory_efficiency(df, output_dir)
    
    print("Generating summary report...")
    generate_summary(df, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
