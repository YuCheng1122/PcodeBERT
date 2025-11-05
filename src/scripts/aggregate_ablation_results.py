#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_adapter_results():
    results_base = "outputs/results/adapter_ablation"
    all_results = []
    
    if not os.path.exists(results_base):
        print(f"Warning: Results directory not found: {results_base}")
        return pd.DataFrame()
    
    for config_dir in Path(results_base).iterdir():
        if not config_dir.is_dir():
            continue
        
        csv_files = list(config_dir.glob("results_*.csv"))
        if not csv_files:
            continue
        
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_csv)
        all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def create_summary_table(df):
    summary_rows = []
    
    for epoch in [10, 20, 30]:
        for loss in ['mse', 'cosine']:
            for layers in [6]:
                subset = df[
                    (df['adapter_epochs'] == epoch) &
                    (df['loss_function'] == loss) &
                    (df['adapter_layers'] == layers)
                ]
                
                if len(subset) == 0:
                    continue
                
                for cpu in subset['cpu'].unique():
                    cpu_data = subset[subset['cpu'] == cpu]
                    
                    row = {
                        'Epoch': epoch,
                        'Loss': loss.upper(),
                        'Layers': layers,
                        'Arch': cpu,
                        'AUC': f"{cpu_data['auc'].mean():.4f} ± {cpu_data['auc'].std():.4f}",
                        'Precision': f"{cpu_data['precision'].mean():.4f} ± {cpu_data['precision'].std():.4f}",
                        'Recall': f"{cpu_data['recall'].mean():.4f} ± {cpu_data['recall'].std():.4f}",
                        'F1-Micro': f"{cpu_data['f1_micro'].mean():.4f} ± {cpu_data['f1_micro'].std():.4f}",
                        'F1-Macro': f"{cpu_data['f1_macro'].mean():.4f} ± {cpu_data['f1_macro'].std():.4f}",
                        'Accuracy': f"{cpu_data['accuracy'].mean():.4f} ± {cpu_data['accuracy'].std():.4f}",
                        'Samples': int(cpu_data['test_samples'].iloc[0])
                    }
                    summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def create_comparison_table(df):
    comparison_rows = []
    
    for epoch in [10, 20, 30]:
        for cpu in df['cpu'].unique():
            subset = df[(df['adapter_epochs'] == epoch) & (df['cpu'] == cpu)]
            
            if len(subset) == 0:
                continue
            
            row = {'Epoch': epoch, 'Arch': cpu}
            
            for loss in ['mse', 'cosine']:
                for layers in [6]:
                    config_data = subset[
                        (subset['loss_function'] == loss) &
                        (subset['adapter_layers'] == layers)
                    ]
                    
                    if len(config_data) > 0:
                        col_name = f"{loss.upper()}-{layers}L"
                        auc = config_data['auc'].mean()
                        row[col_name] = f"{auc:.4f}"
            
            comparison_rows.append(row)
    
    return pd.DataFrame(comparison_rows)


def main():
    print("=" * 80)
    print("Aggregating Adapter Ablation Results")
    print("=" * 80)
    
    df = load_adapter_results()
    
    if df.empty:
        print("\nNo results found!")
        return
    
    print(f"\nLoaded {len(df)} result entries")
    print(f"Configurations: {df['adapter_name'].nunique()}")
    print(f"Epochs: {sorted(df['adapter_epochs'].unique())}")
    print(f"Architectures: {sorted(df['cpu'].unique())}")
    
    output_dir = "outputs/results/ablation_study"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    summary_df = create_summary_table(df)
    summary_path = os.path.join(output_dir, "adapter_ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Comparison Table (AUC)")
    print("=" * 80)
    comparison_df = create_comparison_table(df)
    comparison_path = os.path.join(output_dir, "adapter_ablation_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {comparison_path}")
    
    numeric_path = os.path.join(output_dir, "adapter_ablation_numeric.csv")
    df.to_csv(numeric_path, index=False)
    print(f"\nFull numeric data saved to: {numeric_path}")
    
    markdown_path = os.path.join(output_dir, "adapter_ablation_summary.md")
    with open(markdown_path, 'w') as f:
        f.write("# Adapter Ablation Study Results\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Comparison Table (AUC by Configuration)\n\n")
        f.write(comparison_df.to_markdown(index=False))
    print(f"Markdown summary saved to: {markdown_path}")
    
    print("\n" + "=" * 80)
    print("Aggregation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
