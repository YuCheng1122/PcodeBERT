#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_adapter_results():
    results_base = Path("/home/tommy/Project/PcodeBERT/outputs/results")
    all_results = []
    
    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        return pd.DataFrame()
    
    for config_dir in results_base.iterdir():
        if not config_dir.is_dir() or config_dir.name == 'ablation_summary':
            continue
        
        csv_files = list(config_dir.glob("results_*.csv"))
        if not csv_files:
            continue
        
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        try:
            df = pd.read_csv(latest_csv)
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {latest_csv}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def main():
    print("Aggregating Adapter Ablation Results")
    print("="*50)
    
    df = load_adapter_results()
    
    if df.empty:
        print("No results found!")
        return
    
    output_dir = Path("/home/tommy/Project/PcodeBERT/outputs/results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nTotal configurations: {df['adapter_name'].nunique()}")
    print(f"Results per config: {len(df) // df['adapter_name'].nunique()} seeds")
    print()
    
    for epoch in sorted(df['adapter_epoch'].unique()):
        for loss in sorted(df['loss_function'].unique()):
            subset = df[(df['adapter_epoch'] == epoch) & (df['loss_function'] == loss)]
            
            if len(subset) == 0:
                continue
            
            print(f"\nConfiguration: {loss.upper()}_Epoch{epoch}")
            print("-"*50)
            
            for cpu in sorted(subset['cpu'].unique()):
                cpu_data = subset[subset['cpu'] == cpu]
                
                if len(cpu_data) == 0:
                    continue
                
                print(f"\n{cpu}:")
                print(f"  Accuracy     : {cpu_data['accuracy'].mean():.4f} ± {cpu_data['accuracy'].std():.4f}")
                print(f"  Precision    : {cpu_data['precision'].mean():.4f} ± {cpu_data['precision'].std():.4f}")
                print(f"  Recall       : {cpu_data['recall'].mean():.4f} ± {cpu_data['recall'].std():.4f}")
                print(f"  F1-micro     : {cpu_data['f1_micro'].mean():.4f} ± {cpu_data['f1_micro'].std():.4f}")
                print(f"  F1-macro     : {cpu_data['f1_macro'].mean():.4f} ± {cpu_data['f1_macro'].std():.4f}")
                print(f"  AUC          : {cpu_data['auc'].mean():.4f} ± {cpu_data['auc'].std():.4f}")
    
    summary_data = []
    for epoch in sorted(df['adapter_epoch'].unique()):
        for loss in sorted(df['loss_function'].unique()):
            subset = df[(df['adapter_epoch'] == epoch) & (df['loss_function'] == loss)]
            
            for cpu in sorted(subset['cpu'].unique()):
                cpu_data = subset[subset['cpu'] == cpu]
                if len(cpu_data) > 0:
                    summary_data.append({
                        'config': f"{loss}_epoch{epoch}",
                        'cpu': cpu,
                        'accuracy_mean': cpu_data['accuracy'].mean(),
                        'accuracy_std': cpu_data['accuracy'].std(),
                        'precision_mean': cpu_data['precision'].mean(),
                        'precision_std': cpu_data['precision'].std(),
                        'recall_mean': cpu_data['recall'].mean(),
                        'recall_std': cpu_data['recall'].std(),
                        'f1_micro_mean': cpu_data['f1_micro'].mean(),
                        'f1_micro_std': cpu_data['f1_micro'].std(),
                        'f1_macro_mean': cpu_data['f1_macro'].mean(),
                        'f1_macro_std': cpu_data['f1_macro'].std(),
                        'auc_mean': cpu_data['auc'].mean(),
                        'auc_std': cpu_data['auc'].std(),
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"ablation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    
    all_results_path = output_dir / f"ablation_all_results_{timestamp}.csv"
    df.to_csv(all_results_path, index=False)
    
    print(f"\n\nSummary saved to: {summary_path}")
    print(f"All results saved to: {all_results_path}")
    print("="*50)


if __name__ == "__main__":
    main()
