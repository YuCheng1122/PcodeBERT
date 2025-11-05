import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from utils import load_cross_arch_data, train_epoch, evaluate, simple_early_stopping, create_gnn_scheduler, test_model, plot_training_curves, load_test_data_by_arch
import sys
import os
import argparse
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gnn_models import GCN
from sklearn.metrics import classification_report

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(seed, config, pretrain_epoch):
    set_random_seed(seed)
    
    train_losses = []
    val_losses_list = []
    test_losses_list = []
    val_accuracies = []
    test_accuracies = []

    csv_path = config["csv_path"]
    graph_dir = config["graph_dir"]
    cache_file = config["cache_file"]
    source_cpus = config["source_cpus"]
    target_cpus = config["target_cpus"]
    classification = config["classification"]
    batch_size = config["batch_size"]
    hidden_channels = config["hidden_channels"]
    lr = config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]
    
    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    
    # Create model save directory from config
    model_save_dir = config["model_output_dir"]
    os.makedirs(model_save_dir, exist_ok=True)
    

    train_graphs, val_graphs, test_graphs, label_encoder, num_classes = load_cross_arch_data(
        csv_path=csv_path,
        graph_dir=graph_dir,
        source_cpus=source_cpus,
        target_cpus=target_cpus,
        cache_file=cache_file,
        force_reload=False,
        classification=classification
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model = GCN(num_node_features=256, hidden_channels=hidden_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    scheduler = create_gnn_scheduler(optimizer, "plateau", patience=10)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy, val_loss = evaluate(model, val_loader, device)
        test_accuracy, test_loss = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        val_losses_list.append(val_loss)
        test_losses_list.append(test_loss)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        scheduler.step(val_loss)

        best_val_acc, patience_counter, should_stop = simple_early_stopping(
            val_accuracy, best_val_acc, patience_counter, patience
        )

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Val Accuracy = {val_accuracy:.4f}, Val Loss = {val_loss:.4f}")

    # Save plots with pretrain_epoch in the filename
    plots_dir = f"outputs/plots/epoch_{pretrain_epoch}"
    os.makedirs(plots_dir, exist_ok=True)
    plot_training_curves(train_losses, val_losses_list, test_losses_list, val_accuracies, seed, save_dir=plots_dir)
    
    # Save model for this seed
    mode_str = "classification" if classification else "detection"
    arch_str = "_".join(source_cpus) if source_cpus else "default"
    model_filename = f"gnn_model_{mode_str}_{arch_str}_epoch{pretrain_epoch}_seed_{seed}.pt"
    model_path = os.path.join(model_save_dir, model_filename)
    
    torch.save({
        'seed': seed,
        'pretrain_epoch': pretrain_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses_list,
        'val_accuracies': val_accuracies
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    if config["target_cpus"]:
        test_results_by_arch = {}
        test_graphs_by_arch = load_test_data_by_arch(
            csv_path, graph_dir, config["target_cpus"], label_encoder, classification
        )
        
        for cpu, graphs in test_graphs_by_arch.items():
            cpu_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            cpu_results = test_model(model, cpu_loader, device, label_encoder)
            cpu_results['test_samples'] = len(graphs)
            test_results_by_arch[cpu] = cpu_results
            print(f"\n{cpu} Results:")
            print(f"Accuracy: {cpu_results['accuracy']:.4f}")
            print(f"F1-micro: {cpu_results['f1_micro']:.4f}")
            print(f"F1-macro: {cpu_results['f1_macro']:.4f}")
            print(f"AUC: {cpu_results['auc']:.4f}")
            print(f"Precision: {cpu_results['precision']:.4f}")
            print(f"Recall: {cpu_results['recall']:.4f}")
        
        return test_results_by_arch
    else:
        test_results = test_model(model, test_loader, device, label_encoder)
        test_results['test_samples'] = len(test_graphs)
        return {'overall': test_results}


def main():
    parser = argparse.ArgumentParser(description='GNN training with specific pretrain epoch')
    parser.add_argument('--epoch', type=int, required=True, choices=[25, 50, 100],
                       help='Pretrain model epoch (25, 50, or 100)')
    parser.add_argument('--source_cpus', nargs='+', default=["x86_64"],
                       help='Source CPU architectures for training')
    parser.add_argument('--target_cpus', nargs='+', default=["ARM", "PPC", "MIPS", "x86_64", "Intel"],
                       help='Target CPU architectures for testing')
    args = parser.parse_args()
    
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = {
        "classification": False,  
        "source_cpus": args.source_cpus,     
        "target_cpus": args.target_cpus,

        "csv_path": f"{BASE_PATH}/dataset/csv/merged_adjusted_filtered.csv",
        "graph_dir": f"{BASE_PATH}/outputs/models/GNN/embeddings_epoch_{args.epoch}",
        "cache_file": f"{BASE_PATH}/outputs/cache/gnn_data_epoch_{args.epoch}.pkl",
        "model_output_dir": f"{BASE_PATH}/outputs/models/GNN/models_epoch_{args.epoch}",
        
        "batch_size": 32,
        "hidden_channels": 128,
        "learning_rate": 0.01,
        "epochs": 200,
        "patience": 20,
        
        "seeds": [42, 123, 2025, 31415, 8888],
        "device": "cuda"
    }
    
    seeds = config["seeds"]
    
    # 判斷模式
    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    arch_mode = "單架構" if not config["target_cpus"] else "跨架構"
    
    print(f"Pretrain Epoch: {args.epoch}")
    print(f"模式: {mode}")
    print(f"架構模式: {arch_mode}")
    print(f"Training Architecture: {config['source_cpus']}")
    if config['target_cpus']:
        print(f"Testing Architecture: {config['target_cpus']}")
    print(f"Experiment Count: {len(seeds)}")
    print(f"Graph Data Directory: {config['graph_dir']}")

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n=== Experiment {i+1}, Seed = {seed} ===")
        results = run_experiment(seed, config, args.epoch)
        all_results.append(results)

    if config["target_cpus"]:
        print(f"\n{'='*60}")
        print(f"Summary of {len(seeds)} Experiments (By Architecture)")
        print(f"{'='*60}")
        
        results_by_arch = {}
        for cpu in config["target_cpus"]:
            cpu_accs = [r[cpu]['accuracy'] for r in all_results]
            cpu_f1_micros = [r[cpu]['f1_micro'] for r in all_results]
            cpu_f1_macros = [r[cpu]['f1_macro'] for r in all_results]
            cpu_aucs = [r[cpu]['auc'] for r in all_results]
            cpu_precisions = [r[cpu]['precision'] for r in all_results]
            cpu_recalls = [r[cpu]['recall'] for r in all_results]
            cpu_test_samples = all_results[0][cpu]['test_samples']
            
            results_by_arch[cpu] = {
                'avg_accuracy': np.mean(cpu_accs),
                'std_accuracy': np.std(cpu_accs),
                'avg_f1_micro': np.mean(cpu_f1_micros),
                'std_f1_micro': np.std(cpu_f1_micros),
                'avg_f1_macro': np.mean(cpu_f1_macros),
                'std_f1_macro': np.std(cpu_f1_macros),
                'avg_auc': np.mean(cpu_aucs),
                'std_auc': np.std(cpu_aucs),
                'avg_precision': np.mean(cpu_precisions),
                'std_precision': np.std(cpu_precisions),
                'avg_recall': np.mean(cpu_recalls),
                'std_recall': np.std(cpu_recalls),
                'test_samples': cpu_test_samples
            }
            
            print(f"\n{cpu}:")
            print(f"  Accuracy     : {results_by_arch[cpu]['avg_accuracy']:.4f} ± {results_by_arch[cpu]['std_accuracy']:.4f}")
            print(f"  F1-micro     : {results_by_arch[cpu]['avg_f1_micro']:.4f} ± {results_by_arch[cpu]['std_f1_micro']:.4f}")
            print(f"  F1-macro     : {results_by_arch[cpu]['avg_f1_macro']:.4f} ± {results_by_arch[cpu]['std_f1_macro']:.4f}")
            print(f"  AUC          : {results_by_arch[cpu]['avg_auc']:.4f} ± {results_by_arch[cpu]['std_auc']:.4f}")
            print(f"  Precision    : {results_by_arch[cpu]['avg_precision']:.4f} ± {results_by_arch[cpu]['std_precision']:.4f}")
            print(f"  Recall       : {results_by_arch[cpu]['avg_recall']:.4f} ± {results_by_arch[cpu]['std_recall']:.4f}")
            print(f"  Test Samples : {results_by_arch[cpu]['test_samples']}")
        
        all_results_flat = []
        for seed, result_dict in zip(seeds, all_results):
            for cpu, metrics in result_dict.items():
                all_results_flat.append({
                    'pretrain_epoch': args.epoch,
                    'seed': seed,
                    'cpu': cpu,
                    'accuracy': metrics['accuracy'],
                    'f1_micro': metrics['f1_micro'],
                    'f1_macro': metrics['f1_macro'],
                    'auc': metrics['auc'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'test_samples': metrics['test_samples']
                })
        
        results_summary = {
            'pretrain_epoch': args.epoch,
            'mode': mode,
            'arch_mode': arch_mode,
            'source_cpus': config['source_cpus'],
            'target_cpus': config['target_cpus'],
            'results_by_arch': results_by_arch,
            'seeds': seeds,
            'all_results': all_results_flat
        }
    else:
        overall_accs = [r['overall']['accuracy'] for r in all_results]
        overall_f1_micros = [r['overall']['f1_micro'] for r in all_results]
        overall_f1_macros = [r['overall']['f1_macro'] for r in all_results]
        overall_aucs = [r['overall']['auc'] for r in all_results]
        overall_precisions = [r['overall']['precision'] for r in all_results]
        overall_recalls = [r['overall']['recall'] for r in all_results]
        overall_test_samples = all_results[0]['overall']['test_samples']
        
        avg_acc = np.mean(overall_accs)
        avg_f1_micro = np.mean(overall_f1_micros)
        avg_f1_macro = np.mean(overall_f1_macros)
        avg_auc = np.mean(overall_aucs)
        avg_precision = np.mean(overall_precisions)
        avg_recall = np.mean(overall_recalls)
        std_acc = np.std(overall_accs)
        std_f1_micro = np.std(overall_f1_micros)
        std_f1_macro = np.std(overall_f1_macros)
        std_auc = np.std(overall_aucs)
        std_precision = np.std(overall_precisions)
        std_recall = np.std(overall_recalls)
        
        print(f"\n{len(seeds)} Experiments Summary:")
        print(f"Accuracy     : {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"F1-micro     : {avg_f1_micro:.4f} ± {std_f1_micro:.4f}")
        print(f"F1-macro     : {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
        print(f"AUC          : {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"Precision    : {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall       : {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"Test Samples : {overall_test_samples}")
        
        results_summary = {
            'pretrain_epoch': args.epoch,
            'mode': mode,
            'arch_mode': arch_mode,
            'source_cpus': config['source_cpus'],
            'target_cpus': config['target_cpus'],
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_f1_micro': avg_f1_micro,
            'std_f1_micro': std_f1_micro,
            'avg_f1_macro': avg_f1_macro,
            'std_f1_macro': std_f1_macro,
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'avg_precision': avg_precision,
            'std_precision': std_precision,
            'avg_recall': avg_recall,
            'std_recall': std_recall,
            'test_samples': overall_test_samples,
            'seeds': seeds,
            'all_results': [{'pretrain_epoch': args.epoch, 'seed': seed, 
                           'accuracy': r['overall']['accuracy'], 
                           'f1_micro': r['overall']['f1_micro'], 
                           'f1_macro': r['overall']['f1_macro'],
                           'auc': r['overall']['auc'],
                           'precision': r['overall']['precision'],
                           'recall': r['overall']['recall'],
                           'test_samples': r['overall']['test_samples']} 
                          for seed, r in zip(seeds, all_results)]
        }
    
    # Save results with epoch in filename
    save_dir = f"outputs/results/epoch_{args.epoch}"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    if 'all_results' in results_summary:
        results_df = pd.DataFrame(results_summary['all_results'])
        results_df.to_csv(os.path.join(save_dir, f'results_{timestamp}.csv'), index=False)
    
    summary_data = {k: v for k, v in results_summary.items() if k != 'all_results'}
    with open(os.path.join(save_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved with timestamp: {timestamp}")
    print(f"Results directory: {save_dir}")

if __name__ == "__main__":
    main()
