import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from utils import load_cross_arch_data, train_epoch, evaluate, simple_early_stopping, create_gnn_scheduler, test_model, plot_training_curves, save_experiment_results, load_test_data_by_arch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gnn_models import GCN
from configs.gnn_config import get_gnn_config
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

def run_experiment(seed, config):
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

    plot_training_curves(train_losses, val_losses_list, test_losses_list, val_accuracies, seed)
    
    # Save model for this seed
    mode_str = "classification" if classification else "detection"
    arch_str = "_".join(source_cpus) if source_cpus else "default"
    model_filename = f"gnn_model_{mode_str}_{arch_str}_seed_{seed}.pt"
    model_path = os.path.join(model_save_dir, model_filename)
    
    torch.save({
        'seed': seed,
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
            test_results_by_arch[cpu] = cpu_results
            print(f"\n{cpu} Results:")
            print(f"Accuracy: {cpu_results['accuracy']:.4f}")
            print(f"F1-micro: {cpu_results['f1_micro']:.4f}")
            print(f"F1-macro: {cpu_results['f1_macro']:.4f}")
            print(f"AUC: {cpu_results['auc']:.4f}")
        
        return test_results_by_arch
    else:
        test_results = test_model(model, test_loader, device, label_encoder)
        return {'overall': test_results}


def main():
    config = get_gnn_config()
    seeds = config["seeds"]
    
    # 判斷模式
    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    arch_mode = "單架構" if not config["target_cpus"] else "跨架構"
    
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
        results = run_experiment(seed, config)
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
            
            results_by_arch[cpu] = {
                'avg_accuracy': np.mean(cpu_accs),
                'std_accuracy': np.std(cpu_accs),
                'avg_f1_micro': np.mean(cpu_f1_micros),
                'std_f1_micro': np.std(cpu_f1_micros),
                'avg_f1_macro': np.mean(cpu_f1_macros),
                'std_f1_macro': np.std(cpu_f1_macros),
                'avg_auc': np.mean(cpu_aucs),
                'std_auc': np.std(cpu_aucs)
            }
            
            print(f"\n{cpu}:")
            print(f"  Accuracy     : {results_by_arch[cpu]['avg_accuracy']:.4f} ± {results_by_arch[cpu]['std_accuracy']:.4f}")
            print(f"  F1-micro     : {results_by_arch[cpu]['avg_f1_micro']:.4f} ± {results_by_arch[cpu]['std_f1_micro']:.4f}")
            print(f"  F1-macro     : {results_by_arch[cpu]['avg_f1_macro']:.4f} ± {results_by_arch[cpu]['std_f1_macro']:.4f}")
            print(f"  AUC          : {results_by_arch[cpu]['avg_auc']:.4f} ± {results_by_arch[cpu]['std_auc']:.4f}")
        
        all_results_flat = []
        for seed, result_dict in zip(seeds, all_results):
            for cpu, metrics in result_dict.items():
                all_results_flat.append({
                    'seed': seed,
                    'cpu': cpu,
                    'accuracy': metrics['accuracy'],
                    'f1_micro': metrics['f1_micro'],
                    'f1_macro': metrics['f1_macro'],
                    'auc': metrics['auc']
                })
        
        results_summary = {
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
        
        avg_acc = np.mean(overall_accs)
        avg_f1_micro = np.mean(overall_f1_micros)
        avg_f1_macro = np.mean(overall_f1_macros)
        avg_auc = np.mean(overall_aucs)
        std_acc = np.std(overall_accs)
        std_f1_micro = np.std(overall_f1_micros)
        std_f1_macro = np.std(overall_f1_macros)
        std_auc = np.std(overall_aucs)
        
        print(f"\n{len(seeds)} Experiments Summary:")
        print(f"Accuracy     : {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"F1-micro     : {avg_f1_micro:.4f} ± {std_f1_micro:.4f}")
        print(f"F1-macro     : {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
        print(f"AUC          : {avg_auc:.4f} ± {std_auc:.4f}")
        
        results_summary = {
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
            'seeds': seeds,
            'all_results': [{'seed': seed, 'accuracy': r['overall']['accuracy'], 
                           'f1_micro': r['overall']['f1_micro'], 
                           'f1_macro': r['overall']['f1_macro'],
                           'auc': r['overall']['auc']} 
                          for seed, r in zip(seeds, all_results)]
        }
    
    timestamp = save_experiment_results(results_summary)
    print(f"\nResults saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()
