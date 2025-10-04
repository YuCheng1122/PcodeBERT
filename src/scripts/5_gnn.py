import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from utils import load_cross_arch_data, train_epoch, evaluate, simple_early_stopping, create_gnn_scheduler, test_model, plot_training_curves
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
    test_results = test_model(model, test_loader, device, label_encoder)
    return test_results


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
        print(f"\n Experiment {i+1}，Seed = {seed}")
        results = run_experiment(seed, config)
        all_results.append(results)

        print(f"Experiment {i+1} Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-micro: {results['f1_micro']:.4f}")
        print(f"F1-macro: {results['f1_macro']:.4f}")

    avg_acc = np.mean([r['accuracy'] for r in all_results])
    avg_f1_micro = np.mean([r['f1_micro'] for r in all_results])
    avg_f1_macro = np.mean([r['f1_macro'] for r in all_results])

    std_acc = np.std([r['accuracy'] for r in all_results])
    std_f1_micro = np.std([r['f1_micro'] for r in all_results])
    std_f1_macro = np.std([r['f1_macro'] for r in all_results])

    print(f"\n{len(seeds)} Experiments Summary:")
    print(f"Accuracy     : {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-score (micro): {avg_f1_micro:.4f} ± {std_f1_micro:.4f}")
    print(f"F1-score (macro): {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")

    mode = "Classification (family)" if config["classification"] else "Detection (label)"
    arch_mode = "單架構" if not config["target_cpus"] else "跨架構"
    print(f"\nGNN Training and Testing - {mode} - {arch_mode}")
    print(f"Training Architecture: {config['source_cpus']}")
    if config['target_cpus']:
        print(f"Testing Architecture: {config['target_cpus']}")

if __name__ == "__main__":
    main()
