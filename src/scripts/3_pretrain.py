import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_corpus_dataset, setup_training_environment, create_scheduler
from configs.pretrain_config import get_pretrain_config
from models.tokenizer import create_wordlevel_tokenizer
from models.RoBERTa import init_pretrain_components, create_model


def main():
    # Setup training environment with GPU support
    device = setup_training_environment()
    
    # Load configuration
    config = get_pretrain_config()
    
    # Setup tokenizer
    tokenizer = create_wordlevel_tokenizer(
        vocab_path=config["vocab_path"],
        output_path=config["tokenizer_output_path"],
        special_tokens=config["special_tokens"],
        max_length=config["max_length"]
    )

    # Load dataset
    raw_dataset = load_corpus_dataset(config["corpus_path"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    # Create model and move to device
    model = create_model(config["model_config"])
    model.to(device)
    
    # Initialize trainer components
    trainer = init_pretrain_components(config, model, tokenizer, tokenized_dataset)
    
    # Calculate total training steps for scheduler
    num_training_steps = len(tokenized_dataset) // config["batch_size"] * config["epochs"]
    
    # Create and set scheduler (access optimizer from trainer after initialization)
    if trainer.optimizer is not None:
        scheduler = create_scheduler(
            trainer.optimizer, 
            num_training_steps, 
            scheduler_type="linear",  # You can change to "cosine" if preferred
            warmup_ratio=0.1
        )
        
        # Set the scheduler in trainer
        trainer.lr_scheduler = scheduler
    else:
        print("Warning: Trainer optimizer is None, scheduler not created")

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer to output directory
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Final model and tokenizer saved to {config['output_dir']}")
    print(f"Training checkpoints saved in {config['checkpoint_dir']}")

if __name__ == "__main__":
    main()
