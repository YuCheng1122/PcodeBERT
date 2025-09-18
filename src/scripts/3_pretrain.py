import os
from utils import load_corpus_dataset
from configs.pretrain_config import get_pretrain_config
from models.tokenizer import create_wordlevel_tokenizer, test_tokenizer
from models.RoBERTa import init_pretrain_components



def main():
    # Load configuration
    config = get_pretrain_config()
    
    # Setup tokenizer
    tokenizer = create_wordlevel_tokenizer(config)
    print("Tokenizer configured successfully")

    # Load dataset
    dataset = load_corpus_dataset(config["corpus_path"])
    print(f"Dataset loaded with {len(dataset)} sequences")


    trainer = init_pretrain_components(config, tokenizer, dataset)

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer to output directory
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Final model and tokenizer saved to {config['output_dir']}")
    print(f"Training checkpoints saved in {config['checkpoint_dir']}")

if __name__ == "__main__":
    main()
