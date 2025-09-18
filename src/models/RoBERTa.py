import os

from transformers import (
    RobertaForMaskedLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def init_pretrain_components(config, tokenizer, dataset):
    # Initialize model
    model = RobertaForMaskedLM(config=config["model_config"])
    print(f"Model initialized with {model.num_parameters():,} parameters")

    # Setup data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config["mlm_probability"]
    )

    # Ensure directories exist
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],  # Save checkpoints here
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_dir=f'{config["checkpoint_dir"]}/logs',
        logging_steps=config["logging_steps"],
        prediction_loss_only=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )