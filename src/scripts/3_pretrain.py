from tokenizers import create_wordlevel_tokenizer

"""
    The main function to perform RoBERTa pre-training.
    """
    # 1. 載入訓練配置
    config = get_pretrain_config()
    model_config = config["model_config"]

    # 2. 創建並載入 WordLevel Tokenizer
    # ----------------------------------------------------
    # 讀取 vocab 檔案
    # 這一步確保 tokenizers.json 的詞彙表與你的 vocab.txt 檔案一致
    with open(config["tokenizer_path"], 'r', encoding='utf-8') as f:
        tokens = [token.strip() for token in f.readlines()]
        vocab = {token: i for i, token in enumerate(tokens)}
        # 確保你的 vocab_size 和配置一致
        assert len(vocab) == config["vocab_size"], "Vocab size mismatch!"

    # 創建一個 tokenizers.Tokenizer 物件
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.decoders import WordLevel as WordLevelDecoder
    
    tokenizer_obj = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = WhitespaceSplit()
    tokenizer_obj.decoder = WordLevelDecoder(vocab)

    # 將 tokenizers 物件轉換為 PreTrainedTokenizerFast
    # 這樣它才能和 Transformers 框架一起運作
    # 這些特殊 tokens 必須在你的 vocab.txt 裡
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    print("Tokenizer loaded and configured.")

    # 3. 載入並準備數據集
    # ----------------------------------------------------
    # 由於你的數據是 .pkl 檔案，load_dataset 無法直接處理
    # 你需要自己寫載入邏輯，將其轉換為 datasets 格式
    # 假設你的 .pkl 檔案是一個包含所有 IR 指令字串的列表
    import pickle
    with open(config["corpus_path"], 'rb') as f:
        data = pickle.load(f)
        # 創建一個 datasets.Dataset 物件
        # 這裡假設每個元素是一個完整的指令序列字串
        dataset = Dataset.from_dict({"text": data})

    print(f"Dataset loaded with {len(dataset)} sequences.")

    # 4. 實例化模型和數據收集器
    # ----------------------------------------------------
    model = RobertaForMaskedLM(config=model_config)
    print(f"Model initialized with {model.num_parameters():,} parameters.")

    # 數據收集器負責將序列填充和遮罩
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    print("Data collator created.")

    # 5. 設定訓練參數並開始訓練
    # ----------------------------------------------------
    # 確保輸出目錄存在
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        save_steps=1000, # 每隔 1000 steps 保存一次
        save_total_limit=2, # 只保留最新的兩個 checkpoints
        logging_dir=f'{config["output_dir"]}/logs',
        logging_steps=50,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 6. 保存最終模型
    # ----------------------------------------------------
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Final model and tokenizer saved to {config['output_dir']}")


if __name__ == "__main__":

    vocab_file = "/home/tommy/Projects/PcodeBERT/outputs/tokenizer/vocab.txt"
    tokenizer_config_path = "/home/tommy/Projects/PcodeBERT/outputs/tokenizer/pcode_wordlevel_tokenizer.json"
    pcode_tokenizer = create_wordlevel_tokenizer(vocab_file, tokenizer_config_path)

    # --- 測試 tokenizer ---
    print("\n--- Tokenizer 測試 ---")
    test_sequence = "LOAD CONST INT_ADD STORE RETURN"
    encoding = pcode_tokenizer.encode(test_sequence, is_pretokenized=True)

    print(f"原始序列：'{test_sequence}'")
    print(f"Tokens: {encoding.tokens}")
    print(f"Token IDs: {encoding.ids}")
    
    # 測試解碼
    decoded_sequence = pcode_tokenizer.decode(encoding.ids)
    print(f"解碼回來的序列: '{decoded_sequence}'")

    # 測試 OOV (Out of Vocabulary) token
    print("\n--- OOV 測試 ---")
    oov_sequence = "LOAD NEW_INSTRUCTION BRANCH"
    oov_encoding = pcode_tokenizer.encode(oov_sequence, is_pretokenized=True)
    print(f"原始序列：'{oov_sequence}'")
    print(f"Tokens: {oov_encoding.tokens}") # 'NEW_INSTRUCTION' 會被轉換成 '[UNK]'
    print(f"Token IDs: {oov_encoding.ids}")