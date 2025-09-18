from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import WordLevel


import os

def create_wordlevel_tokenizer(vocab_path, output_path):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {token.strip(): i for i, token in enumerate(f.readlines())}

    print(f"Successfully loaded {len(vocab)} tokens from {vocab_path}.")

    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordLevel()

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
            ("[SEP]", vocab["[SEP]"]),
        ],
    )

    tokenizer.save(output_path)
    print(f"Tokenizer saved to: {output_path}")
    
    return tokenizer
