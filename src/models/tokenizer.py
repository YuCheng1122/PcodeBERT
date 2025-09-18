from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
import os

def create_wordlevel_tokenizer(vocab_path, output_path, special_tokens, max_length):
    """
    Create a WordLevel tokenizer for Pcode instructions.
    
    Args:
        vocab_path: Path to vocabulary file (one token per line)
        output_path: Path to save the tokenizer configuration
        
    Returns:
        tokenizer: Configured Tokenizer instance
    """

    tokenizer_dir = os.path.dirname(output_path)
    tokenizer_file = os.path.basename(output_path)
    print(f"Tokenizer file: {tokenizer_file}, Directory: {tokenizer_dir}")

    if os.path.exists(output_path):
        print(f"Loading existing tokenizer from {output_path}")
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            model_max_length=max_length,
            **special_tokens
        )
        return fast_tokenizer

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {token.strip(): i for i, token in enumerate(f.readlines())}

    print(f"Successfully loaded {len(vocab)} tokens from {vocab_path}.")

    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']}",
        special_tokens=[
            (special_tokens['cls_token'], vocab[special_tokens['cls_token']]),
            (special_tokens['sep_token'], vocab[special_tokens['sep_token']])
        ],
    )

    os.makedirs(tokenizer_dir, exist_ok=True)
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=max_length,
        **special_tokens
    )

    fast_tokenizer.save_pretrained(tokenizer_dir)
    return fast_tokenizer

def test_tokenizer(tokenizer):
    """
    Test tokenizer functionality with example Pcode sequences.
    """
    print("\n=== Tokenizer Test ===")
    
    # Test normal sequence
    test_seq = "LOAD", "UNIQUE", "CONST", "REG"
    encoding = tokenizer.encode_plus(test_seq, padding='max_length', truncation=True)
    print(f"\nInput: {test_seq}")
    print(f"Tokens: {encoding.tokens()}")
    print(f"IDs: {encoding['input_ids']}")

    decoded_string = tokenizer.decode(encoding['input_ids'], skip_special_tokens=True)
    
    # Test unknown token handling
    oov_seq = "LOAD", "UNKNOWN_TOKEN", "CONST"
    encoding = tokenizer.encode(oov_seq)
    print(f"\nInput with OOV: {oov_seq}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoding)}")
    print(f"IDs: {encoding}")
