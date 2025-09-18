from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import WordLevel
import os

def create_wordlevel_tokenizer(vocab_path, output_path):
    """
    Create a WordLevel tokenizer for Pcode instructions.
    
    Args:
        vocab_path: Path to vocabulary file (one token per line)
        output_path: Path to save the tokenizer configuration
        
    Returns:
        tokenizer: Configured Tokenizer instance
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {token.strip(): i for i, token in enumerate(f.readlines())}

    print(f"Successfully loaded {len(vocab)} tokens from {vocab_path}.")

    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()  

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

def test_tokenizer(tokenizer):
    """
    Test tokenizer functionality with example Pcode sequences.
    """
    print("\n=== Tokenizer Test ===")
    
    # Test normal sequence
    test_seq = ["LOAD", "UNIQUE", "CONST", "REG"]
    encoding = tokenizer.encode(" ".join(test_seq))
    print(f"\nInput: {test_seq}")
    print(f"Tokens: {encoding.tokens}")
    print(f"IDs: {encoding.ids}")
    
    # Test unknown token handling
    oov_seq = ["LOAD", "UNKNOWN_TOKEN", "CONST"]
    encoding = tokenizer.encode(" ".join(oov_seq))
    print(f"\nInput with OOV: {oov_seq}")
    print(f"Tokens: {encoding.tokens}")
    print(f"IDs: {encoding.ids}")
