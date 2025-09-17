from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 測試你的數據
sample_text = "INT_EQUAL"
tokens = tokenizer.tokenize(sample_text)
print(f"Unknown tokens: {tokens.count('[UNK]')}/{len(tokens)}")