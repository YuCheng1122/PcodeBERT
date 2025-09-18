import os
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from tokenizers import Tokenizer

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                    unk_token="[UNK]", 
                                    pad_token="[PAD]", 
                                    cls_token="[CLS]", 
                                    sep_token="[SEP]", 
                                    mask_token="[MASK]")