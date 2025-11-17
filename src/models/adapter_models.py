import torch
import torch.nn as nn
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import os


class AdapterEmbeddingModel(nn.Module):
    def __init__(self, model_name, adapter_config, adapter_name):
        super().__init__()
        
        self.adapter_name = adapter_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.add_adapter(adapter_name, config=adapter_config)
        self.model.train_adapter(adapter_name)
        self.model.set_active_adapters(adapter_name)
        
        for name, param in self.model.named_parameters():
            if 'adapter' not in name.lower():
                param.requires_grad = False
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nAdapter Model: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)\n")
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]
    
    def save_adapter(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_adapter(save_path, self.adapter_name)
    
    def load_adapter(self, load_path):
        self.model.load_adapter(load_path, load_as=self.adapter_name)
        self.model.set_active_adapters(self.adapter_name)
        print(f"Active adapters: {self.model.active_adapters}")
        print(f"Loaded adapters: {list(self.model.adapters_config.adapters.keys())}")
