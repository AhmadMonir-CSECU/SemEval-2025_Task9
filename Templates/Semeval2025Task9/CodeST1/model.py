import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from long_seq import process_long_input

class AIModel(nn.Module):
    def __init__(self, config, model, num_class=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.num_classes = num_class
        self.hidden_size = config.hidden_size
        self.loss_fct = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def encode(self, input_ids, attention_mask):
        config = self.config
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
        sequence_output = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                ):

        sequence_output = self.encode(input_ids, attention_mask)
        pooled_outputs = sequence_output[:,0,:]
        
        # pooled_outputs = torch.cat((pooled_output,pooled_output),dim=-1)
        logits = self.classifier(pooled_outputs)
        result = {}
        result["logits"] = logits
        
        if labels is not None:
          labels = torch.tensor(labels)
          labels = labels.to(input_ids)
          loss = self.loss_fct(logits, labels)
          result["loss"] = loss
        
        return result