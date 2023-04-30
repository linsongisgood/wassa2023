import torch
from torch import nn
from transformers import AutoModel, XLMRobertaModel
from torch.nn import CrossEntropyLoss
from focalloss import focal_loss

class xlm_robertamodel(nn.Module):
    def __init__(self, config):
        super(xlm_robertamodel, self).__init__()
        self.h_size = config.hidden_size
        self.encoder = AutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment', return_dict=True, output_hidden_states=True )
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 12)

    def forward(self, input_ids, attention_mask, labels):

        outputs = self.encoder(input_ids, attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]
        hidden_states = self.dropout(cls_embeddings)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        loss_fct = focal_loss()
        loss = loss_fct(logits.view(-1, 12), labels.view(-1))
        return loss, logits


