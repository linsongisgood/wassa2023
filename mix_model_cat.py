import torch
from torch import nn
from transformers import AutoModel
import numpy as np
class xlm_robertamodel(nn.Module):
    def __init__(self):
        super(xlm_robertamodel, self).__init__()
        self.encoder = AutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment', return_dict=True, output_hidden_states=True)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 12)

    def forward(self, input_ids, attention_mask, input_ids1=None, attention_mask1=None, lbeta=None):
        outputs = self.encoder(input_ids, attention_mask)
        if input_ids1 is not None:
            outputs1 = self.encoder(input_ids1, attention_mask1)
            #cls_embeddings = outputs.last_hidden_state[:, 0]
            cls_embeddings = outputs.hidden_states[-1]
            cls_embeddings = cls_embeddings[:, 0]

            cls_embeddings1 = outputs1.hidden_states[-1]
            cls_embeddings1 = cls_embeddings1[:, 0]
            #cls_embeddings1 = outputs1.last_hidden_state[:, 0]
            cls_embeddings = lbeta * cls_embeddings + (1 - lbeta) * cls_embeddings1
            #cls_embeddings = lbeta * cls_embeddings + lbeta * cls_embeddings1
        else:
            #cls_embeddings = outputs.last_hidden_state[:, 0]
            cls_embeddings = outputs.hidden_states[-1]
            cls_embeddings = cls_embeddings[:, 0]

        hidden_states = self.dropout(cls_embeddings)
        hidden_states = torch.tanh(hidden_states)
        logits1 = self.out_proj(hidden_states)
        output = torch.sigmoid(logits1)
        probs_u = torch.softmax(logits1, dim=1)
        logits = probs_u.log()
        return logits, hidden_states, logits1, output


