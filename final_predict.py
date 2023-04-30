# coding=gbk


import pandas as pd
import torch
from sklearn.metrics import classification_report

path = './mcec_test.csv'
data = pd.read_csv(path)
nclass = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'love': 5,
          'optimism': 6, 'pessimism': 7, 'sadness': 8, 'surprise': 9, 'trust': 10, 'neutral': 11}
nclass1 = {v: k for k, v in nclass.items()}
sentences = data.Text
labels_to_ids = []
'''for label in labels:
   labels_to_id = int(nclass[label])
   labels_to_ids.append(labels_to_id)
labels = torch.tensor(labels_to_ids)'''

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

input_ids = []
attention_masks = []
for sent in sentences:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=90,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )

    # �ѱ���ľ��Ӽ���list.
    input_ids.append(encoded_dict['input_ids'])

    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# ��lists תΪ tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

from torch.utils.data import TensorDataset, random_split

# ��input ���� TensorDataset��
dataset = TensorDataset(input_ids, attention_masks)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 64
train_dataloader = DataLoader(
    dataset,  # ѵ������.
    # sampler=RandomSampler(dataset), # ����˳��
    batch_size=batch_size
)

from transformers import AdamW, AutoModel, AutoModelForSequenceClassification, AutoConfig
from transformers import XLMRobertaForSequenceClassification, AdamW, XLMRobertaConfig
from mix_model_cat import xlm_robertamodel
from torch import nn
# model = XLMRobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=12, ignore_mismatched_sizes=True)
#config = XLMRobertaConfig.from_json_file("./binary1/config.json")
#config = AutoConfig.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment')
model = xlm_robertamodel()
model = nn.DataParallel(model).cuda()
#model = XLMRobertaForSequenceClassification(config)
model.load_state_dict(torch.load('./new3/pytorch_model.bin'))


# AdamW ��һ�� huggingface library ���࣬'W' ��'Weight Decay fix"����˼��
optimizer = AdamW(model.parameters(),
                  lr=5e-5,  # args.learning_rate - Ĭ���� 5e-5
                  eps=1e-8  # args.adam_epsilon  - Ĭ���� 1e-8�� ��Ϊ�˷�ֹ˥���ʷ�ĸ����0
                  )

from transformers import get_linear_schedule_with_warmup

# bert �Ƽ� epochs ��2��4֮��Ϊ�á�
epochs = 1
# training steps ������: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs
# ��� learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # ���� hh:mm:ss ��ʽ��ʱ��
    return str(datetime.timedelta(seconds=elapsed_rounded))


import os
import random
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# �����������.
#seed_val = 32
#random.seed(seed_val)
#np.random.seed(seed_val)
#torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# ������ʱ��.
total_t0 = time.time()
# ��¼ÿ�� epoch ���õ�ʱ��
t0 = time.time()
labels_pred = []
# ���� model Ϊvaluation ״̬����valuation״̬ dropout layers ��dropout rate�᲻ͬ
model.eval()
# ���ò���
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

for batch in train_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    # ��valuation ״̬��������Ȩֵ�����ı����ͼ
    with torch.no_grad():
        logits2 = model(b_input_ids,
                        attention_mask=b_input_mask
                        )
    logit = logits2[0].detach().cpu().numpy()
    pred_flat = np.argmax(logit, axis=1).flatten()
    labels_pred = np.append(labels_pred, pred_flat)

labels_pred = labels_pred.flatten()
labels_pred.astype('int32')
#print(labels_pred)
#print(classification_report(labels, labels_pred))
true_label = []
for label in labels_pred:
    labels_to_id = nclass1[int(label)]
    true_label.append(labels_to_id)

data['Emotion'] = true_label
data.to_csv('./ensemble_result/predictions_MCECnew2.csv')
