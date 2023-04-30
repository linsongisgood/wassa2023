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

    # 把编码的句子加入list.
    input_ids.append(encoded_dict['input_ids'])

    # 加上 attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# 把lists 转为 tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

from torch.utils.data import TensorDataset, random_split

# 把input 放入 TensorDataset。
dataset = TensorDataset(input_ids, attention_masks)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 64
train_dataloader = DataLoader(
    dataset,  # 训练数据.
    # sampler=RandomSampler(dataset), # 打乱顺序
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


# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr=5e-5,  # args.learning_rate - 默认是 5e-5
                  eps=1e-8  # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                  )

from transformers import get_linear_schedule_with_warmup

# bert 推荐 epochs 在2到4之间为好。
epochs = 1
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs
# 设计 learning rate scheduler.
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
    # 返回 hh:mm:ss 形式的时间
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

# 设置随机种子.
#seed_val = 32
#random.seed(seed_val)
#np.random.seed(seed_val)
#torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# 设置总时间.
total_t0 = time.time()
# 记录每个 epoch 所用的时间
t0 = time.time()
labels_pred = []
# 设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
model.eval()
# 设置参数
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

for batch in train_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    # 在valuation 状态，不更新权值，不改变计算图
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
