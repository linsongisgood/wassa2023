# coding=gbk

import os
import pandas as pd
import torch
from sklearn.metrics import classification_report
from mix_model_cat import xlm_robertamodel
import torch.nn.functional as F
from torch import nn
import numpy as np

path3 = './new_mlec_train.csv'
data3 = pd.read_csv(path3)
sentences3 = data3.Text
labels3 = pd.read_csv(path3, usecols=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                                      'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral'])
labels3 = np.array(labels3)

path4 = './new_mlec_dev.csv'
data4 = pd.read_csv(path4)
sentences4 = data4.Text
labels4 = pd.read_csv(path4, usecols=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                                      'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral'])
labels4 = np.array(labels4)


path = './mcec_train.csv'
path2 = './mcec_dev.csv'
data = pd.read_csv(path)
data2 = pd.read_csv(path2)
nclass = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'love': 5,
          'optimism': 6, 'pessimism': 7, 'sadness': 8, 'surprise': 9, 'trust': 10, 'neutral': 11}
sentences2 = data2.Text
labels2 = data2.Emotion
labels_to_ids2 = []
for label in labels2:
    labels_to_id = int(nclass[label])
    labels_to_ids2.append(labels_to_id)
labels2 = labels_to_ids2

labels_to_ids2 = np.eye(12)[labels_to_ids2]


sentences = data.Text
sentences = pd.concat([sentences, sentences3], axis=0)
sentences = pd.concat([sentences, sentences4], axis=0)
#sentences = pd.concat([sentences, sentences2], axis=0)
print(len(sentences))

labels = data.Emotion
labels_to_ids = []
for i, label in enumerate(labels):
    labels_to_id = int(nclass[label])
    labels_to_ids.append(labels_to_id)
print(classification_report(labels_to_ids, labels_to_ids))

labels_t = torch.tensor(labels_to_ids)
labels_to_ids = np.eye(12)[labels_to_ids]

labels_to_ids = np.concatenate((labels_to_ids, labels3), axis=0) #axis=0��������ƴ�ӣ�axis=1���ź���ƴ��
labels_to_ids = np.concatenate((labels_to_ids, labels4), axis=0) #axis=0��������ƴ�ӣ�axis=1���ź���ƴ��
#labels_to_ids = np.concatenate((labels_to_ids, labels_to_ids2), axis=0) #axis=0��������ƴ�ӣ�axis=1���ź���ƴ��
print(len(labels_to_ids))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("M-CLIP/M-BERT-Distil-40")

max_len = 0
lengthOfsentence = []
# ѭ��ÿһ������...
for sent in sentences:
    lengthOfsentence.append(len(sent))
    # �ҵ�������󳤶�
    max_len = max(max_len, len(sent))
print(lengthOfsentence)
print('��ľ��ӳ���Ϊ: ', max_len)

input_ids = []
attention_masks = []
input_ids2 = []
attention_masks2 = []

for sent in sentences:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=90,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # �ѱ���ľ��Ӽ���list.
    input_ids.append(encoded_dict['input_ids'])

    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

for sent in sentences2:
    encoded_dict2 = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=90,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # �ѱ���ľ��Ӽ���list.
    input_ids2.append(encoded_dict2['input_ids'])

    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks2.append(encoded_dict2['attention_mask'])
# ��lists תΪ tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels_to_ids)
input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
labels2 = torch.tensor(labels2)

from torch.utils.data import TensorDataset, random_split

# ��input ���� TensorDataset��
dataset = TensorDataset(input_ids, attention_masks, labels, labels)
val_dataset = TensorDataset(input_ids2, attention_masks2, labels2, labels2)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 110
train_dataloader = DataLoader(
    dataset,  # ѵ������.
    sampler=RandomSampler(dataset),  # ����˳��
    batch_size=batch_size
)
train_dataloader1 = DataLoader(
    dataset,  # ѵ������.
    sampler=RandomSampler(dataset),  # ����˳��
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,  # ��֤����.
    # sampler = RandomSampler(val_dataset), # ����˳��
    batch_size=batch_size
)
from transformers import XLMRobertaForSequenceClassification, AdamW, XLMRobertaConfig, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaConfig

# config = AutoConfig.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment')
# config = XLMRobertaConfig.from_json_file("./ensemble_cpt6/config.json")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
model = xlm_robertamodel()
model.to('cuda:0')
model = nn.DataParallel(model)

# AdamW ��һ�� huggingface library ���࣬'W' ��'Weight Decay fix"����˼��
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - Ĭ���� 5e-5
                  eps=1e-8  # args.adam_epsilon  - Ĭ���� 1e-8�� ��Ϊ�˷�ֹ˥���ʷ�ĸ����0
                  )

from transformers import get_linear_schedule_with_warmup

epochs = 30

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

'''if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
'''
output_dir = "./new3"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# �����������.
seed_val = 62
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# ��¼training ,validation loss ,validation accuracy and timings.
training_stats = []

# ������ʱ��.
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

    # ��¼ÿ�� epoch ���õ�ʱ��
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    labels_pred = []
    labels_pred1 = []
    for step, (batch, batch1) in enumerate(zip(train_dataloader, train_dataloader1)):

        # ÿ��40��batch ���һ������ʱ��.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        lbeta = np.random.beta(0.75, 0.75)
        #lbeta = 1
        # `batch` ����3�� tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to('cuda:0')
        b_input_mask = batch[1].to('cuda:0')
        b_labels = batch[2].to('cuda:0')
       # labels = batch[3].to('cuda:1')
        b_input_ids1 = batch1[0].to('cuda:0')
        b_input_mask1 = batch1[1].to('cuda:0')
        b_labels1 = batch1[2].to('cuda:0')
        # ����ݶ�
        model.zero_grad()

        # forward
        outputs = model(b_input_ids,
                        b_input_mask,
                        b_input_ids1,
                        b_input_mask1,
                        lbeta)
        logits = outputs[0]
        mixed_label = lbeta * b_labels + (1 - lbeta) * b_labels1
        #mixed_label = lbeta * b_labels + lbeta * b_labels1
        loss = F.kl_div(logits.to(torch.double), mixed_label.to(torch.double),
                        None, None, 'batchmean')
        total_train_loss += loss.item()

        # backward ���� gradients.
        loss.backward()

        # ��ȥ����1 ���ݶȣ�������Ϊ 1.0, �Է��ݶȱ�ը.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ����ģ�Ͳ���
        optimizer.step()

        # ���� learning rate.
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    # ����ѵ��ʱ��.
    training_time = format_time(time.time() - t0)
    # ѵ������׼ȷ��.
    print("  ƽ��ѵ����ʧ loss: {0:.2f}".format(avg_train_loss))
    print("  ѵ��ʱ��: {:}".format(training_time))

    # ______________________________________val________________________________________________

    t0 = time.time()
    # ���� model Ϊvaluation ״̬����valuation״̬ dropout layers ��dropout rate�᲻ͬ
    model.eval()
    # ���ò���
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to('cuda:0')
        b_input_mask = batch[1].to('cuda:0')
        b_labels = batch[2].to('cuda:0')

        # ��valuation ״̬��������Ȩֵ�����ı����ͼ
        with torch.no_grad():
            outputs2 = model(b_input_ids,
                             attention_mask=b_input_mask,
                             )

        logits2 = outputs2[0]
        # ���� validation loss.
        criterion = nn.CrossEntropyLoss()
        loss2 = criterion(logits2.view(-1, 12), b_labels.view(-1))
        total_eval_loss += loss2.item()
        logit = logits2.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_pred1 = np.append(labels_pred1, pred_flat)
        # ���� validation ���ӵ�׼ȷ��.
        total_eval_accuracy += flat_accuracy(logit, label_id)

    labels_pred1 = labels_pred1.flatten()
    print(labels_pred1)
    print(classification_report(labels2, labels_pred1))
    # ���� validation ��׼ȷ��.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  ����׼ȷ��: {0:.2f}".format(avg_val_accuracy))

    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), output_model_file)
        # model.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(output_dir)

    # ����batches��ƽ����ʧ.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # ����validation ʱ��.
    validation_time = format_time(time.time() - t0)

    print("  ƽ��������ʧ Loss: {0:.2f}".format(avg_val_loss))
    print("  ����ʱ��: {:}".format(validation_time))

print("ѵ��һ������ {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))