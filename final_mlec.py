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
labels4_t = np.array(labels4)


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

labels_to_ids2_t = np.eye(12)[labels_to_ids2]


sentences = data.Text
sentences = pd.concat([sentences, sentences3], axis=0)
#sentences = pd.concat([sentences, sentences4], axis=0)
sentences = pd.concat([sentences, sentences2], axis=0)
print(len(sentences))

labels = data.Emotion
labels_to_ids = []
for i, label in enumerate(labels):
    labels_to_id = int(nclass[label])
    labels_to_ids.append(labels_to_id)
print(classification_report(labels_to_ids, labels_to_ids))

labels_t = torch.tensor(labels_to_ids)
labels_to_ids = np.eye(12)[labels_to_ids]

labels_to_ids = np.concatenate((labels_to_ids, labels3), axis=0) #axis=0沿着纵轴拼接，axis=1沿着横轴拼接
#labels_to_ids = np.concatenate((labels_to_ids, labels4), axis=0) #axis=0沿着纵轴拼接，axis=1沿着横轴拼接
labels_to_ids = np.concatenate((labels_to_ids, labels_to_ids2_t), axis=0) #axis=0沿着纵轴拼接，axis=1沿着横轴拼接
print(len(labels_to_ids))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

max_len = 0
lengthOfsentence = []
# 循环每一个句子...
for sent in sentences:
    lengthOfsentence.append(len(sent))
    # 找到句子最大长度
    max_len = max(max_len, len(sent))
print(lengthOfsentence)
print('最长的句子长度为: ', max_len)

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

    # 把编码的句子加入list.
    input_ids.append(encoded_dict['input_ids'])

    # 加上 attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

for sent in sentences4:
    encoded_dict2 = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=90,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # 把编码的句子加入list.
    input_ids2.append(encoded_dict2['input_ids'])

    # 加上 attention mask (simply differentiates padding from non-padding).
    attention_masks2.append(encoded_dict2['attention_mask'])
# 把lists 转为 tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels_to_ids)
input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
labels2 = torch.tensor(labels2)
labels_to_ids2 = torch.tensor(labels4_t)

from torch.utils.data import TensorDataset, random_split

# 把input 放入 TensorDataset。
dataset = TensorDataset(input_ids, attention_masks, labels, labels)
val_dataset = TensorDataset(input_ids2, attention_masks2, labels_to_ids2, labels2)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 64
train_dataloader = DataLoader(
    dataset,  # 训练数据.
    sampler=RandomSampler(dataset),  # 打乱顺序
    batch_size=batch_size
)
train_dataloader1 = DataLoader(
    dataset,  # 训练数据.
    sampler=RandomSampler(dataset),  # 打乱顺序
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,  # 验证数据.
    # sampler = RandomSampler(val_dataset), # 打乱顺序
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

# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - 默认是 5e-5
                  eps=1e-8  # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                  )

from transformers import get_linear_schedule_with_warmup

epochs = 30

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
from sklearn.metrics import f1_score
import torch

'''if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
'''
output_dir = "./final_mlec1"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# 设置随机种子.
seed_val = 62
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 记录training ,validation loss ,validation accuracy and timings.
training_stats = []
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# 设置总时间.
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

    # 记录每个 epoch 所用的时间
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    labels_pred = []
    labels_pred1 = []
    for step, (batch, batch1) in enumerate(zip(train_dataloader, train_dataloader1)):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        lbeta = np.random.beta(0.75, 0.75)

        # `batch` 包括3个 tensors:
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
        # 清空梯度
        model.zero_grad()

        # forward
        outputs = model(b_input_ids,
                        b_input_mask,
                        b_input_ids1,
                        b_input_mask1,
                        lbeta)
        logits = outputs[2]
        mixed_label = b_labels + b_labels1
        criterion2 = nn.BCEWithLogitsLoss()
        loss2 = criterion2(logits, mixed_label)
        total_train_loss += loss2.item()

        # backward 更新 gradients.
        loss2.backward()

        # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optimizer.step()

        # 更新 learning rate.
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    # 计算训练时间.
    training_time = format_time(time.time() - t0)
    # 训练集的准确率.
    print("  平均训练损失 loss: {0:.2f}".format(avg_train_loss))
    print("  训练时间: {:}".format(training_time))

    # ______________________________________val________________________________________________

    t0 = time.time()
    # 设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
    model.eval()
    # 设置参数
    labels_pred = np.empty([0, 12])
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    avg_val_accuracy, val_acc, n = 0.0, 0.0, 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to('cuda:0')
        b_input_mask = batch[1].to('cuda:0')
        b_labels = batch[2].to('cuda:0')

        # 在valuation 状态，不更新权值，不改变计算图
        with torch.no_grad():
            outputs2 = model(b_input_ids,
                             attention_mask=b_input_mask,
                             )

        logits2 = outputs2[2]
        pred_flat = torch.where(logits2 > 0.5, 1, 0)
        pred_flat = pred_flat.to('cpu').numpy()
        labels_pred = np.append(labels_pred, pred_flat, axis=0)
        # 计算 validation loss.
        criterion2 = nn.BCEWithLogitsLoss()
        loss2 = criterion2(logits2, b_labels)
        total_eval_loss += loss2.item()
        avg_val_accuracy += (torch.where(outputs2[3] > 0.5, 1, 0) == b_labels).min(axis=1)[0].sum()
        n += len(b_labels)
    #predictions = torch.stack(predictions).detach().cpu()
    #labels = torch.stack(labels).detach().cpu()

    avg_val_accuracy = avg_val_accuracy / n
    print(avg_val_accuracy)
    f1 = f1_score(labels4_t, labels_pred, average="macro")  # 0.6
    print(f1)
    f2 = f1_score(labels4_t, labels_pred, average="micro")  # 0.66666666666666666
    print(f2)
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), output_model_file)
        # model.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(output_dir)'''

    # 计算batches的平均损失.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # 计算validation 时间.
    validation_time = format_time(time.time() - t0)

    print("  平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
    print("  测试时间: {:}".format(validation_time))

print("训练一共用了 {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


