import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

path = './mcec_train.csv'
path2 = './mcec_dev.csv'
data = pd.read_csv(path)
data2 = pd.read_csv(path2)
nclass = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'love':5,
           'optimism': 6, 'pessimism': 7, 'sadness': 8, 'surprise':9, 'trust':10, 'neutral':11}
sentences = data.Text
labels = data.Emotion
labels_to_ids = []
for label in labels:
    labels_to_id = int(nclass[label])
    labels_to_ids.append(labels_to_id)

sentences2 = data2.Text
labels2 = data2.Emotion
labels_to_ids2 = []
for label in labels2:
    labels_to_id = int(nclass[label])
    labels_to_ids2.append(labels_to_id)



from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

#tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
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

for sent in sentences2:
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
labels2 = torch.tensor(labels_to_ids2)

from torch.utils.data import TensorDataset, random_split

# 把input 放入 TensorDataset。
dataset = TensorDataset(input_ids, attention_masks, labels)
val_dataset = TensorDataset(input_ids2, attention_masks2, labels2)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 64
train_dataloader = DataLoader(
            dataset,  # 训练数据.
            #sampler=RandomSampler(dataset), # 打乱顺序
            batch_size=batch_size
        )
validation_dataloader = DataLoader(
            val_dataset, # 验证数据.
           # sampler = RandomSampler(val_dataset), # 打乱顺序
            batch_size = batch_size
        )

#from transformers import XLMRobertaForSequenceClassification, AdamW
#model = XLMRobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=12, ignore_mismatched_sizes=True)
from transformers import XLMRobertaForSequenceClassification, AdamW, XLMRobertaConfig
#model = XLMRobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=12, ignore_mismatched_sizes=True)
#config = XLMRobertaConfig.from_json_file("./models/config.json")
#model = XLMRobertaForSequenceClassification(config)
#model.load_state_dict(torch.load('./xlmroberta.pt'))
import os
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaConfig,AutoConfig
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

#config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
config = AutoConfig.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
config.num_labels = 12
model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis', config=config, ignore_mismatched_sizes=True)
model.cuda()
#model = nn.DataParallel(model)

# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - 默认是 5e-5
                  eps = 1e-8 # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                )

from transformers import get_linear_schedule_with_warmup

# bert 推荐 epochs 在2到4之间为好。
epochs = 25

# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np
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

output_dir = "./models4/"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

print(output_model_file)
# 设置随机种子.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 记录training ,validation loss ,validation accuracy and timings.
training_stats = []

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
    labels_pred=[]
    labels_pred1 =[]
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # `batch` 包括3个 tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # 清空梯度
        model.zero_grad()

        # forward
        # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss, logits = outputs[:2]

        total_train_loss += loss.item()

        # backward 更新 gradients.
        loss.backward()

        # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optimizer.step()

        # 更新 learning rate.
        scheduler.step()

        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # 计算training 句子的准确度.
        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_pred = np.append(labels_pred, pred_flat)
        total_train_accuracy += flat_accuracy(logit, label_id)

    labels_pred = labels_pred.flatten()
    labels_pred.astype('int32')
    print(labels_pred)
    print(classification_report(labels, labels_pred))
    avg_train_loss = total_train_loss / len(train_dataloader)
    # 计算训练时间.
    training_time = format_time(time.time() - t0)

    # 训练集的准确率.
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print("  训练准确率: {0:.2f}".format(avg_train_accuracy))
    print("  平均训练损失 loss: {0:.2f}".format(avg_train_loss))
    print("  训练时间: {:}".format(training_time))


    t0 = time.time()
    # 设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
    model.eval()
    # 设置参数
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # 在valuation 状态，不更新权值，不改变计算图
        with torch.no_grad():
             outputs2 = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
        loss2, logits2 = outputs2[:2]
        # 计算 validation loss.
        total_eval_loss += loss2.item()
        logit = logits2.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_pred1 = np.append(labels_pred1, pred_flat)
        # 计算 validation 句子的准确度.
        total_eval_accuracy += flat_accuracy(logit, label_id)

    labels_pred1 = labels_pred1.flatten()
    print(labels_pred1)
    print(classification_report(labels2, labels_pred1))
    # 计算 validation 的准确率.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  测试准确率: {0:.2f}".format(avg_val_accuracy))

    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)

    # 计算batches的平均损失.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # 计算validation 时间.
    validation_time = format_time(time.time() - t0)

    print("  平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
    print("  测试时间: {:}".format(validation_time))

    # 记录模型参数
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("训练一共用了 {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

#保存
#读取
#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))


#15个epoch达到99%，18个达到100%,但是验证集只有67%的准确率




