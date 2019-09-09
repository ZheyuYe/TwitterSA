#!/usr/bin/env python
# coding: utf-8
# In[1]:


from pytorch_transformers import BertTokenizer, BertModel,BertForSequenceClassification, AdamW, BertConfig
# from pytorch_transformers import *
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from enum import Enum
import csv
import sys
import os
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import emoji
from replace_emoji import topEmojis,replace_emojis
from datetime import datetime
# logging.basicConfig(level=logging.INFO)
from main import test_predictions
from pytorch_transformers import BertConfig,AdamW,WarmupLinearSchedule,WarmupConstantSchedule,WarmupCosineSchedule
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, WeightedRandomSampler
from train_evalute import do_train, evaluate
from dataloader import dataloader,get_data
from config import *
from pretrained.BertOrigin import BertOrigin
from pretrained.BertCNN import BertCNN
from pretrained.BertAttn import BertAttn
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import re


def pltout(step,sentence):
    text = list(input_ids[sentence].detach().cpu().numpy())
    zero = text.index(0)
    tokens = model.tokenizer.convert_ids_to_tokens(text)[:zero]
    # print(tokens)
    # indexs = re.split(r'!| ',)
    test_attention = attention_text[-1][sentence]
    sum_attention = test_attention[0]
    sum_attention = torch.mean(test_attention,dim=0)[:zero,:zero]
    # sum_attention = torch.sum(sum_attention,dim=1)
#     print(sum_attention.shape)
    print_attention = sum_attention.detach().cpu().numpy()

    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 23}

    df = pd.DataFrame(print_attention, columns=tokens, index=tokens)
    fig = plt.figure()
    fig.set_size_inches(8,8)
    ax = fig.add_subplot(111)
    cax = ax.matshow(df, interpolation='nearest', cmap='gist_yarg')
    fig.colorbar(cax,shrink=0.8)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    #     tick.label.set_fontsize(14)
    # for tick in ax.get_yticklabels():
    #     tick.label.set_fontsize(14)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig(f"./figures/{step}_{sentence}.png")
    # plt.show()


model_dir = './B96_lr1e-06_s1.0_0903_1905/'
config = BertConfig(num_labels=3,output_attentions = True)
# PRETRAINED_WEIGHTS = "bert-base-cased"
config.from_pretrained('bert-base-cased')
model = BertAttn(config, option='emoji', dropout=0.1,
            gpu=False, seed=0, do_lower_case=False)
# model.set_focal_loss(alpha=class_weights,gamma=-1)
model.load_model(True,model_dir)
# model.bert.save_pretrained('./bert-cased/')

class_weights,train, dev, test = get_data(
        option='emoji', dataset_size=1,unbalanced=False)


# In[8]:


x_train, _ , y_train = train
x_dev, _ , y_dev = dev
x_test, emoji_test, y_test = test


# In[9]:


model.set_focal_loss(alpha=class_weights,gamma=-1)
print(model.device)
# test_predictions(model, test,"test.csv", batch_size=96)


# In[10]:


from dataloader import generate_features
xdata, emojidata, ydata = train
features = generate_features(xdata, ydata, 128, model.tokenizer,emojidata)
input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
emoji_ids = torch.tensor([f.emoji_ids for f in features], dtype=torch.long)
emoji_mask = torch.tensor([f.emoji_mask for f in features], dtype=torch.long)


# In[ ]:


# torch.cuda.empty_cache()


# In[11]:


data = TensorDataset(input_ids, input_mask, label_ids,emoji_ids,emoji_mask)
dataloader = DataLoader(data, batch_size=32,shuffle=True)


# In[12]:


# print(model.tokenizer.encode('hello :) 😍'))
# emojis_ids = convert_emojitokens('hello :) 😍',20,model.tokenizer)
# emojis_ids = torch.tensor([emojis_ids],dtype=torch.long)


# In[17]:


for step, batch_data in enumerate(tqdm(dataloader, desc="Iteration")):
    batch_data = tuple(t.to(model.device) for t in batch_data)
    input_ids, input_mask, gnd_labels,emoji_ids, emoji_mask = batch_data
    train_loss,logits,attention_text, attention_emoji = model(input_ids, attention_mask = input_mask,labels = gnd_labels,emoji_ids = emoji_ids,emoji_mask = emoji_mask)
    for i in range(32):
        text = list(input_ids[i].detach().cpu().numpy())
        label = gnd_labels[i].detach().cpu().numpy()
        zero = text.index(0)
        if zero < 15 and label!=1:
            pltout(step,i)
