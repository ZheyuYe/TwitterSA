#!/usr/bin/env python
# coding: utf-8

import torch
from pytorch_transformers import BertPreTrainedModel
import torch.nn as nn
import os
import random
import numpy as np
#focal loss implemented by https://github.com/clcarwin/focal_loss_pytorch
from FocalLoss import *
from torch.nn import CrossEntropyLoss
from replace_emoji import topEmojis,replace_emojis

class Footstone(BertPreTrainedModel):
    def __init__(self,config,option, gpu,seed):
        super(BertPreTrainedModel, self).__init__(config)
        self.gpu = gpu
        self.option = option
        self.device_count = 0

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)

    def init_device(self):
        if self.gpu:
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
        # self.to(self.device)
        self.device_count = torch.cuda.device_count()

    def set_focal_loss(self,alpha=None,gamma=0):
        if gamma >=0:
            self.loss_fct = FocalLoss(alpha=alpha,gamma=gamma)
        else:
            if alpha:
                weight=torch.Tensor(alpha)
                self.loss_fct = CrossEntropyLoss(weight=weight)
            else:
                self.loss_fct = CrossEntropyLoss()

        self.to(self.device)
        # print('loss function:',self.loss_fct)
    def save_model(self, path_model):
        #e.g. './results/B32_lr1e-06_s0.08/.pt'
        torch.save(self.state_dict(), path_model[:-1]+'.pt')

    def load_model(self, model_load,path_model):
        if model_load and path_model:
            # pretrained_dict=torch.load(path_model[:-1]+'.pt', map_location='cpu')
#             pretrained_dict=torch.load(path_model[:-1]+'.pt')
#             model_dict=self.state_dict()
#             # 1. filter out unnecessary keys
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             # 2. overwrite entries in the existing state dict
#             model_dict.update(pretrained_dict)
            # print('load model successfully from {}'.format(path_model))
            self.load_state_dict(torch.load(path_model[:-1]+'.pt', map_location='cpu'))
    def save_bert(self,path_bert):
                # Simple serialization for models and tokenizers
        if not os.path.isdir(path_bert):
            os.mkdir(path_bert)
            # save model
            self.save_pretrained(path_bert)
            # model = model_class.from_pretrained('./directory/to/save/')  # re-load

    def add_emoji_as_token(self):
        top_occur = topEmojis()
        # print(top_occur)
        num_added_toks = self.tokenizer.add_tokens(top_occur)
        print('We have added', num_added_toks, 'tokens')
        # string = " ".join(top_occur)
        # print(string)
        self.resize_token_embeddings(len(self.tokenizer))

        # self.apply(self.init_weights)
