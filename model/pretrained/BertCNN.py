import torch
import torch.nn.functional as F
from torch import nn
from pytorch_transformers import BertPreTrainedModel, BertModel,BertTokenizer
# from torch.autograd import Variable
from Footstone import Footstone
from config import *

from Modules.Conv1d_pool import Conv1d_pool
from Modules.Linear import Linear

class BertCNN(Footstone,BertPreTrainedModel):

    def __init__(self, config, option, dropout, gpu, seed, do_lower_case, n_filters=128, filter_sizes=[1,3,5,7,9]):
        super(BertCNN, self).__init__(config,option, gpu,seed)

        #basic Config
        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(get_tokenizer(do_lower_case), do_lower_case=do_lower_case)
        self.resize_token_embeddings(len(self.tokenizer))
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        #output Config
        self.dropout = nn.Dropout(dropout)
        self.convs = Conv1d_pool(config.hidden_size, n_filters, filter_sizes)
        self.classifier = nn.Linear(len(filter_sizes) * n_filters, self.num_labels)
        # self.classifier = kaiming_Linear(len(filter_sizes) * n_filters, self.num_labels)

        # self.apply(self.init_weights)
        self.init_device()
        self.to(self.device)
        print(config)

    def forward(self, input_ids, attention_mask=None, labels=None, emoji_ids=None,emoji_mask=None):
        """
        Args:
            input_ids: corresponding id for word embeddings
            attention_mask: differentiate padding and token, 0 for padding
        """
        #outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # The last hidden-state
        position_ids = torch.LongTensor(range(MAX_SEQ_LENGTH)).to(self.device)
        outputs = self.bert(input_ids,attention_mask=attention_mask,position_ids=position_ids)
        # outputs = self.bert(input_ids,attention_mask=attention_mask)
        encoded_layers = outputs[0]
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        encoded_layers = self.dropout(encoded_layers)
        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim=768, seq_len]

        conved = self.convs(encoded_layers)
        # conved 是一个列表， conved[0]: [batch_size, filter_num, *]

        # torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #using seq_len as a stride, get most effective features from whole sequence for each CNN
        CNN_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # CNN_pool 是一个列表， CNN_pool[0]: [batch_size, filter_num]

        cat = self.dropout(torch.cat(CNN_pool, dim=1))
        # cat: [batch_size, filter_num * len(filter_sizes)]
        logits = self.classifier(cat)
        # logits: [batch_size, output_dim]
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # x = Variable(logits.view(-1, self.num_labels).cuda())
            # l = Variable(labels.view(-1).cuda())
            loss = self.loss_fct(logits.view(-1, self.num_labels),labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
