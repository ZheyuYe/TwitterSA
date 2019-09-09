import torch
import torch.nn.functional as F
from torch import nn
from pytorch_transformers import BertPreTrainedModel, BertModel,BertTokenizer
# from torch.autograd import Variable
from Footstone import Footstone
from config import *
import math
from Modules.Linear import Linear

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class BertAttn(Footstone,BertPreTrainedModel):
    def __init__(self, config, option, dropout, gpu, seed, do_lower_case):
        super(BertAttn, self).__init__(config,option, gpu,seed)
        #basic Config
        # self.conf.hidden_dropout_prob = dropout
        # self.conf.attention_probs_dropout_prob = dropout
        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(get_tokenizer(do_lower_case), do_lower_case=do_lower_case)

        self.resize_token_embeddings(len(self.tokenizer))
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        # self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        # self.load_model(True,'./results/B96_lr1e-06_s1.0_0830_2305/')


        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.intermediate_size = 3072
        #output Config
        self.dropout = nn.Dropout(dropout)
        self.ff_emoji = PositionwiseFeedForward(config.hidden_size, self.intermediate_size, dropout)
        self.ff_text = PositionwiseFeedForward(config.hidden_size, self.intermediate_size, dropout)

        self.classifier_text = nn.Linear(self.hidden_size,self.num_labels)
        self.classifier_emoji = nn.Linear(self.hidden_size,self.num_labels)
        self.classifier_compound = nn.Linear(2,1)
        # for reload orginal model

        # self.apply(self.init_weights)
        self.init_device()
        self.to(self.device)
        # print(config)

    def calc_attention(self,query, key, value, attention_mask):
        all_query = self.transpose_for_scores(query)
        all_key = self.transpose_for_scores(key)
        all_value = self.transpose_for_scores(value)


        attention_scores = torch.matmul(all_query, all_key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(attention_scores.shape)
        # print(attention_mask.shape)

        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # print(attention_probs.shape)
        # print(value.shape)
        context_layer = torch.matmul(attention_probs, all_value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids, attention_mask=None, labels=None, emoji_ids=None, emoji_mask=None):
        """
        Args:
            input_ids: corresponding id for word embeddings
            attention_mask: differentiate padding and token, 0 for padding
        """
        emoji_mask = emoji_mask.transpose(0, 1)[:MAX_EMOJI_LENGTH].transpose(0,1)
        extended_attention_mask = emoji_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(extended_attention_mask)

        #outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        position_ids = torch.LongTensor(range(MAX_SEQ_LENGTH)).to(self.device)
        outputs = self.bert(input_ids,attention_mask=attention_mask,position_ids=position_ids,head_mask=None)
        # outputs = self.bert(input_ids,attention_mask=,position_ids=raneg(MAX_SEQ_LENGTH),head_mask=[1,1,1,1,0,0,0,0,0,0,0,0])

        sequence_output = outputs[0]
        # sequence_output: [batch_size, seq_len, bert_dim=768]
        pooled_output = outputs[1].unsqueeze(1)
        emoji_ids = emoji_ids.transpose(0, 1)[:MAX_EMOJI_LENGTH].transpose(0,1)
        emoji_embeddings = self.bert.embeddings(emoji_ids)
        # emoji_beddings= [batch_size, MAX_EMOJI_LENGTH, bert_dim=768]

        all_query = self.query(pooled_output)
        all_key = self.key(emoji_embeddings)
        all_value = self.value(emoji_embeddings)


        result, attention_score = self.calc_attention(all_query,all_key,all_value,attention_mask=extended_attention_mask)
        emoji_score = self.classifier_emoji(self.ff_emoji(result))
        text_score = self.classifier_text(self.ff_text(pooled_output))
        # [batch_size, 1, 3]

        compound = torch.cat([text_score,emoji_score], dim=1)
        # [batch_size, 2, 3]
        # print(compound.shape)

        # weights = [0.4,0.5]
        logits = self.classifier_compound(compound.transpose(1,2)).squeeze(2)
        # logits: [batch_size, 3]
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # x = Variable(logits.view(-1, self.num_labels).cuda())
            # l = Variable(labels.view(-1).cuda())
            loss = self.loss_fct(logits.view(-1, self.num_labels),labels.view(-1))
            outputs = (loss,) + outputs
        return outputs + (attention_score,)
