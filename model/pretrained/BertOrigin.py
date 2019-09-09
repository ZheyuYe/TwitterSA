import torch
from torch import nn
from pytorch_transformers import BertPreTrainedModel, BertModel,BertTokenizer
# from torch.autograd import Variable
from Footstone import Footstone
from config import *

class BertOrigin(Footstone,BertPreTrainedModel):
    def __init__(self,config,option,dropout, gpu,seed,do_lower_case):
        super(BertOrigin, self).__init__(config,option, gpu,seed)

        #basicConfig
        self.bert = BertModel(config)
        #get tokenizer with emoji
        self.tokenizer = BertTokenizer.from_pretrained(get_tokenizer(do_lower_case), do_lower_case=do_lower_case)
        self.resize_token_embeddings(len(self.tokenizer))

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        #output Config
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.emoji_classifier = nn.Linear(self.hidden_size, self.num_labels)
        # self.ff_emoji = PositionwiseFeedForward(config.hidden_size, self.intermediate_size, dropout)
        # self.apply(self.init_weights)
        self.init_device()
        self.to(self.device)
        print(config)
    def forward(self, input_ids, attention_mask=None, labels=None, emoji_ids=None,emoji_mask=None):
        position_ids = torch.LongTensor(range(MAX_SEQ_LENGTH)).to(self.device)
        outputs = self.bert(input_ids,attention_mask=attention_mask,position_ids=position_ids)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        emoji_ids = emoji_ids.transpose(0, 1)[:1].transpose(0,1)
        emoji_embeddings = self.bert.embeddings(emoji_ids).squeeze(1)
        emoji_logits = self.emoji_classifier(emoji_embeddings)
        logits += emoji_logits

        if labels is not None:
            # x = Variable(logits.view(-1, self.num_labels).cuda())
            # l = Variable(labels.view(-1).cuda())
            loss = self.loss_fct(logits.view(-1, self.num_labels),labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

        # loss, logits, attentions = outputs
