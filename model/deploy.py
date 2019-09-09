import torch
import sys
from pytorch_transformers import BertConfig,AdamW,WarmupLinearSchedule,WarmupConstantSchedule,WarmupCosineSchedule
from dataloader import convert_tokens
from torch import nn
from config import *
from pretrained.BertAttn import BertAttn
from preprocessing import clean_tweet, remove_special_characters
from replace_emoji import topEmojis,replace_emojis
sys.path.append('../nginx_sites/py')
from vader import sentiment_analyzer_scores
# print(os.getcwd())

def load_model():
    model_dir = '../../model/model/'
    config = BertConfig(num_labels=3,output_attentions = True)
    config.from_pretrained('../../model/bert-cased/')
    model = BertAttn(config, option='feed', dropout=0.1,
                gpu=False, seed=0, do_lower_case=False)
    class_weights = [0.6058, 0.1161, 0.2781]
    model.set_focal_loss(alpha=class_weights,gamma=-1)
    model.load_model(True,model_dir)
    return model
model = load_model()
top_occur = topEmojis()

def get_ids(text,emoji,max_seq_length,tokenizer):
    input_ids,input_mask = convert_tokens(text,max_seq_length,tokenizer)
    emoji_ids, emoji_mask = convert_tokens(emoji,max_seq_length,tokenizer,True)
def get_scores(string):
    clearedTweet = clean_tweet(string)
    sentiment = sentiment_analyzer_scores(clearedTweet)
    # print(string)
    # print('VADER:',sentiment)
    if sentiment['compound']>0.6 or sentiment['compound']<-0.6:
        return sentiment
    else:
        clearedTweet, emoji_list = replace_emojis(clearedTweet,top_occur)
        removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
        input_ids,input_mask = convert_tokens(clearedTweet,MAX_SEQ_LENGTH,model.tokenizer)
        emoji_ids, emoji_mask = convert_tokens(emoji_list,MAX_SEQ_LENGTH,model.tokenizer,True)
        input_ids_one=torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
        input_mask_one=torch.tensor(input_mask,dtype=torch.long).unsqueeze(0)
        emoji_ids_one=torch.tensor(emoji_ids,dtype=torch.long).unsqueeze(0)
        emoji_mask_one=torch.tensor(emoji_mask,dtype=torch.long).unsqueeze(0)
        logits,attention_text, attention_emoji = model(input_ids_one, attention_mask = input_mask_one,emoji_ids = emoji_ids_one,emoji_mask = emoji_mask_one)
        softmaxed_logits= nn.Softmax(dim=1)(logits).squeeze(0).tolist()
        softmaxed_logits = [round(a, 3) for a in softmaxed_logits]
        compound = round(-softmaxed_logits[0]**0.8+softmaxed_logits[2]**0.8,3)
        results = {'neg':softmaxed_logits[0],'neu':softmaxed_logits[1],'pos':softmaxed_logits[2],'compound':compound}
        # print('FEA-BERT:',results)
        return results

# get_scores("At least it is't a horrible book.")
