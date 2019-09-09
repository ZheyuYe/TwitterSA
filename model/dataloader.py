import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, WeightedRandomSampler
from config import *
import csv
import numpy as np
import random

def split_emoji(x):
    text, emoji = zip(*x)
    return np.array(text),np.array(emoji)

def data_split(x, y, proportion, dataset=False):
    second_half_size = int(x.shape[0]*proportion)
    if not dataset:
        second_half_size = min(20000,second_half_size)
    return x[:-second_half_size],x[-second_half_size:],y[:-second_half_size],y[-second_half_size:]

def shuffle_data(x, y):
    '''
    used to shuffle the data
    '''
    c = list(zip(x, y))
    random.shuffle(c)
    x[:], y[:] = zip(*c)
    return x,y
def load_dataset(option, dataset_size=1,file_type=None):
    fname = DATAFILE[option]
    flist = []
    if option == 'SemEval2017':
        if file_type != 'train':
            flist.append(fname+"twitter-2017{}-A.csv".format(file_type))
        else:
            list = ['2013train','2013dev','2013test','2014test','2015test','2015train','2016train','2016devtest','2016devtest']
            for l in list:
                flist.append(fname+"twitter-{}-A.csv".format(l))
    else:
        flist=[fname]
    x = [] #tweets
    y = [] #labels
    maxlen = 0
    for fnamei in flist:
        with open(fnamei) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            data_header = next(csv_reader)  # 读取第一行每一列的标题

            for row in csv_reader:
                y.append(int(row[1]))
                if len(row)>2:
                    x.append((row[0],row[2]))
                    # if len(row[2])>0:
                        # print(row[2])
                        # count+=1
                else:
                    x.append((row[0],[]))
            maxlen = max(maxlen,len(row[0]))
            print(f'max length: {maxlen} for file {fname}')

    #used to shuffle the data
    x, y = shuffle_data(x, y)
    x, y = np.array(x), np.array(y)
    if dataset_size!=1:
        _,x,_, y = data_split(x, y, proportion = dataset_size,dataset=True)
    # x = (text,emoji)
    return np.array(x), np.array(y)

def sample_data(x_data,y_data):
    '''
    Artificially collect an unbalanced data set
    8:2 for 16,000 tweetswe
    '''
    x1, x2= 0, 0
    x,y = [],[]
    for i in range(len(x_data)):
        if x1<12800 or x2<3200:
            if y_data[i]==1 and x1<12800:
                x1+=1
                x.append(x_data[i])
                y.append(y_data[i])
            elif y_data[i]==0 and x2<3200:
                x2+=1
                x.append(x_data[i])
                y.append(y_data[i])
        else:
            break
    x, y = shuffle_data(x, y)
    x, y= np.array(x), np.array(y)
    print('sample data:', x.shape)
    return x, y

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id, emoji_ids=None,emoji_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.emoji_ids = emoji_ids
        self.emoji_mask = emoji_mask

def convert_tokens(x,max_seq_length,tokenizer,emoji=False):
    tokens = tokenizer.tokenize(x)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    #CLS label, SEP = split sentence
    if not emoji:
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #tokens that are NOT MASKED, 0 for MASKED tokens.
    # Mask to avoid performing attention on padding token indices.
#       if attention_mask is None:
#          attention_mask = torch.ones_like(input_ids)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    return input_ids,input_mask

def generate_features(x, y, max_seq_length, tokenizer,emoji=None):
    features = []

    for i in range(len(x)):

        input_ids,input_mask = convert_tokens(x[i],max_seq_length,tokenizer)
        emoji_ids, emoji_mask = convert_tokens(emoji[i],max_seq_length,tokenizer,True)
        label_id = y[i]
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                label_id=label_id,
                emoji_ids=emoji_ids,
                emoji_mask=emoji_mask))
    return features

def get_weights(label_ids):
    _, counts = np.unique(label_ids, return_counts=True)
    class_weights = [sum(counts)/c for c in counts]
    weights_sum = sum(class_weights)
    return_weights = [round(weight/weights_sum,4) for weight in class_weights]
    return class_weights, return_weights

def dataloader(data,max_seq_length, tokenizer, batch_size,is_sample=False):
    '''
    return data = input_ids, input_mask, label_ids, emoji_ids
    '''
    xdata, emojidata, ydata = data
    features = generate_features(xdata, ydata, max_seq_length, tokenizer,emojidata)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    emoji_ids = torch.tensor([f.emoji_ids for f in features], dtype=torch.long)
    emoji_mask = torch.tensor([f.emoji_mask for f in features], dtype=torch.long)
    # [101, 102]
    data = TensorDataset(input_ids, input_mask, label_ids, emoji_ids, emoji_mask)
    #balance weights for train_data

    if is_sample:
        class_weights,_ = get_weights(label_ids)
        example_weights = [class_weights[e] for e in label_ids]
        sampler = WeightedRandomSampler(example_weights, len(ydata))
        # sampler = SequentialSampler(data)
        # sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(data, batch_size=batch_size,shuffle=True)
    return dataloader

def get_data(option,dataset_size=1,unbalanced=True):

    if option != 'SemEval2017':
    # train:dev:test = 6:2:2
        test_size=0.2
        dev_size=0.25

        x , y = load_dataset(option,dataset_size)
        if unbalanced:
            x , y = sample_data(x,y)
        _,return_weights = get_weights(y)
        print(f'Total dataset size is {x.shape}, class weights:{return_weights}')

        #split traing set and testing set
        x_train, x_test, y_train, y_test = data_split(x, y, proportion = test_size)
        #split validation set
        x_train, x_dev, y_train, y_dev, = data_split(x_train, y_train, proportion = dev_size)

        #calculate weights
        _,train_weights = get_weights(y_train)
        _,test_weights = get_weights(y_test)
        _,dev_weights = get_weights(y_dev)

    else:

        x_train , y_train = load_dataset(option,file_type = 'train')
        x_dev , y_dev = load_dataset(option,file_type = 'dev')
        x_test , y_test = load_dataset(option,file_type = 'test')

        #calculate weights
        _,train_weights = get_weights(y_train)
        _,test_weights = get_weights(y_test)
        _,dev_weights = get_weights(y_dev)

        y_shape = y_train.shape[0] + y_test.shape[0] + y_dev.shape[0]

        return_weights = (train_weights*np.array([y_train.shape[0]]) +
            test_weights * np.array([y_test.shape[0]]) + dev_weights*np.array([y_dev.shape[0]]))/np.array([y_shape])

        return_weights = list([round(i,4) for i in return_weights])
        print(f'Total dataset size is {y_shape}, class weights:{return_weights}')

    print(f'Train set size is {x_train.shape}, Class weights:{train_weights}')
    print(f'Dev set size is {x_dev.shape}, Class weights:{dev_weights}')
    print(f'Test set size is {x_test.shape}, Class weights:{test_weights}')

    text_train, emoji_train = split_emoji(x_train)
    text_dev, emoji_dev = split_emoji(x_dev)
    text_test, emoji_test = split_emoji(x_test)

    return test_weights,[text_train,emoji_train,y_train],[text_dev,emoji_dev,y_dev],[text_test,emoji_test,y_test]

# train,dev,test = get_data('sentiment140',0.00001)
# text,emoji,label = train
# print(label)
# x , y = load_dataset('emoji',1)
