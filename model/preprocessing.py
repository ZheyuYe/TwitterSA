import csv
import os
import sys
import re
from replace_emoji import topEmojis,replace_emojis
from sklearn.metrics import f1_score

import emoji
import random
import argparse

sys.path.append('../nginx_sites/py')
from vader import sentiment_analyzer_scores

def get_dataset_name():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-o','--option',type=str,help="Dataset to be preprocessd",choices=['sentiment140','SemEval2017','emoji','emoji2','status'])

    args = parser.parse_args()
    return args

INFILE={
    'emoji':'../data/dataset.csv',
    'emoji2':'../data/dataset.csv',
    'sentiment140':'../data/training.1600000.processed.noemoticon.csv',
    'SemEval2017':'../data/SemEval2017-task4/GOLD/Subtask_A/',
    'status':'../data/status.csv'
}

OUTPUTFILE={
    'emoji' :"../data/3-emoji.csv",
    'emoji2' : "../data/2-emoji.csv",
    'sentiment140':'../data/sentiment_train_set.csv',
    'SemEval2017':'../data/SemEval2017-task4-preprocessed/'
    }
#HappyEmoticons 😄
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
# Sad Emoticons ☹️
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

contraction_mapping = {"ain't": "is not", "aren't": "are not", "arent": "are not", "can't": "cannot",
               "can't've": "cannot have", "'cause": "because", "could've": "could have",
               "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",
               "doesn't": "does not", "don't": "do not", "dont": "do not", "hadn't": "had not", "hadnt": "had not",
               "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
               "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
               "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
               "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
               "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
               "I'll've": "I will have","I'm": "I am", "I've": "I have",
               "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
               "i'll've": "i will have","i'm": "i am", "i've": "i have",
               "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
               "it'll": "it will", "it'll've": "it will have","it's": "it is",
               "let's": "let us", "ma'am": "madam", "mayn't": "may not",
               "might've": "might have","mightn't": "might not","mightn't've": "might not have",
               "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
               "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
               "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
               "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
               "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
               "she's": "she is", "should've": "should have", "shouldn't": "should not",
               "shouldn't've": "should not have", "so've": "so have","so's": "so as", "shouldvetaken": "should have taken",
               "this's": "this is", "doesnt": "does not",
               "that'd": "that would", "that'd've": "that would have","that's": "that is",
               "there'd": "there would", "there'd've": "there would have","there's": "there is",
               "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
               "they'll've": "they will have", "they're": "they are", "they've": "they have",
               "to've": "to have", "wasn't": "was not", "we'd": "we would",
               "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
               "we're": "we are", "we've": "we have", "weren't": "were not",
               "what'll": "what will", "what'll've": "what will have", "what're": "what are",
               "what's": "what is", "what've": "what have", "when's": "when is",
               "when've": "when have", "where'd": "where did", "where's": "where is",
               "where've": "where have", "who'll": "who will", "who'll've": "who will have",
               "who's": "who is", "who've": "who have", "why's": "why is",
               "why've": "why have", "will've": "will have", "won't": "will not",
               "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
               "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
               "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
               "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
               "you'll've": "you will have", "you're": "you are", "you've": "you have", "youre": "you are"}
def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence
def remove_special_characters(string,remove_digits=False,non_ASCII=False):
    #replace consecutive non-ASCII characters with a space including emoji!!!!
    if non_ASCII:
        string = re.sub(r'[^\x00-\x7F]+',' ', string)
    if remove_digits:
        string = re.sub(r'[0-9]+', ' ',string)
        #remove all special characters punctuation expect exclamation mark
    string = re.sub(r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）',' ',string)
    string = re.sub(" +", " ", string)
    return string.strip()
def clean_tweet(tweet):
    tweet = re.sub(r'[\r|\n|\r\n]+', ' ',tweet)
    tweet = re.sub(r'https?:\/\/[^ ]+',' ', tweet)
    tweet = re.sub(r'www.[^ ]+',' ',tweet)
    tweet = re.sub(r"[‘’]","\'",tweet)
    tweet = expand_contractions(tweet, contraction_mapping)

    tweet = re.sub(r'@[A-Za-z0-9_]+',' ',tweet)
    tweet = re.sub(r"_"," ",tweet)
    # remove repeat characters to match emoticons
    tweet = re.sub(r':+', ':', tweet)
    tweet = re.sub(r'\(+', '(', tweet)
    tweet = re.sub(r'\)+', ')', tweet)
    tweet = re.sub(r'-+', '-', tweet)
    tweet = repalce_emoticon(tweet)
    return tweet.strip()

def repalce_emoticon(tweet):
    for emoticon in emoticons_happy:
        tweet.replace(emoticon,':)')
    for emoticon in emoticons_sad:
        tweet.replace(emoticon,':(')
    return tweet

def data_preprocess(option):
    foutput = OUTPUTFILE[option]
    fin = INFILE[option]
    top_occur = topEmojis()
    count={'positive':0,'neutral':0,'negative':0}
    # translate={'positive':2,'negative':0,'neutral':1}

    data = []
    with open(fin) as csvfile:
        csv_reader = csv.reader(csvfile)
        #"id","tweet_id","user_id","screen_name","user_name","location","content","created_time","sentiment"
        data_header = next(csv_reader)  # first line of csvfile
        for row in csv_reader:
            clearedTweet = clean_tweet(row[6])
            #send to vader first
            sentiment = sentiment_analyzer_scores(clearedTweet)['compound']
            if option=='emoji' and (sentiment>0.6 or sentiment<-0.6 or -0.05<sentiment<0.05):
                if sentiment>0.6:
                    count['positive']+=1
                    sentiment = 2
                elif sentiment<-0.6:
                    count['negative']+=1
                    sentiment = 0
                else:
                    count['neutral']+=1
                    sentiment = 1
                #then extract emojis and emoticons, save into dataset
                clearedTweet,emoji_list = replace_emojis(clearedTweet,top_occur)
                removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
                data.append([removedTweet,sentiment,emoji_list])

            if option=='emoji2' and (sentiment>0.6 or sentiment<-0.6):
                if sentiment>0.6:
                    count['positive']+=1
                    sentiment = 1
                elif sentiment<-0.6:
                    count['negative']+=1
                    sentiment = 0
                #then extract emojis and emoticons, save into dataset
                clearedTweet,emoji_list = replace_emojis(clearedTweet,top_occur)
                removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
                data.append([removedTweet,sentiment,emoji_list])
    writer = csv.writer(open(foutput, 'w'))
    writer.writerow(['twitter','sentiment','emoji'])
    ##randomlize the order of data
    random.shuffle(data)
    writer.writerows(data)
    print(count)
    print(len(data))

def sentiment140(option):
    """
    for sentiment140 dataset
    0 = negative, 4 = positive
    """
    fout = OUTPUTFILE[option]
    fin = INFILE[option]
    count={'positive':0,'negative':0,'neutral':0}
    with open(fin, buffering=200000, encoding='latin-1') as f:
        data = []
        try:
            for line in f:
                line = line.replace('"', '')
                sentiment = line.split(',')[0]
                if sentiment == '4':
                    sentiment = 1
                    count['positive'] +=1
                elif sentiment =='0':
                    sentiment = 0
                    count['negative'] +=1
                # else:
                    # sentiment = 0
                    # count['neutral'] +=1
                tweet = line.split(',')[-1].strip()
                clearedTweet = clean_tweet(tweet)
                removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
                data.append([removedTweet,sentiment])
        except Exception as e:
            print(str(e))

    writer = csv.writer(open(fout, 'w'))
    writer.writerow(['tweet','sentiment'])
    #randomlize the order of data
    random.shuffle(data)
    writer.writerows(data)
    print(count)
    print(len(data))

def SemEval2017(option):
    """
    for SemEval2017-task4 dataset
    negative, positive, neutral
    """
    # print(os.getcwd())
    fout = OUTPUTFILE[option]
    fin = INFILE[option]
    paths = []
    for years in ['2014']:
        # for types in ['train','dev','test']:
        for types in ['test','train']:
            path = 'twitter-'+years+types+'-A.txt'
            paths.append(path)
    # paths.append('twitter-2016devtest-A.txt')
    # paths.append('SemEval2017-task4-test.subtask-A.english.txt')
    # paths.append('SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
    translate={'positive':2,'negative':0,'neutral':1}
    top_occur = topEmojis()
    for path in paths:
        count={'positive':0,'negative':0,'neutral':0}
        with open(fin+path, buffering=200000, encoding='utf-8') as f:
            data = []
            try:
                for line in f:
                    splited = line.split('\t')
                    # id=splited[0]
                    # sentiment,tweet  = splited[-2:]
                    # splited = line.replace(r'\t',' ').split(' ')
                    id = splited[0]
                    sentiment = splited[1]
                    tweet  = ' '.join(splited[2:])
                    tweet = tweet.strip()
                    # print(id,sentiment,tweet)
                    count[sentiment]+=1
                    clearedTweet = clean_tweet(tweet)
                    clearedTweet,emoji_list = replace_emojis(clearedTweet,top_occur)
                    removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
                    data.append([removedTweet,translate[sentiment],emoji_list])

            except Exception as e:
                print(str(e))
            path = path.replace('txt','csv')
            writer = csv.writer(open(fout+path, 'w'))
            writer.writerow(['tweet','sentiment'])
            #randomlize the order of data
            # random.shuffle(data)
            writer.writerows(data)
            print(f'file name:{path} Total number:{len(data)} Distribution: {count}')
def status(option):
    fin = INFILE[option]
    data = []
    top_occur = topEmojis()
    accuracy = 0
    vader_count = 0
    count ={'positive':0,'negative':0,'neutral':0}

    # "content","pos","spos","neu","sneg","neg","count","pos_mask","spos_mask","neu_mask","sneg_mask","neg_mask","count_mask"
    inverse_translate={2:'positive',0:'negative',1:'neutral'}
    gnd_labels = []
    vader_predict = []
    with open(fin) as csvfile:
        csv_reader = csv.reader(csvfile)
        data_header = next(csv_reader)  # first line of csvfile
        for line in csv_reader:
            # print(line)
            # line = line.replace('"', '').split(',')
            content = line[0]
            pos = int(line[1])+int(line[7])
            spos = int(line[2])+int(line[8])
            neu = int(line[3])+int(line[9])
            sneg = int(line[4])+int(line[10])
            neg = int(line[5])+int(line[11])
            tweet_count = int(line[6])+int(line[12])
            if tweet_count>1 and pos+spos+sneg+neg+neu==tweet_count:
                f_pos = pos+0.5*spos
                f_neu = neu+0.5*spos+0.5*sneg
                f_neg = neg+0.5*sneg
                # translate={'positive':2,'negative':0,'neutral':1}

                sentiment_list = [f_neg,f_neu,f_pos]
                # sentiment_list = [f_neg,f_pos]

                sentiment = sentiment_list.index(max(sentiment_list))
                count[inverse_translate[sentiment]]+=1
                clearedTweet = clean_tweet(content)

        #         vader_sentiment = sentiment_analyzer_scores(clearedTweet)['compound']
        #         # print(vader_sentiment)
        #         if vader_sentiment>0.6:
        #             vader_sentiment = 2
        #             vader_predict.append(vader_sentiment)
        #             gnd_labels.append(sentiment)
        #
        #         elif vader_sentiment<-0.6:
        #             vader_sentiment = 0
        #             vader_predict.append(vader_sentiment)
        #             gnd_labels.append(sentiment)
        #         elif -0.05<vader_sentiment<0.05:
        #             vader_sentiment = 1
        #             vader_predict.append(vader_sentiment)
        #             gnd_labels.append(sentiment)
        # print(gnd_labels)
        # print(len(gnd_labels))
        # accuracy,macro_f1,micro_f1 = get_metrics(vader_predict,gnd_labels)
        # print(accuracy,macro_f1,micro_f1)
                clearedTweet,emoji_list = replace_emojis(clearedTweet,top_occur)
                removedTweet = remove_special_characters(clearedTweet,remove_digits=True,non_ASCII=True)
                data.append([removedTweet,sentiment,emoji_list])


    # print(data)
    path = 'status_extracted.csv'
    writer = csv.writer(open(path, 'w'))
    writer.writerow(['content','sentiment'])
    random.shuffle(data)
    writer.writerows(data)
    print(f'file name:{path} Total number:{len(data)} Distribution: {count}')
def get_metrics(vader_predict,gnd_labels):
    accuracy = 0
    macro_f1 = f1_score(vader_predict, gnd_labels, average='macro')
    micro_f1 = f1_score(vader_predict, gnd_labels, average='micro')
    for i in range(len(vader_predict)):
        if vader_predict[i]==gnd_labels[i]:
            accuracy+=1
    return accuracy/len(vader_predict) ,macro_f1,micro_f1,
# def vader_test():
#     filename = '../data/sentiment_train_set.csv'
#     x = []
#     y = []
#     predictions = []
#     with open(filename) as csvfile:
#         csv_reader = csv.reader(csvfile)
#         data_header = next(csv_reader)  # first line of csvfile
#         for row in csv_reader:
#             tweet,sentiment = row
#             x.append(tweet)
#             y.append(tweet)
#             vader_sentiment = sentiment_analyzer_scores(tweet)['compound']
#             if vader_sentiment>0.6:
#                 vader_sentiment = 1
#                 vader_predict.append(vader_sentiment)
#                 gnd_labels.append(sentiment)
#             elif vader_sentiment<-0.6:
#     print(len(x[:16000]))

if __name__ == "__main__":
    args = get_dataset_name()
    if args.option == 'sentiment140':
        sentiment140(option=args.option)
    elif args.option == 'SemEval2017':
        SemEval2017(option=args.option)
    elif args.option == 'status':
        status(args.option)
    else:
        data_preprocess(option=args.option)

    # vader_test()
