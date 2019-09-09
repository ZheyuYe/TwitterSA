import csv
import re
import numpy as np
import emoji
import random

def emoji_regex():
    return re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def get_emojiDict(emoji_file = '../data/Emojis/Emoji_Sentiment_Data_v1.0.csv'):
    Emojis = []
    with open(emoji_file) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        data_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:
            Emojis.append([codepoint2unicode(row[1]),row[-2],row[2],row[4],row[5],row[6]])
    return np.array(Emojis)
def codepoint2unicode(codepoint):
    code = r'\U'+'0'*(10-len(codepoint))+codepoint[2:]
    return code.encode().decode('unicode_escape')
def feed_emoji(emoji_file = '../data/Emojis/Emoji_Sentiment_Data_v1.0.csv'):
    top_occur = topEmojis()
    print(top_occur)
    emoji_dict = get_emojiDict()
    data = []
    for emoji in emoji_dict[:55]:
        code = emoji[0]
        print(code)
        for i in range(int(emoji[3])):
            #Negative
            data.append(['',0,code])
        for i in range(int(emoji[4])):
            data.append(['',1,code])
        for i in range(int(emoji[5])):
            data.append(['',2,code])
    for i in range(10000):
        data.append(['',2,':)'])
        data.append(['',0,':()'])

    writer = csv.writer(open('./feed_emoji.csv', 'w'))
    writer.writerow(['twitter','sentiment','emoji'])
    ##randomlize the order of data
    random.shuffle(data)
    writer.writerows(data)
    print(len(data))

def topEmojis():
    emoji_dict = get_emojiDict()
    # Emoji	Unicode_codepoint	Occurrences	Position	Negative	Neutral	Positive	Unicode name	Unicode block
    top_occur = [':)',':(']+[emoji_dict[i][0] for i in range(55)]
    # top_occur = [':)', ':(', '😂', '❤', '♥', '😍', '😭', '😘', '😊', '👌', '💕', '👏', '😁', '☺', '♡', '👍', '😩', '🙏', '✌', '😏', '😉', '🙌', '🙈', '💪', '😄', '😒', '💃', '💖', '😃', '😔', '😱', '🎉', '😜', '☯', '🌸', '💜', '💙', '✨', '😳', '💗', '★', '█', '☀', '😡', '😎', '😢', '💋', '😋', '🙊', '😴', '🎶', '💞', '😌', '🔥', '💯', '🔫', '💛']
    return top_occur
def replace_emojis(str,top_occur):
    emoji_list = []
    for c in str:
        if c in top_occur:
            emoji_list.append(c)
            str = str.replace(c,'')
        elif c in emoji.UNICODE_EMOJI:
            str = str.replace(c,emoji.demojize(c)[1:-1].replace('_',' ')+' ')
    while ':)' in str:
        str = str.replace(':)','',1)
        emoji_list.append(':)')
    while ':(' in str:
        str = str.replace(':(','',1)
        emoji_list.append(':(')
    emoji_list = list(set(emoji_list))
    return str.strip()," ".join(emoji_list)
def emoji_to_id(emoji_list,top_occur):
    repeated = [top_occur.index(e)+1 for e in emoji_list]
    return list(set(repeated))

def test():
    top_occur = topEmojis()
    print(top_occur)
    print(len(top_occur))

    # removeUSER = 'ThatGoldenOreo 🍓🍀🛡🌷❤'
    tweet ='ying over 1d, 9 years 🏳️‍🌈 😂😂 if you kno :) :('
    # print(extract_emojis(removeUSER))
    # removeUSER = 'DatBajjuEdoBoi 🌚'
    #removeUSER = """!Democrats " throw Hail Lucifer's!🔥☻😈👿👹👺☠💀🔥"""
    str,emoji_list = replace_emojis(tweet,top_occur)
    # emoji_ids = emoji_to_id(emoji_list,top_occur)
    # print(str,emoji_list,emoji_ids)
    print(emoji_list)

    #test emoji writting
    # data = []
    # data.append([str,emoji_list])
    # writer = csv.writer(open('test.csv', 'w'))
    # writer.writerow(['twitter','emoji'])
    # writer.writerows(data)

    # RE_EMOJI = emoji_regex()
    # search_res = RE_EMOJI.finditer(removeUSER)
    # for i in search_res: #
    #     if i.group(0) in top_occur:
    #         pass
    #     else:
    #         # index= np.argwhere(emoji_dict == i.group(0))[0][0]
    #         # removeUSER = removeUSER.replace(i.group(0),emoji_dict[index][1])
    #         emoji_desc  = emoji.demojize(i.group(0))[1:-1].replace('_',' ')+' '
    #         removeUSER = removeUSER.replace(i.group(0),emoji_desc)

    # print(removeUSER)
if __name__ == "__main__":
    # test()
    feed_emoji()
