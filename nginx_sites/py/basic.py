import sys
import re
import os
import tweepy
import random
import pymysql
from datetime import datetime,timedelta,date
from vader import sentiment_analyzer_scores
from conf import init_tweepy
sys.path.append('../../model')
from deploy import get_scores

# %%
local_time_zone = -1;

def get_status(id):
    status = api.get_status(id)
    return status

def non_RT(status):
    # remove retweet and reply and tweets without content
    return (not status.retweeted) and ('RT @' not in status.full_text) and (not status.in_reply_to_user_id_str)  and ('https://t.co/' not in status.full_text[:14])

def addslash(txt):
    separate = txt.split("'")
    res = "\\\'".join(separate)
    return res

def date2str(datetime):
    return datetime.strftime("%Y/%m/%d %H:%M:%S")

def my_following(api,db):
    for status in api.home_timeline(result_type='mixed',count=100, tweet_mode='extended'):
        if non_RT(status):
            insert_tweet(db,status,'');
def crawl_tweets(api,db,localtime):
    cursor = db.cursor()
    cursor.connection.ping()
    sql = "SELECT id FROM `status` order by id desc limit 1";
    cursor.execute(sql)
    id = cursor.fetchone()[0]
    num = random_tweets(api,db,localtime);
    try:
        cursor = db.cursor()
        sql = "SELECT id FROM `status` where id>"+str(id)+" order by id asc limit 1";
        cursor.execute(sql)
        new_id = cursor.fetchone()[0]
        sql = "INSERT INTO `update`(datetime,initial_tweet,total_num) VALUES ('%s','%s','%s')" %(localtime,new_id,num);
        cursor.execute(sql)
        sql = "SELECT id FROM `update` where initial_tweet="+str(new_id);
        cursor.execute(sql)
        update_id = cursor.fetchone()[0]
        db.commit()
    except:
        print('fail crawl')
        db.rollback()
    cursor.close()

    res = {
        'update_id':update_id,
        'new_id':new_id,
        'tnum':num}

    return res

def random_tweets(api,storage,time):
    datestring = time.split('+')
    timezone = '+'+datestring[1][0:4]
    timestring = datestring[0][0:24]+' '+timezone
    today = datetime.strptime(timestring, '%a %b %d %Y %H:%M:%S %z')
    yesterday = today+timedelta(days=-1);
    # today_str = today.strftime('%Y-%m-%d')
    # yes_str = yesterday.strftime('%Y-%m-%d')
    #
    millis = yesterday.timestamp()*1000
    shuffled = []
    # split 24 hours into 12*2
    for i in range(8):
        # randomly add noise between -30mins and -60mins to right boundary
        left_millis = time2id(millis+i*3*3600000)
        if (i!=11):
            noise = -1800000*2+random.randint(-1800000,1800000)
            right_millis = time2id(millis+noise+(i+1)*3*3600000)
        else:
            right_millis = time2id(millis+(i+1)*3*3600000)
        # print(left_millis)
        # print(right_millis)
        search_results = tweepy.Cursor(api.search,
                               q = "``",
                               # since=yes_str,
                               # until=today_str,
                               since_id=left_millis,
                               max_id=right_millis,
                               lang="en",
                               result_type='mixed',
                               tweet_mode='extended').items(50)

        subshuffled = []
        for status in search_results:
            if non_RT(status):
                subshuffled.append(status);

        random.shuffle(subshuffled)
        # store first three elements
        shuffled.append(subshuffled[0])
        shuffled.append(subshuffled[1])
    print('total number: ',len(shuffled))
    random.shuffle(shuffled)
    insert_tweet(storage,shuffled[0],'status',True);
    for status in shuffled[1:]:
        insert_tweet(storage,status,'status');
    return len(shuffled)

def insert_tweet(db,status,table,init=False):
    cursor = db.cursor()
    if init:
        cursor.connection.ping()
    normalized_time =  status.created_at-timedelta(hours=local_time_zone)
    content = status.full_text.replace('\n','').replace('\r','') if table == 'dataset' else status.full_text
    sql = """INSERT INTO %s(tweet_id,screen_name,user_name,user_id,location, content,created_time)
             VALUES ('%s','%s','%s','%s','%s', '%s', '%s')""" %(table,status.id_str,status.user.screen_name,addslash(status.user.name),status.user.id_str,status.user.location,addslash(content), str(normalized_time))
    try:
        cursor.execute(sql)
        db.commit()
    except:
        #print('insert tweet fail')
        db.rollback()
    cursor.close()


def insert_track_tweet(db,status,init):
    cursor = db.cursor()
    if not init:
        cursor.connection.ping()
        # print('ping',init)

    normalized_time =  status.created_at-timedelta(hours=local_time_zone)
    sql = """INSERT INTO track_status(tweet_id,screen_name,user_id,location, content,created_time)
             VALUES ('%s','%s','%s','%s','%s', '%s')""" %(status.id_str,status.user.screen_name,status.user.id_str,status.user.location,addslash(status.full_text), str(normalized_time))
    try:
        cursor.execute(sql)
        db.commit()
    except:
        # print('insert tracked tweet fail')
        db.rollback()
    cursor.close()


def save_in_file(fo,status,i):
    fo.write("i=: "+str(i)+'\n')
    fo.write("tweet_id: "+status.id_str+'\n')
    fo.write("user_name: "+status.user.screen_name+'\n')
    fo.write("user_id: "+status.user.id_str+'\n')
    fo.write("content: "+status.full_text+'\n')
    fo.write("time: "+str(status.created_at)+'\n')
    # fo.write("name: "+status.user.time_zone+'\n')
    fo.write("location: "+status.user.location+'\n')
    fo.write("name: "+status.user.name+'\n')

def time2id(millis):
    intid = (int(millis)-1288834974663)<<22
    return str(intid)
def id2time(id):
    # int(
    bit = bin(id).replace('0b','');
    time_stamp = bit[:-22];
    res = int(time_stamp,2)+1288834974663;
    return res

def get_user(api,db,str_id):
    if ' ' not in str_id:
        user = api.get_user(str_id)
        if not user.id:
            user = search_user(str_id,api)
    else:
        user = search_user(str_id,api)
    if user.id:
        insert_user(db,user)
        array = {
            'user_id':user.id_str,
            'screen_name':user.screen_name,
            'user_name':addslash(user.name),
            'description':addslash(user.description),
            'location':user.location,
            'followers_count':user.followers_count,
            'friends_count':user.friends_count,
            'img_url':user.profile_image_url.replace('normal','400x400'),
            'state':'found'}
        return array
    else:
        return {'state':'None'}
def search_user(str_id,api):
    search_results = tweepy.Cursor(api.search_users,q = str_id).items(2)
    return next(search_results)

def insert_user(db,user):
    cursor = db.cursor()
    cursor.connection.ping()
    sql = """INSERT INTO user(user_id,screen_name,user_name,description,location,followers_count,friends_count,profile_image_url)
             VALUES ('%s','%s','%s','%s','%s', '%s', '%s','%s')""" %(user.id_str,user.screen_name,addslash(user.name),addslash(user.description),user.location,user.followers_count,user.friends_count,user.profile_image_url)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print('insert user fail')
        db.rollback()
    cursor.close()
        # db.close()
def track_activities(api,db,id_str):
    # print(id_str)
    # cursor = db.cursor()
    # cursor.connection.ping()
    # cursor.close()
    res = [[] for j in range(7)]
    today = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    aWeekBefore = (today-timedelta(days=6))
    def morethan100(page):
        i = 0
        for status in api.user_timeline(id=id_str,since_id=time2id(aWeekBefore.timestamp()*1000),count=100, tweet_mode='extended',page=page):
            i +=1
            if non_RT(status):
                insert_track_tweet(db,status,i-1);
                whichday = (status.created_at-aWeekBefore).days
                if whichday>=0:
                    res[whichday].append(get_scores(status.full_text)['compound'])
        print('Crazy Twitter',i)
        return i==100
    page = 1
    while morethan100(page):
        page+=1
    mean = [round(float(sum(l))/len(l),4) if len(l)> 0 else 'nan' for l in res]
    amount = [len(l) for l in res]
    return [mean,amount]
def get_details(db,index,user_id):

    today = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    left = (today-timedelta(days=index))
    right = (today-timedelta(days=index-1))
    left_id = time2id(left.timestamp()*1000)
    right_id = time2id(right.timestamp()*1000)

    cursor = db.cursor(cursor=pymysql.cursors.DictCursor)
    cursor.connection.ping()
    sql = "SELECT created_time,content FROM `track_status` where `user_id` = %s and `tweet_id`>=%s and `tweet_id`< %s" %(user_id,left_id,right_id);
    cursor.execute(sql)
    # print(sql)
    db_res = cursor.fetchall()
    res = []
    print(len(db_res))
    for i in range(len(db_res)):
        temp = []
        temp.append(db_res[i]['created_time'].split(' ')[1])
        temp.append(re.sub(r'(https:\/\/t\.co\/[a-zA-Z\d]*)', '', db_res[i]['content']))
        temp.append(get_scores(db_res[i]['content'])['compound'])
        res.append(temp)
    res.sort()
    cursor.close()
    # print(res[0])
    return res

def ping(self, reconnect=True):
    """Check if the server is alive"""
    if self._sock is None:
        if reconnect:
            self.connect()
            reconnect = False
        else:
            raise err.Error("Already closed")
    try:
        self._execute_command(COMMAND.COM_PING, "")
        self._read_ok_packet()
    except Exception:
        if reconnect:
            self.connect()
            self.ping(False)
        else:
            raise

def crawl_dataset(api,storage):
    now = datetime.now()
    hourBefore = datetime.now()+timedelta(seconds=-90);
    shuffled = []
    left_millis = time2id(hourBefore.timestamp()*1000)
    right_millis = time2id(now.timestamp()*1000)
    search_results = tweepy.Cursor(api.search,
                           q = "``",
                           since_id = left_millis,
                           max_id = right_millis,
                           lang="en",
                           result_type='mixed',
                           tweet_mode='extended').items(100)
    for status in search_results:
        if non_RT(status):
            shuffled.append(status);

    print('dataset increase number: ',len(shuffled))
    insert_tweet(storage,shuffled[0],'dataset',True);
    for status in shuffled[1:]:
        insert_tweet(storage,status,'dataset');
    return len(shuffled)
# %%
