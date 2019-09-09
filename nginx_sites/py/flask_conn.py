# @Author: Zheyu Ye
# @Date:   Mon 08 Jul 2019 16:23:20
# @Last modified by:   Zheyu Ye
# @Last modified time: Fri 12 Jul 2019 01:50:37

from flask import Flask,render_template, request,json, jsonify
from flask_cors import CORS
from basic import crawl_tweets,get_user,track_activities,get_details
from conf import init_tweepy,get_dbconfig,get_host
from vader import sentiment_analyzer_scores
import pymysql
import pymysqlpool
import time
import sys
sys.path.append('../../model')
from deploy import get_scores

pymysqlpool.logger.setLevel('DEBUG')
config = get_dbconfig()
hosturl = get_host()
pool = pymysqlpool.ConnectionPool(size=30, name='pool', **config)
api = init_tweepy()
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello local Mac!"

@app.route('/crawl', methods=['POST'])
def crawl():
    db = pool.get_connection()
    print('poll size:',pool.size())
    localtime = request.form['localtime']
    res = crawl_tweets(api,db,localtime)
    db.close()
    return jsonify(res)

@app.route('/analysis', methods=['POST'])
def analysis():
    text = request.form['text']
    print(text)
    print('poll size:',pool.size())
    # score = sentiment_analyzer_scores(text)
    score = get_scores(text)
    return jsonify(score)

@app.route('/checkUser', methods=['POST'])
def check_user():
    track_name = request.form['track_name']
    db = pool.get_connection()
    print('poll size:',pool.size())
    user = get_user(api,db,track_name)
    db.close()
    return jsonify(user)

@app.route('/track', methods=['POST'])
def check_activities():
    id_str = request.form['id']
    db = pool.get_connection()
    [mean,amount] = track_activities(api,db,id_str)
    db.close()
    return jsonify({'sentiment':mean,'amount':amount})

@app.route('/details', methods=['POST'])
def details():
    db = pool.get_connection()
    print('poll size:',pool.size())
    index = int(request.form['index'])
    user_id = request.form['user_id']
    res = get_details(db,index,user_id)
    db.close()
    return jsonify(res)

if __name__ == "__main__":
    app.run(host= hosturl, port = 5000, debug = True)
