from basic import crawl_dataset
from conf import init_tweepy,get_dbconfig
from vader import sentiment_analyzer_scores
import pymysqlpool
import time

pymysqlpool.logger.setLevel('DEBUG')
config = get_dbconfig()
pool = pymysqlpool.ConnectionPool(size=30, name='pool', **config)
api = init_tweepy()

def dataset(n):
    while True:
        db = pool.get_connection()
        crawl_dataset(api,db)
        db.close()
        time.sleep(n)
if __name__ == "__main__":
    dataset(90)
