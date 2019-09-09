# PRETRAINED_WEIGHTS = "bert-base-cased"
PRETRAINED_WEIGHTS = "./bert-cased/"
MAX_SEQ_LENGTH = 128
MAX_EMOJI_LENGTH = 16

# max length 244, average length 60

TASK_LABELS = {
    'sentiment140': {
        "POSIVITE": 1,
        # "NEUTRAL": 1,
        "NEGATIVE": 0,
    },
    'SemEval2017': {
        "POSIVITE": 2,
        "NEUTRAL": 1,
        "NEGATIVE": 0,
    },
    'emoji': {
        "POSIVITE": 2,
        "NEUTRAL": 1,
        "NEGATIVE": 0,
    },
    'feed': {
        "POSIVITE": 2,
        "NEUTRAL": 1,
        "NEGATIVE": 0,
    }
}

DATAFILE = {
    'emoji' :"../data/3-emoji.csv",
    'emoji2' : "../data/2-emoji.csv",
    'sentiment140':'../data/sentiment_train_set.csv',
    'SemEval2017':'../data/SemEval2017-task4-preprocessed/',
    'feed':'../data/feed_emoji.csv'
}


def get_tokenizer(do_lower_case):
    if not do_lower_case:
        return './tokenizer-base-cased/'
    else:
        return './tokenizer-base-uncased/'
