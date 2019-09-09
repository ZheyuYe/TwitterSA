
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

# sentiment_analyzer_scores(input);
# sentiment_analyzer_scores("The phone is super cool.")
# sentiment_analyzer_scores("The food here is great!")
# sentiment_analyzer_scores("The food here is great!!")
# sentiment_analyzer_scores("The food here is is GREAT!")
# sentiment_analyzer_scores("The food here is grat, but the service is horrible")
# # %%Emojis
# sentiment_analyzer_scores('😄')
# test =  sentiment_analyzer_scores(':)')
# print(test['pos'])


# sentiment_analyzer_scores('I’m so fucking hungry I can’t wait')
# sentiment_analyzer_scores('😥')
# sentiment_analyzer_scores('☹️')
# sentiment_analyzer_scores(':(')
#
# sentiment_analyzer_scores("it is so bad :(")

# #%% Slang
# sentiment_analyzer_scores("Today SUX!")
# sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol")
# #%% Emoticons
# sentiment_analyzer_scores("Make sure you :) or :D today!")

# print(sentiment_analyzer_scores('Incredible! beautiful techie'))
#
