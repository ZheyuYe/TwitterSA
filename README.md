# TwitterSA
Twitter Sentiment Analysis and its web application  
Website url: [Test before Tweet](http://testbeforetweet.uksouth.cloudapp.azure.com/)

Twitter Sentiment Analysis and its web application A Natural Language Processing poject and its web application aimed to analyse the sentiment index of social media (Twitter for now). A web Application based on the template of [Start Bootstrap - SB Admin 2](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2) and [Start Bootstrap - Creative](https://github.com/BlackrockDigital/startbootstrap-creative) tring to collecting sentiment information from visitor (gold standard). Some attrative functions are supplied involving tracking a twitter user of latest activities with sentiment analysis showing in a diagram, analysing a tweet into three types of emotion (posivite, neutral, negative).

Analysis model is based on the BERT trained model provided by [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) , and we use a PyTorch implementation of [focal loss function](https://github.com/clcarwin/focal_loss_pytorch) during the experiments.