
There were multiple stages in which we collected the data:

1) Domain data for individual domain seq2seq models
We got this data from bigquery reddit tables that were there in the google cloud platform. We got this for movies and gaming. I have attached the queries that we used to extract this data. There are two main tables: one for posts and the other for comments. comments table consists of replies to posts and other comments. We had to join these to get the question-answer pair for the models. We did some preprocessing: e.g. replacing words that appear at a frequency lesser than some number of times but we have lost access to the script and can't seem to find the data as well. We used twitter data to train our Out of domain seq2seq model.

2) Conversation data
We extracted this data using reddit API and then used SVM to label each utterance in this data with a domain. I have attached the scripts for both attraction and labelling here. We also have a original and labelled conversation file and I have attached that too here.
