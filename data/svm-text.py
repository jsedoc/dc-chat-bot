import codecs
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn import svm
from sklearn.linear_model import LogisticRegression
from scipy.ndimage.interpolation import shift
import cPickle
import sys

lin_clf = None
vectorizer = None

NUM_PREV_DOMAINS = 3
NUM_DOMAINS = 3

DOMAIN_MAP = {'Movie':0,'Gaming':1,'ood':2}

TEST_SET_PERCENT = 0.2

# Reading dialog file
def read_dialog_file(dialog_filename):
        with codecs.open(dialog_filename,'r',encoding='utf-8') as dialog_file:
                dialogs = dialog_file.readlines()
                return dialogs

# Process dialog file to have a list of conversations with each conversation being a list of tuples in the form on (utterance,domain)
def process_dialog_file(dialog_filename):
        dialogs = read_dialog_file(dialog_filename)
        conversations = []
        conversation = []
        for dialog in dialogs:
                if not dialog.strip():
                        conversations.append(conversation)
                        conversation = []
                else:
                        speaker = dialog.split(':')[0].strip()
                        if ( speaker=='B' ):
                                continue
                        diag = dialog[4:].strip()
                        if len(diag.rsplit('\t')) < 2:
                                continue
                        utterance = diag.rsplit('\t')[0]
                        domain = diag.rsplit('\t')[1]
                        if domain not in DOMAIN_MAP:
                                domain = 'ood'
                        conversation.append((utterance,DOMAIN_MAP[domain]))
        return conversations

# Prediction from svm
def predict(sent):
        global lin_clf
        global vectorizer
        if(lin_clf == None):
                with open('domain_model.pkl','rb') as fid:
                        lin_clf = cPickle.load(fid)
        if(vectorizer == None):
                with open('domain_vectorizer.pkl','rb') as fid:
                        vectorizer = cPickle.load(fid)
        sent = [sent,sent]
        X_test = vectorizer.transform(sent)
        return lin_clf.predict(X_test)[0]



conversations = process_dialog_file('test_conv_responses')
test_set_size = int(len(conversations)*TEST_SET_PERCENT)
train_set_size = len(conversations) - test_set_size
train_conversations = conversations[:train_set_size]
test_conversations = conversations[train_set_size:len(conversations)]

with codecs.open('svm_predictions_test','w',encoding='utf-8') as f:
        for conversation in conversations:
                for dialog in conversation:
                        f.write(predict(dialog[0]) + '\n')
                f.write(str(3)+"\n")

# with open('actual_x','w') as f:
#         for conversation in conversations:
#                 for dialog in conversation:
#                         f.write(str(dialog[1]) + '\n')
#                 f.write(str(3)+"\n")        


