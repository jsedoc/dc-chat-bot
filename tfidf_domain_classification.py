from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import cPickle
import sys
def readTrain():
 f = open('data/train_set', 'r')
 train_X = []
 train_Y = []
 for line in f:
  train_X.append(line)
 f = open('data/train_set_target', 'r')
 for line in f:
  train_Y.append(line)
 return (train_X,train_Y)

model = None
vectorizer = None

##TRAIN
#This is one time, only during training
#train_X, train_Y = readTrain()
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
#X_train = vectorizer.fit_transform(train_X)
#model = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")
#model.fit(X_train,train_Y)
#Model pickled after train

#Testing
def predict(sent):
 global model
 global vectorizer
 if(model == None):
  with open('dc_tfidf_model.pkl','rb') as fid:
   model = cPickle.load(fid)
 if(vectorizer == None):
  with open('dc_vectorizer.pkl','rb') as fid:
   vectorizer = cPickle.load(fid)
 sent = [sent]
 X_test = vectorizer.transform(sent)
 return model.predict(X_test)[0]


if __name__ == "__main__":
 sent = sys.argv[1]
 print(predict(sent)) 





