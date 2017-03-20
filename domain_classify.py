from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import cPickle

model = None
vectorizer = None

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("cont_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("resp_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "data", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(session, forward_only, path, vocab_size,domain):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      vocab_size,
      vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype) 
  all_vars = tf.all_variables()
  model_vars = [k for k in all_vars if k.name.startswith(domain)]
  ckpt = tf.train.get_checkpoint_state(path) 
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path) 
    tf.train.Saver(model_vars).restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def load_model(sess,checkpoint,vocab_size,domain):
    # Create model and load parameters. 
    model = create_model(sess, True,checkpoint,vocab_size,domain)
    model.batch_size = 1  # We decode one sentence at a time. 

    return model

def decode(sess,model,cont_vocab,resp_vocab,sentence):
    # Decode from standard input.
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), cont_vocab)
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence) 

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      response = " ".join([tf.compat.as_str(resp_vocab[output]) for output in outputs])
      return response 

def predict(sent):
 global model
 global vectorizer
 if(model == None):
  with open('domain_model.pkl','rb') as fid:
   model = cPickle.load(fid)
 if(vectorizer == None):
  with open('domain_vectorizer.pkl','rb') as fid:
   vectorizer = cPickle.load(fid)
 sent = [sent]
 X_test = vectorizer.transform(sent)
 return model.predict(X_test)[0]

def main(_):
  with tf.Session() as sess:
  	with tf.variable_scope("movies") as movies_scope:
		movies_model = load_model(sess,'movies_checkpoint',158470,"movies")	
	with tf.variable_scope("gaming") as gaming_scope:
		gaming_model = load_model(sess,'gaming_checkpoint',91935,"gaming")
	with tf.variable_scope("ood") as ood_scope:
		ood_model = load_model(sess,'ood_checkpoint',40000,"ood")
	
	# Load vocabularies.
	movie_cont_vocab_path = os.path.join(FLAGS.data_dir,
				 "movies/vocab158470.cont")
	movie_resp_vocab_path = os.path.join(FLAGS.data_dir,
				 "movies/vocab158470.resp")

	movie_cont_vocab, _ = data_utils.initialize_vocabulary(movie_cont_vocab_path)
	_, movie_resp_vocab = data_utils.initialize_vocabulary(movie_resp_vocab_path)

	gaming_cont_vocab_path = os.path.join(FLAGS.data_dir,
				 "gaming/vocab91935.cont")
	gaming_resp_vocab_path = os.path.join(FLAGS.data_dir,
				 "gaming/vocab91935.resp")

	gaming_cont_vocab, _ = data_utils.initialize_vocabulary(gaming_cont_vocab_path)
	_, gaming_resp_vocab = data_utils.initialize_vocabulary(gaming_resp_vocab_path)

	ood_cont_vocab_path = os.path.join(FLAGS.data_dir,
                                 "ood/vocab40000.cont")
        ood_resp_vocab_path = os.path.join(FLAGS.data_dir,
                                 "ood/vocab40000.resp")

        ood_cont_vocab, _ = data_utils.initialize_vocabulary(ood_cont_vocab_path)
        _, ood_resp_vocab = data_utils.initialize_vocabulary(ood_resp_vocab_path)

	sys.stdout.write("> ")
    	sys.stdout.flush()
    	sentence = sys.stdin.readline()
    	while sentence:
		pred_str = predict(sentence)
		pred = int(pred_str.strip())
		response = ''
        	if pred==0:
                        print('Movie Predicted!')
			with tf.variable_scope("movies"):
                		response = decode(sess,movies_model,movie_cont_vocab,movie_resp_vocab,sentence)
        	elif pred==1:
                        print('Gaming Predicted!')
			with tf.variable_scope("gaming"):
                		response = decode(sess,gaming_model,gaming_cont_vocab,gaming_resp_vocab,sentence)
		elif pred==2:
			print('Out of domain Predicted!')
			with tf.variable_scope("ood"):
				response = decode(sess,ood_model,ood_cont_vocab,ood_resp_vocab,sentence)
		print(response)
        	sys.stdout.write("> ")
		sys.stdout.flush()
      		sentence = sys.stdin.readline()

if __name__ == "__main__":
  tf.app.run()
