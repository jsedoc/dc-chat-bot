# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

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

def create_model(session, forward_only, path, vocab_size):
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
  ckpt = tf.train.get_checkpoint_state(path)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def load_model(sess):
    # Create model and load parameters.
    model_movies = create_model(sess, True, 'movies_checkpoint',158470)
    model_movies.batch_size = 1  # We decode one sentence at a time.

    #model_gaming = create_model(sess,True, 'gaming_checkpoint',91935)
    #model_gaming.batch_size = 1

    return model_movies

def decode(session,model,cont_vocab,resp_vocab,sentence):
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

def main(_):
  with tf.Session() as sess:
  	movies_model = load_model(sess)
	#gaming_model = load_model(sess)
	# Load vocabularies.
	movie_cont_vocab_path = os.path.join(FLAGS.data_dir,
				 "movies/vocab158470.cont")
	movie_resp_vocab_path = os.path.join(FLAGS.data_dir,
				 "movies/vocab158470.resp")

	movie_cont_vocab, _ = data_utils.initialize_vocabulary(movie_cont_vocab_path)
	_, movie_resp_vocab = data_utils.initialize_vocabulary(movie_resp_vocab_path)

	#gaming_cont_vocab_path = os.path.join(FLAGS.data_dir,
	#			 "gaming/vocab91935.cont")
	#gaming_resp_vocab_path = os.path.join(FLAGS.data_dir,
	#			 "gaming/vocab91935.resp")

	#gaming_cont_vocab, _ = data_utils.initialize_vocabulary(gaming_cont_vocab_path)
	#_, gaming_resp_vocab = data_utils.initialize_vocabulary(gaming_resp_vocab_path)

	sys.stdout.write("> ")
    	sys.stdout.flush()
    	sentence = sys.stdin.readline()
    	while sentence:
		pred = 0
		response = ''
        	if pred:
                	response = decode(sess,movies_model,movie_cont_vocab,movie_resp_vocab,sentence)
        	#else:
                	#response = decode(sess,gaming_model,gaming_cont_vocab,gaming_resp_vocab,sentence)
		print(response)
        	sys.stdout.write("> ")
		sys.stdout.flush()
      		sentence = sys.stdin.readline()

if __name__ == "__main__":
  tf.app.run()
