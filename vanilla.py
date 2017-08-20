# Sample code for uni-directional LSTM based Seq2Seq without attention
# optimizer - sgd
# init_op - uniform
# init_weight - 0.1
# GPU - False
# metrics - bleu
# source-reverse - False
# time-major - True

import sys
import random
import collections
import numpy as np
import tensorflow as tf
sys.path.append(".")
import model as nmt_model
from tensorflow.python.ops import lookup_ops

# hyper-parameters
num_layers = 2
num_units = 15
dropout = 0.2
learning_rate = 1.0
start_decay_step = 0
decay_steps = 1000
decay_factor = 0.98
num_train_steps = 100
sos = '<s>'
eos = '</s>'
UNK_ID = 0
src_max_len = 50
tgt_max_len = 50
src_embed_size = 5
tgt_embed_size = 5
num_threads = 4
batch_size = 10
random_seed = 123

# data
train_src_file = 'data/train.en'
train_targ_file = 'data/train.vi'
test_src_file = 'data/tst2012.en'
test_targ_file = 'data/tst2012.vi'
src_vocab_file = 'data/vocab.en'
tgt_vocab_file = 'data/vocab.vi'

# set seed
random.seed(random_seed)
np.random.seed(random_seed)

# iterator over dataset
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass
def get_iterator(src_dataset, tgt_dataset, src_vocab_table, tgt_vocab_table):
  output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(
      src_vocab_table.lookup(tf.constant(eos)),
      tf.int32)
  tgt_sos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(sos)),
      tf.int32)
  tgt_eos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(eos)),
      tf.int32)
  
  src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

  # shuffle the data
  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed)

  # split the line to get values
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_threads=num_threads,
      output_buffer_size=output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # employ src_max_len
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

  # employ targ_max_len
  src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

  # Convert the word strings to ids.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_threads=num_threads, output_buffer_size=output_buffer_size)

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_threads=num_threads, output_buffer_size=output_buffer_size)

  # Add in the word counts.  Subtract one from the target to avoid counting
  # the target_input <eos> tag (resp. target_output <sos> tag).
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_threads=num_threads,
      output_buffer_size=output_buffer_size)

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(tf.TensorShape([None]),  # src
                       tf.TensorShape([None]),  # tgt_input
                       tf.TensorShape([None]),  # tgt_output
                       tf.TensorShape([]),      # src_len
                       tf.TensorShape([])),     # tgt_len
        # Pad the source and target sequences with eos tokens.
        padding_values=(src_eos_id,  # src
                        tgt_eos_id,  # tgt_input
                        tgt_eos_id,  # tgt_output
                        0,           # src_len -- unused
                        0))          # tgt_len -- unused
  batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
      batched_iter.get_next())

  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)

class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass

def train():
  """Train a translation model."""
  model_creator = nmt_model.Model
  graph = tf.Graph()
  with graph.as_default():
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    src_vocab_table = lookup_ops.index_table_from_file(
      src_vocab_file, default_value=UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_file(
      tgt_vocab_file, default_value=UNK_ID)

    src_dataset = tf.contrib.data.TextLineDataset(train_src_file)
    tgt_dataset = tf.contrib.data.TextLineDataset(train_targ_file)
    iterator = get_iterator(src_dataset, tgt_dataset, src_vocab_table, tgt_vocab_table)

    with tf.device(None):
      hparams = {}
      hparams['src_vocab_size'] = 17191 + 3
      hparams['tgt_vocab_size'] = 7709 + 3
      hparams['num_layers'] = num_layers
      hparams['src_embed_size'] = src_embed_size
      hparams['tgt_embed_size'] = tgt_embed_size
      hparams['learning_rate'] = learning_rate
      hparams['start_decay_step'] = 300
      hparams['decay_steps'] = 1000
      hparams['decay_factor'] = 0.5
      hparams['sos'] = 'sos'
      hparams['eos'] = 'eos'
      hparams['forget_bias'] = 1.0
      hparams['dropout'] = 0.2
      hparams['num_units'] = num_units
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table)
  train_model = TrainModel( 
    graph=graph,
    model=model,
    iterator=iterator)

  avg_step_time = 0.0

  # TensorFlow model
  config_proto = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True)
  config_proto.gpu_options.allow_growth = True

  train_sess = tf.Session(
    target="", config=config_proto, graph=train_model.graph)
  with train_model.graph.as_default():
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    loaded_train_model = train_model.model
    global_step = train_model.model.global_step.eval(session=train_sess)

  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step
  
  step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
  checkpoint_total_count = 0.0
  speed, train_ppl = 0.0, 0.0
  epoch_step = 0
  batch_size = 10

  skip_count = batch_size * epoch_step
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={})

  while global_step < num_train_steps:
    ### Run a step ###
    try:
      step_result = loaded_train_model.train(train_sess)
      (_, step_loss, step_predict_count, step_summary, global_step,
       step_word_count, batch_size) = step_result
      epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={})
      continue

if __name__ == "__main__":
  train()
