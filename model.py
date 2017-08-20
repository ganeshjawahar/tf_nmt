import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import abc
import sys
sys.path.append(".")
import model_helper

__all__ = ["BaseModel", "Model"]

class BaseModel(object):
  """
  Sequence-to-sequence base class.
  """
  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table):
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table
    self.src_vocab_size = hparams['src_vocab_size']
    self.tgt_vocab_size = hparams['tgt_vocab_size']
    self.num_layers = hparams['num_layers']
    self.src_embed_size = hparams['src_embed_size']
    self.tgt_embed_size = hparams['tgt_embed_size']
    self.tgt_max_len_infer = 50

    # Initializer
    initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=123)
    tf.get_variable_scope().set_initializer(initializer)

    # Init embeddings
    self.init_embeddings(hparams)
    self.batch_size = tf.size(self.iterator.source_sequence_length)
    self.time_major = True

    # Projection
    with tf.variable_scope("build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = layers_core.Dense(
            hparams['tgt_vocab_size'], use_bias=False, name="output_projection")

    ## Train graph
    res = self.build_graph(hparams)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.cond(
            self.global_step < hparams['start_decay_step'],
            lambda: tf.constant(hparams['learning_rate']),
            lambda: tf.train.exponential_decay(
                hparams['learning_rate'],
                (self.global_step - hparams['start_decay_step']),
                hparams['decay_steps'],
                hparams['decay_factor'],
                staircase=True),
          name="learning_rate")
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      tf.summary.scalar("lr", self.learning_rate)
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=True)

      clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
      gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
      gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", hparams['learning_rate']),
          tf.summary.scalar("train_loss", self.train_loss),
      ] + gradient_norm_summary)

      # Saver
      self.saver = tf.train.Saver(tf.global_variables())

  def init_embeddings(self, hparams):
    """Init embeddings."""
    partitioner = None
    with tf.variable_scope("embeddings", dtype=tf.float32, partitioner=partitioner) as scope:
      with tf.variable_scope("encoder", partitioner=partitioner):
        self.embedding_encoder = tf.get_variable(
            "embedding_encoder", [hparams['src_vocab_size'], hparams['src_embed_size']], tf.float32)
      with tf.variable_scope("decoder", partitioner=partitioner):
        self.embedding_decoder = tf.get_variable(
            "embedding_decoder", [hparams['tgt_vocab_size'], hparams['tgt_embed_size']], tf.float32)

  def train(self, sess):
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size])

  def build_graph(self, hparams):
    dtype = tf.float32
    num_layers = hparams['num_layers']
    with tf.variable_scope("dynamic_seq2seq", dtype=dtype):
      # Encoder
      encoder_outputs, encoder_state = self._build_encoder(hparams)

      ## Decoder
      logits, sample_id, final_context_state = self._build_decoder(
          encoder_outputs, encoder_state, hparams)

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device("/cpu:0"):
          loss = self._compute_loss(logits)
      else:
        loss = None

      return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.
    Build and run an RNN encoder.
    Args:
      hparams: Hyperparameters configurations.
    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers):
    """Build a multi-layer RNN cell that can be used by encoder."""
    return model_helper.create_rnn_cell(hparams, self.mode)

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer."""
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams['sos'])),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams['eos'])),
                         tf.int32)
    num_layers = hparams['num_layers']
    iterator = self.iterator
    maximum_iterations = self.tgt_max_len_infer
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)
      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input
        target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        with tf.device("/cpu:0"):
          logits = self.output_layer(outputs.rnn_output)
    return logits, sample_id, final_context_state


  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.
    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.
    Returns:
      A tuple of a multi-layer RNN cell used by decoder
        and the intial state of the decoder RNN.
    """
    pass
  
  def get_max_time(self, tensor):
    time_axis = 0
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  def _compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.iterator.target_output
    target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
    target_weights = tf.transpose(target_weights)
    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)
    return loss

    
class Model(BaseModel):
  """
  Sequence-to-sequence dynamic model.
  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    num_residual_layers = 0
    iterator = self.iterator

    source = iterator.source
    source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
          self.embedding_encoder, source)

      cell = model_helper.create_rnn_cell(hparams, self.mode)

      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major)

    return encoder_outputs, encoder_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    cell = model_helper.create_rnn_cell(hparams, self.mode)
    decoder_initial_state = encoder_state
    return cell, decoder_initial_state

