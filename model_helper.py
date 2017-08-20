import tensorflow as tf

__all__ = ["create_rnn_cell"]

def create_rnn_cell(hparams, mode):
  cell_list = _cell_list(hparams['num_units'],
                         hparams['num_layers'],
                         0,
                         hparams['forget_bias'],
                         hparams['dropout'],
                         mode)
  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)

def _cell_list(num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode,
               single_cell_fn=None):
  """Create a list of RNN cells."""
  single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    single_cell = single_cell_fn(
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        device_str="/cpu:0",
    )
    cell_list.append(single_cell)

  return cell_list

def _single_cell(num_units, forget_bias, dropout,
                 mode, residual_connection=False, device_str=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
  single_cell = tf.contrib.rnn.BasicLSTMCell(
      num_units,
      forget_bias=forget_bias)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

  return single_cell