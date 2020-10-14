import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Wrapper, RNN
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape

class LearnedInitLSTM(LSTM):
    """
    Extension of LSTM layer that learns mappings from initial inputs to initial states (c,h), where c is the memory
    state and h is the carry state.
    This aims to overcome the poor performance of RNNs within the first few timesteps due to naive zero init.
    """
    def __init__(self, units, learned_init=True, **kwargs):
        super(LearnedInitLSTM, self).__init__(units, **kwargs)
        self.learned_init = learned_init

    def build(self, input_shape):
        super(LearnedInitLSTM, self).build(input_shape)
        # Cell state
        self.wc = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='zeros',
                                  trainable=True,
                                  dtype=tf.float32)
        self.bc = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  dtype=tf.float32)
        # Hidden state
        self.wh = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='zeros',
                                  trainable=True,
                                  dtype=tf.float32)
        self.bh = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  dtype=tf.float32)

    def get_initial_state(self, inputs):

        if nest.is_sequence(inputs):
            # The input are nested sequences. Use the first element in the seq to get
            # batch size and dtype.
            inputs = nest.flatten(inputs)[0]

        input_shape = array_ops.shape(inputs)
        batch_size = input_shape[1] if self.time_major else input_shape[0]
        dtype = inputs.dtype
        init_state = self._generate_initial_state(inputs, batch_size,
                                                  self.cell.state_size, dtype)
        # Keras RNN expect the states in a list, even if it's a single state tensor.
        if not nest.is_sequence(init_state):
            init_state = [init_state]
        # Force the state to be a list in case it is a namedtuple eg LSTMStateTuple.
        return list(init_state)

    def _generate_initial_state(self, inputs, batch_size_tensor, state_size, dtype):
        """Generate a zero filled tensor with shape [batch_size, state_size]."""
        if batch_size_tensor is None or dtype is None:
            raise ValueError(
                'batch_size and dtype cannot be None while constructing initial state: '
                'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

        def create_zeros(unnested_state_size, state_index):
            flat_dims = tensor_shape.TensorShape(unnested_state_size).as_list()
            init_state_size = [batch_size_tensor] + flat_dims
            if self.learned_init:
                if state_index == 0:
                    return inputs[:,0,:] @ self.wc + self.bc
                else:
                    return inputs[:,0,:] @ self.wh + self.bh
            else:
                return array_ops.zeros(init_state_size, dtype=dtype)

        if nest.is_nested(state_size):
            return nest.map_structure(create_zeros, state_size, list(range(len(state_size))) )
        else:
            return create_zeros(inputs, state_size, 0)