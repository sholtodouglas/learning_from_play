import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Wrapper, RNN, GRU
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
        # Add learnable weights for each state
        self.w, self.b = [], []
        for _ in self.cell.state_size:
            w = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='zeros',
                                    trainable=True,
                                    dtype=tf.float32)
            b = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True,
                                    dtype=tf.float32)
            self.w.append(w)
            self.b.append(b)

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

        def create_init_values(unnested_state_size, s_i=0):
            flat_dims = tensor_shape.TensorShape(unnested_state_size).as_list()
            init_state_size = [batch_size_tensor] + flat_dims
            if self.learned_init:
                return inputs[:,0,:] @ self.w[s_i] + self.b[s_i]
            else:
                return array_ops.zeros(init_state_size, dtype=dtype)

        if nest.is_nested(state_size):
            return nest.map_structure(create_init_values, state_size, list(range(len(state_size))) )
        else:
            return create_init_values(state_size)

class LearnedInitGRU(GRU):
    """
    Same as LSTM - only key difference is GRU only has one state to keep track of
    Todo: See if we can combine both implementations into one general RNN learned init wrapper class
    """
    def __init__(self, units, learned_init=True, **kwargs):
        super(LearnedInitGRU, self).__init__(units, **kwargs)
        self.learned_init = learned_init

    def build(self, input_shape):
        super(LearnedInitGRU, self).build(input_shape)
        # Add learnable weights for each state
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='zeros',
                                trainable=True,
                                dtype=tf.float32)
        self.b = self.add_weight(shape=(self.units,),
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

        def create_init_values(unnested_state_size):
            flat_dims = tensor_shape.TensorShape(unnested_state_size).as_list()
            init_state_size = [batch_size_tensor] + flat_dims
            if self.learned_init:
                return inputs[:,0,:] @ self.w + self.b
            else:
                return array_ops.zeros(init_state_size, dtype=dtype)

        if nest.is_nested(state_size):
            return nest.map_structure(create_init_values, state_size)
        else:
            return create_init_values(state_size)