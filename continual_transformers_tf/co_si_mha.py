import tensorflow.compat.v2 as tf
from keras.layers import MultiHeadAttention

from .common import WithCallMode, enqueue_and_peek_all


class CoSiMultiHeadAttention(WithCallMode, MultiHeadAttention):
    """Continual Single-output MultiHeadAttention layer from http://arxiv.org/abs/2201.06268
    The computes identical results to the temporally newest value of the regular MultiHeadAttention.
    The layer expects inputs of shape:
    - call_regular: (B, T, ...)
    - call_steps: (B, T, ...)
    - call_step: (B, ...)

    Args:
      capacity: Sequence length
      num_heads: Number of attention heads.
      key_dim: Size of each attention head for query and key.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      output_shape: The expected shape of an output tensor, besides the batch and
        sequence dims. If not specified, projects back to the key feature dim.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.

    Call arguments:
      query: Query `Tensor` of shape `(B, T, dim)`.
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be `(attention_output, attention_scores)` if `True`, or `attention_output`
        if `False`. Defaults to `False`.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model,
        or False (inference) if there is no parent layer.

    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are project to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coefficients over
        attention axes.
    """

    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        key_dim: int,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        call_mode="regular",
        **kwargs,
    ):
        MultiHeadAttention.__init__(
            self,
            num_heads,
            key_dim,
            value_dim,
            dropout,
            use_bias,
            output_shape,
            attention_axes,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )
        self._seq_len = seq_len
        self.call_mode = call_mode

    def _build_from_signature(self, query, value, key=None, dtype=tf.float32):
        """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to True.

        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
          dtype: DType.
        """
        MultiHeadAttention._build_from_signature(self, query, value, key)

        capacity = self._seq_len or value.shape[1]
        self._query_step_shape = tf.TensorShape(
            [s for i, s in enumerate(self._query_shape) if i != 1]
        )
        self._key_step_shape = tf.TensorShape(
            (*[s for i, s in enumerate(self._key_shape) if i != 1], self._key_dim)
        )
        self._value_step_shape = tf.TensorShape(
            (*[s for i, s in enumerate(self._value_shape) if i != 1], self._value_dim)
        )
        self._key_queue = tf.queue.FIFOQueue(
            capacity=capacity,
            dtypes=[dtype],
            shapes=[self._key_step_shape],
            name="key_queue",
        )
        self._value_queue = tf.queue.FIFOQueue(
            capacity=capacity,
            dtypes=[dtype],
            shapes=[self._value_step_shape],
            name="value_queue",
        )
        self._init_key_value_queues(dtype)

    def _init_key_value_queues(self, dtype=tf.float32):
        self._key_queue.enqueue_many(
            tf.zeros((self._seq_len, *self._key_step_shape), dtype)
        )
        self._value_queue.enqueue_many(
            tf.zeros((self._seq_len, *self._value_step_shape), dtype)
        )
        self._init_steps_left = self._seq_len - 1

    def clean_state(self):
        self._key_queue.dequeue_many(self._seq_len)
        self._value_queue.dequeue_many(self._seq_len)
        self._init_key_value_queues(dtype=tf.float32)

    def call_step(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        **kwargs,
    ):
        if not self._built_from_signature:

            def add_capacity(shape, capacity):
                return tf.TensorShape((shape[0], capacity, *shape[1:]))

            self._build_from_signature(
                query=add_capacity(query.shape, self._seq_len),
                value=add_capacity(value.shape, self._seq_len),
                key=add_capacity(key.shape, self._seq_len) if key is not None else key,
                dtype=query.dtype,
            )
        if key is None:
            key = value
        # Project query, value, and keys steps
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `value` = [B, T, N ,H]
        query = self._query_dense(tf.expand_dims(query, 1))

        key = self._key_dense(tf.expand_dims(key, 1))
        key = tf.squeeze(key, 1)
        # `key` = [B, S, N, H]
        keys = enqueue_and_peek_all(self._key_queue, key, self._seq_len)

        value = self._value_dense(tf.expand_dims(value, 1))
        value = tf.squeeze(value, 1)  # [B, N, H]
        # `value` = [B, S, N, H]
        values = enqueue_and_peek_all(self._value_queue, value, self._seq_len)

        if self._init_steps_left > 0:
            self._init_steps_left -= 1
            if return_attention_scores:
                return None, None
            return None

        # Compute attention
        attention_output, attention_scores = self._compute_attention(
            query, keys, values, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)
        attention_output = tf.squeeze(attention_output, 1)
        attention_scores = tf.squeeze(attention_scores, 2)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def call_steps(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        **kwargs,
    ):
        assert query.shape[1] == value.shape[1] and (
            key is None or query.shape[1] == key.shape[1]
        )
        outs, scores = [], []
        for i in range(query.shape[1]):
            o = self.call_step(
                query[:, i],
                value[:, i],
                key[:, i] if key is not None else None,
                attention_mask,
                return_attention_scores,
                training,
            )
            if return_attention_scores:
                o, s = o
                if s is not None:
                    scores.append(s)
            if o is not None:
                outs.append(o)

        if len(outs) == 0:
            return None

        if return_attention_scores:
            return tf.stack(outs, axis=1), tf.stack(scores, axis=2)
        return tf.stack(outs, axis=1)

    def call_regular(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        **kwargs,
    ):
        query = tf.expand_dims(query[:, -1], 1)
        return MultiHeadAttention.call(
            self,
            query,
            value,
            key,
            attention_mask,
            return_attention_scores,
            training,
        )
