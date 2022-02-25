import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

from .common import CallMode


class CircularPositionalEncoding(Layer):
    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        call_update_index_steps=1,  # Number of steps that a call updates index
        temporal_dim=-1,
        call_mode="regular",
    ):
        super(CircularPositionalEncoding, self).__init__()
        self._max_len = max_len
        self._pos_emb = Embedding(input_dim=max_len, output_dim=embed_dim)
        self._index = tf.Variable(0, trainable=False, name="pos_index")
        self._call_update_index_steps = tf.constant(call_update_index_steps)
        self._temporal_dim = temporal_dim
        self._call_mode = CallMode(call_mode)

    @property
    def call_mode(self) -> CallMode:
        return self._call_mode

    @call_mode.setter
    def call_mode(self, value):
        self._call_mode = CallMode(value)

    def call_regular(self, x: tf.Tensor, update_index_steps: int = None) -> tf.Tensor:
        rank = tf.rank(x)
        t_dim = (self._temporal_dim + rank) % rank
        L = tf.shape(x)[t_dim]
        pos_ids = tf.math.floormod(
            tf.range(start=0, limit=L, delta=1) + self._index, self._max_len
        )
        index_update = (
            self._call_update_index_steps
            if update_index_steps is None
            else update_index_steps
        )
        self._index.assign(
            tf.math.floormod((self._index + index_update), self._max_len)
        )
        pos_embs = tf.expand_dims(self._pos_emb(pos_ids), 0)
        # E.g. 1, T, C -> 1, C, T
        inds = list(range(rank))
        inds[t_dim], inds[1] = inds[1], inds[t_dim]
        pos_embs = tf.transpose(pos_embs, inds)

        return x + pos_embs

    def call_step(self, x: tf.Tensor) -> tf.Tensor:
        output = x + self._pos_emb((self._index))
        self._index.assign(tf.math.floormod((self._index + 1), self._max_len))
        return output

    def call_steps(self, x: tf.Tensor) -> tf.Tensor:
        return self.call_regular(x, update_index_steps=tf.shape(x)[self._temporal_dim])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        call_fn = {
            "regular": self.call_regular,
            "step": self.call_step,
            "steps": self.call_steps,
        }[self.call_mode.value]
        return call_fn(x)
