import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import Sequential
from tensorflow.compat.v2.keras.layers import (  # from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
)

from .co_delay import CoDelay
from .co_si_mha import CoSiMultiHeadAttention
from .common import WithCallMode


class CoSiTransformerEncoder(WithCallMode, Layer):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate=0.1,
        call_mode="regular",
    ):
        Layer.__init__(self)
        self.att = CoSiMultiHeadAttention(
            seq_len=seq_len, num_heads=num_heads, key_dim=embed_dim
        )
        self.delay = CoDelay(delay=seq_len - 1)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="gelu"),
                Dense(embed_dim),
            ]
        )
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self._temporal_dim = 1
        self._seq_len = seq_len
        self.call_mode = call_mode

    def call_regular(self, inputs, training):
        sel = tf.gather(inputs, indices=[self._seq_len - 1], axis=self._temporal_dim)
        attn_output = self.att.call_regular(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.ln1(sel + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.ln2(out1 + ffn_output)

    def call_step(self, input, training=None):
        sel = input
        attn_output = self.att.call_step(input, input)
        if sel is None or attn_output is None:
            return None
        attn_output = self.dropout1(attn_output, training=training)
        out1 = sel + attn_output
        out1 = tf.expand_dims(out1, axis=self._temporal_dim)
        out1 = self.ln1(out1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.ln2(out1 + ffn_output)
        out2 = tf.squeeze(out2, axis=self._temporal_dim)
        return out2

    def call_steps(self, inputs, training=None):
        attn_output = self.att.call_steps(inputs, inputs)
        if attn_output is None:
            return None
        L = tf.shape(inputs)[self._temporal_dim]
        T = tf.shape(attn_output)[self._temporal_dim]
        sel = tf.gather(
            inputs,
            indices=range(L - T, L),
            axis=self._temporal_dim,
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.ln1(sel + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.ln2(out1 + ffn_output)
        return out2
