import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras.layers import Layer

from .common import WithCallMode


class CoDelay(WithCallMode, Layer):
    def __init__(
        self,
        delay: int,
        dtype=tf.float32,
        temporal_dim=-1,
        call_mode="regular",
    ):
        Layer.__init__(self)
        self._queue = tf.queue.FIFOQueue(
            capacity=delay,
            dtypes=[dtype],
            name="queue",
        )
        self._state_index = tf.Variable(-delay)
        self._temporal_dim = temporal_dim
        self.call_mode = call_mode

    def call_regular(self, x, **kwargs):
        return x

    def call_step(self, x, **kwargs):
        ret = None
        if self._state_index == 0:
            ret = self._queue.dequeue()
        else:
            self._state_index.assign_add(1)
        self._queue.enqueue(x)
        return ret

    def call_steps(self, x, **kwargs):
        outs = []
        for xx in tf.unstack(x, axis=self._temporal_dim):
            o = self.call_step(xx)
            if o is not None:
                outs.append(o)

        if len(outs) == 0:
            return None

        outs = tf.stack(outs, axis=self._temporal_dim)
        return outs
