import abc
from enum import Enum

import tensorflow.compat.v2 as tf


class CallMode(Enum):
    REGULAR = "regular"
    STEPS = "steps"
    STEP = "step"


class WithCallMode(abc.ABC):
    @property
    def call_mode(self) -> CallMode:
        return self._call_mode

    @call_mode.setter
    def call_mode(self, value):
        self._call_mode = CallMode(value)

    @abc.abstractmethod
    def call_regular(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def call_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def call_steps(self, *args, **kwargs):
        pass

    def call(self, *args, **kwargs):
        call_fn = {
            "regular": self.call_regular,
            "step": self.call_step,
            "steps": self.call_steps,
        }[self.call_mode.value]
        return call_fn(*args, **kwargs)


def enqueue_and_peek_all(queue: tf.queue.FIFOQueue, val: tf.Tensor, capacity: int):
    queue.dequeue()
    queue.enqueue(val)
    # Currently, there is no peek operation in tf queues.
    # Unfortunately, this necessitates a full dequeue and encueue.
    vals = queue.dequeue_many(capacity)
    queue.enqueue_many(vals)
    # [L, B, ...] -> [B, L, ...]
    return tf.experimental.numpy.swapaxes(vals, 0, 1)
