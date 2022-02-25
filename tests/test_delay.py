import tensorflow as tf

from continual_transformers_tf import CoDelay


def test_delay():
    B, C, T = 2, 3, 4

    x = tf.random.normal((B, C, T))

    delay = CoDelay(delay=2, dtype=x.dtype, temporal_dim=-1)

    # Regular
    x2 = delay(x)
    tf.debugging.assert_equal(x, x2)

    # Steps
    x3 = delay.call_steps(x[:, :, :1])
    assert x3 is None  # still passing through

    x4 = delay.call_steps(x[:, :, 1:3])
    tf.debugging.assert_equal(x[:, :, :1], x4)

    # Step
    x5 = delay.call_step(x[:, :, 3])
    tf.debugging.assert_equal(x[:, :, 1], x5)
