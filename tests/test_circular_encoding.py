import tensorflow as tf

from continual_transformers_tf import CircularPositionalEncoding


def test_CircularPositionalEncoding():
    B, C, T = 2, 3, 4

    x = tf.zeros((B, T, C))

    cpe = CircularPositionalEncoding(max_len=T, embed_dim=C, call_update_index_steps=0)
    cpe.call_mode = "regular"
    o = cpe(x)

    o_steps = cpe.call_steps(x[:, :-1])
    tf.debugging.assert_equal(o[:, :-1], o_steps)

    o_step = cpe.call_step(x[:, -1])
    tf.debugging.assert_equal(o[:, -1], o_step)
