import tensorflow as tf

from continual_transformers_tf import CoSiTransformerEncoder


def test_co_si_trans_enc():
    tf.keras.backend.set_learning_phase(0)

    B = 3  # batch size
    L = 10  # target sequence length
    E = 2  # embed dimension

    # Module
    layer = CoSiTransformerEncoder(
        seq_len=L,
        embed_dim=E,
        num_heads=2,
        ff_dim=16,
        dropout_rate=0.1,
        call_mode="regular",
    )

    # Data
    def update(olds, step):
        return tf.concat([olds[:, 1:], tf.expand_dims(step, 1)], axis=1)

    x0 = tf.random.normal((B, L, E))
    x_step = tf.random.normal((B, E))
    x1 = update(x0, x_step)
    tf.debugging.assert_equal(x0[:, 1:], x1[:, :-1])

    # Regular
    t0 = layer(x0)
    t1 = layer(x1)

    # Steps
    assert layer.call_steps(x0[:, :1]) is None  # still warming up
    assert layer.call_step(x0[:, 1]) is None  # still warming up
    o0 = layer.call_steps(x0[:, 2:])
    tf.debugging.assert_near(t0, o0)

    # Step
    o1 = layer.call_step(x_step)
    tf.debugging.assert_near(t1, tf.expand_dims(o1, axis=1))
