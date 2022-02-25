import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

from continual_transformers_tf import CoSiMultiHeadAttention


def test_multi_head_attention():
    B = 2  # batch size
    L = 10  # target sequence length
    K = 6  # key dimension
    H = 4  # num heads

    # Define data
    query = tf.random.normal((B, L, H))
    key = tf.random.normal((B, L, H))
    value = tf.random.normal((B, L, H))

    query_step2 = tf.random.normal((B, H))
    value_step2 = tf.random.normal((B, H))
    key_step2 = tf.random.normal((B, H))

    def update(olds, step):
        return tf.concat([olds[:, 1:, :], tf.expand_dims(step, 1)], axis=1)

    # and the corresponding full Q,K,V:
    query2 = update(query, query_step2)
    value2 = update(value, value_step2)
    key2 = update(key, key_step2)
    tf.debugging.assert_near(value2[:, :-1], value[:, 1:])  # sanity check

    query_step3 = tf.random.normal((B, H))
    value_step3 = tf.random.normal((B, H))
    key_step3 = tf.random.normal((B, H))

    query3 = update(query2, query_step3)
    value3 = update(value2, value_step3)
    key3 = update(key2, key_step3)

    # Init regular version
    mha = MultiHeadAttention(num_heads=H, key_dim=K)

    # call instantiates EinsumDense layers
    a1, s1 = mha(query, value, key, return_attention_scores=True)

    # Init Continual Single-output version
    comha = CoSiMultiHeadAttention(seq_len=L, num_heads=H, key_dim=K)
    # instantiate EinsumDense layers
    comha.call_steps(query, value, key)
    comha.clean_state()
    # comha(query, value, key)  # alternative
    comha.set_weights(mha.get_weights())

    # Regular call is unchanged
    a2 = comha(query, value, key)
    tf.debugging.assert_near(a1[:, -1:], a2)

    # Continual calls arrive at same result
    # for i in range(L - 1):
    #     # Warm up
    #     assert comha.call_step(query[:, i], value[:, i], key[:, i]) is None
    # a3, s3 = comha.call_step(query[:,-1], value[:,-1], key[:,-1], return_attention_scores=True)
    a3, s3 = comha.call_steps(query, value, key, return_attention_scores=True)
    tf.debugging.assert_near(a1[:, -1:], a3)
    tf.debugging.assert_near(s1[:, :, -1:], s3)

    a4 = mha(query2, value2, key2)
    a5 = comha.call_step(query_step2, value_step2, key_step2)
    tf.debugging.assert_near(a4[:, -1], a5)

    # A similar behaviour can be overloaded by setting call_mode
    comha.call_mode = "step"

    a6 = mha(query3, value3, key3)
    a7 = comha(query_step3, value_step3, key_step3)
    tf.debugging.assert_near(a6[:, -1], a7)
