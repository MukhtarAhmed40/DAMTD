import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, v)
    return output, weights

def multi_head_attention(x, num_heads=4, key_dim=32):
    # x: (batch, seq_len, dim)
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    dim = x.shape[-1]
    depth = key_dim
    # linear layers
    wq = layers.Dense(num_heads * depth)(x)
    wk = layers.Dense(num_heads * depth)(x)
    wv = layers.Dense(num_heads * depth)(x)
    # split heads
    def split_heads(x):
        x = tf.reshape(x, (batch_size, seq_len, num_heads, depth))
        return tf.transpose(x, perm=[0,2,1,3])  # (batch, heads, seq, depth)
    q = split_heads(wq)
    k = split_heads(wk)
    v = split_heads(wv)
    # compute attention per head
    # reshape to combine batch and heads for matmul
    q_ = tf.reshape(q, (-1, seq_len, depth))
    k_ = tf.reshape(k, (-1, seq_len, depth))
    v_ = tf.reshape(v, (-1, seq_len, depth))
    attn_out, _ = scaled_dot_product_attention(q_, k_, v_)
    attn_out = tf.reshape(attn_out, (batch_size, num_heads, seq_len, depth))
    attn_out = tf.transpose(attn_out, perm=[0,2,1,3])  # (batch, seq, heads, depth)
    attn_out = tf.reshape(attn_out, (batch_size, seq_len, num_heads*depth))
    # final linear
    out = layers.Dense(dim)(attn_out)
    return out

def build_convbilstm_mha(input_shape, cnn_filters=32, lstm_units=64, heads=4):
    # input_shape: (seq_len, feat_dim)
    inputs = Input(shape=input_shape, name='input')
    # per-timestep feature transform (acts like 1D conv across features)
    x = layers.TimeDistributed(layers.Dense(cnn_filters, activation='relu'))(inputs)
    # BiLSTM to capture temporal patterns
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    # Multi-head attention
    attn = multi_head_attention(x, num_heads=heads, key_dim=lstm_units//heads if heads>0 else 32)
    # Pool and classification head
    pooled = layers.GlobalAveragePooling1D()(attn)
    dense = layers.Dense(128, activation='relu')(pooled)
    outputs = layers.Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=outputs)
    return model
