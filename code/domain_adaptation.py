import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x):
        @tf.custom_gradient
        def _flip(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad
        return _flip(x)

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = _pdist(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def _pdist(a, b):
    a_norm = tf.reduce_sum(tf.square(a), axis=1)
    b_norm = tf.reduce_sum(tf.square(b), axis=1)
    a_norm = tf.reshape(a_norm, [-1,1])
    b_norm = tf.reshape(b_norm, [1,-1])
    return tf.maximum(a_norm + b_norm - 2.0*tf.matmul(a, tf.transpose(b)), 0.0)

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2*tf.reduce_mean(kernel(x, y))
    return cost
