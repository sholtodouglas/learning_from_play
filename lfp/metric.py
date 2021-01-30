import tensorflow as tf

class MaxMetric(tf.keras.metrics.Metric):

    def __init__(self, name='max_metric', **kwargs):
        super(MaxMetric, self).__init__(name=name, **kwargs)
        self.max = self.add_weight(name='max', initializer='zeros')
        self.abs_err = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def update_state(self, y_true, y_pred, mask=1.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        values = self.abs_err(y_true, y_pred) * mask
        values = tf.cast(values, self.dtype)
        max = tf.reduce_max(values)
        if tf.math.greater(max, self.max):
            self.max.assign(max)

    def result(self):
        return self.max