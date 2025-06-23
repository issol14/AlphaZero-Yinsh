import tensorflow as tf


@tf.function
def predict_local(model, inputs):
    """
    Local prediction function optimized with tf.function
    """
    return model(inputs) 