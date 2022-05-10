"""
This is the official implementation of the paper:
L.-H. Chen, et al., 
"Estimating the Resize Parameter in End-to-end Learned Image Compression"
https://arxiv.org/abs/2204.12022

The purpose of this code is `educational'. To reproduce the exact results from 
the paper, tuning of hyper parameters may be necessary.
"""

import tensorflow as tf


class ResizeParamNet(tf.keras.layers.Layer):

  def __init__(self, is_train=True, *args, **kwargs):
    self.is_train = is_train
    super(ResizeParamNet, self).__init__(*args, **kwargs)

  def res_act_maxpool(self, input, f_num=32, f_size=3, s=1, name="conv2d", act_fn=tf.nn.relu):
    with tf.variable_scope(name):
      w1 = tf.get_variable('w1', [f_size, f_size, input.get_shape()[-1], f_num],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
      _ = tf.nn.conv2d(input, filter=w1, strides=1, padding='SAME')
      _ = tf.layers.batch_normalization(_, training=self.is_train, trainable=self.is_train)
      _ = act_fn(_)

      w2 = tf.get_variable('w2', [f_size, f_size, f_num, f_num],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
      _ = tf.nn.conv2d(_, filter=w2, strides=1, padding='SAME')
      _ = tf.layers.batch_normalization(_, training=self.is_train, trainable=self.is_train)

      w = tf.get_variable('w', [1, 1, input.get_shape()[-1], f_num],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
      input_1x1 = tf.nn.conv2d(input, filter=w, strides=1, padding='SAME')

      _ = _ + input_1x1
      _ = tf.layers.max_pooling2d(_, s, s)
      _ = act_fn(_)

    return _

  def call(self, tensor):
    tensor = self.res_act_maxpool(tensor, 16, 3, 2, "layer_0")
    tensor = self.res_act_maxpool(tensor, 32, 3, 2, "layer_1")
    tensor = self.res_act_maxpool(tensor, 64, 3, 2, "layer_2")
    tensor = tf.reduce_mean(tensor, axis=[1,2,3])
    tensor = tf.reshape(tensor, [-1, 1])
    tensor = tf.nn.relu(tensor)

    return tensor


class PreSubSample(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    self.num_filters = 32
    super(PreSubSample, self).__init__(*args, **kwargs)

  def conv2d_relu(self, input, f_num=32, f_size=3, s=1, name="conv2d", act_fn=tf.nn.relu):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      w = tf.get_variable('w', [f_size, f_size, input.get_shape()[-1], f_num],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
      _ = tf.nn.conv2d(input, filter=w, strides=s, padding='SAME')
      if act_fn is not None:
        _ = act_fn(_)

    return _

  def call(self, tensor):
    tensor_skip = tensor
    tensor = self.conv2d_relu(tensor, self.num_filters, 3, 1, "pre_layer_0")
    tensor = self.conv2d_relu(tensor, self.num_filters, 3, 1, "pre_layer_1")
    tensor = self.conv2d_relu(tensor, 3,                3, 1, "pre_layer_2", tf.nn.tanh)
    tensor = tensor + tensor_skip

    return tensor


class PostReSample(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    self.num_filters = 32
    super(PostReSample, self).__init__(*args, **kwargs)

  def conv2d_relu(self, input, f_num=32, f_size=3, s=1, name="conv2d", act_fn=tf.nn.relu):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      w = tf.get_variable('w', [f_size, f_size, input.get_shape()[-1], f_num],
                          initializer=tf.truncated_normal_initializer(stddev=0.02))
      _ = tf.nn.conv2d(input, filter=w, strides=s, padding='SAME')
      if act_fn is not None:
        _ = act_fn(_)

    return _

  def call(self, tensor):
    tensor_skip = tensor
    tensor = self.conv2d_relu(tensor, self.num_filters, 3, 1, "post_layer_0")
    tensor = self.conv2d_relu(tensor, self.num_filters, 3, 1, "post_layer_1")
    tensor = self.conv2d_relu(tensor, 3,                3, 1, "post_layer_2", tf.nn.tanh)
    tensor = tensor + tensor_skip

    return tensor
