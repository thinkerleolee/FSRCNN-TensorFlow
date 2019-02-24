import tensorflow as tf
from utils import tf_ssim


class Model(object):

    def __init__(self, config):
        self.name = "FSRCNN"
        # Different model layer counts and filter sizes for FSRCNN vs FSRCNN-s (fast), (d, s, m) in paper
        model_params = [56, 12, 4, 1]
        self.model_params = model_params
        self.scale = config.scale
        self.radius = config.radius
        self.padding = config.padding
        self.images = config.images
        self.batch = config.batch
        self.image_size = config.image_size - self.padding
        self.label_size = config.label_size

    def model(self, fast):
        if fast:
            self.model_params = [32, 5, 1, 1]
        d, s, m, r = self.model_params
        # Feature Extraction
        with tf.variable_scope("feature_extraction_block") as scope:
            size = self.padding + 1
            # 定义权重和偏置变量
            # shape参数是一个问四维矩阵，前面两个维度代表了过滤器的尺寸
            # 第三个维度表示当前层的深度 第四个表示filter深度
            weights = tf.get_variable('feature_extraction_w', shape=[size, size, 1, d],
                                      initializer=tf.variance_scaling_initializer())
            biases = tf.get_variable('feature_extraction_b', initializer=tf.zeros([d]))
            # 卷积层，stride=1，padding=0
            features = tf.nn.conv2d(self.images, weights, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
            # 添加偏置
            features = tf.nn.bias_add(features, biases, data_format='NHWC')

        # Shrinking
        with tf.variable_scope("shrinking_block") as scope:
            if self.model_params[1] > 0:
                features = self.prelu(features, 1)
                weights = tf.get_variable('shrinking_w', shape=[1, 1, d, s],
                                          initializer=tf.variance_scaling_initializer())
                biases = tf.get_variable('shrinking_b', initializer=tf.zeros([s]))
                features = tf.nn.conv2d(features, weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                features = tf.nn.bias_add(features, biases, data_format='NHWC')
            else:
                s = d

        conv = features
        # Mapping (# mapping layers = m)
        with tf.variable_scope("mapping_block") as scope:
            for ri in range(r):
                for i in range(3, m + 3):
                    weights = tf.get_variable('w{}'.format(i), shape=[3, 3, s, s],
                                              initializer=tf.variance_scaling_initializer())
                    biases = tf.get_variable('b{}'.format(i), initializer=tf.zeros([s]))
                    if i > 3:
                        conv = self.prelu(conv, i)
                    conv = tf.nn.conv2d(conv, weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                    conv = tf.nn.bias_add(conv, biases, data_format='NHWC')
                    if i == m + 2:
                        conv = self.prelu(conv, m + 3)
                        weights = tf.get_variable('w{}'.format(m + 3), shape=[1, 1, s, s],
                                                  initializer=tf.variance_scaling_initializer())
                        biases = tf.get_variable('b{}'.format(m + 3), initializer=tf.zeros([s]))
                        conv = tf.nn.conv2d(conv, weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                        conv = tf.nn.bias_add(conv, biases, data_format='NHWC')
                        conv = tf.add(conv, features)
                scope.reuse_variables()
        conv = self.prelu(conv, 2)

        # Expanding
        with tf.variable_scope("expanding_block") as scope:
            if self.model_params[1] > 0:
                expand_weights = tf.get_variable('w{}'.format(m + 4), shape=[1, 1, s, d],
                                                 initializer=tf.variance_scaling_initializer())
                expand_biases = tf.get_variable('b{}'.format(m + 4), initializer=tf.zeros([d]))
                conv = tf.nn.conv2d(conv, expand_weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                conv = tf.nn.bias_add(conv, expand_biases, data_format='NHWC')
                conv = self.prelu(conv, m + 4)

        # Sub-pixel convolution
        with tf.variable_scope("deconvolution_block") as scope:
            size = self.radius * 2 + 1
            deconv_weights = tf.get_variable('deconv_w', shape=[size, size, d, self.scale ** 2],
                                             initializer=tf.variance_scaling_initializer(scale=0.01))
            deconv_biases = tf.get_variable('deconv_b', initializer=tf.zeros([self.scale ** 2]))
            deconv = tf.nn.conv2d(conv, deconv_weights, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            deconv = tf.nn.bias_add(deconv, deconv_biases, data_format='NHWC')
            deconv = tf.depth_to_space(deconv, self.scale, name='pixel_shuffle', data_format='NHWC')

        # 返回反卷积层输出
        return deconv

    def prelu(self, _x, i):
        """
        PreLU tensorflow implementation
        """
        alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.2),
                                 dtype=tf.float32)

        return tf.nn.relu(_x) - alphas * tf.nn.relu(-_x)

    def loss(self, Y, X):
        dY = tf.image.sobel_edges(Y)
        dX = tf.image.sobel_edges(X)
        M = tf.sqrt(tf.square(dY[:, :, :, :, 0]) + tf.square(dY[:, :, :, :, 1]))
        return tf.losses.absolute_difference(dY, dX) \
               + tf.losses.absolute_difference((1.0 - M) * Y, (1.0 - M) * X, weights=2.0)
