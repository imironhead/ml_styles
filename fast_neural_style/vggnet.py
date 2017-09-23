"""
"""
import numpy as np
import tensorflow as tf


class GFileAdapter(object):
    """
    Duck type for numpy.load(...) and GFile.
    numpy.load need a file object which has seek/read/readline methods.
    GFile can read data from Google Cloud Storage.
    """
    def __init__(self, path):
        """
        Initialize the adapter.
        """
        self._gfile = tf.gfile.GFile(path)

    def seek(self, offset, whence=0):
        """
        Implement the seek method of the file object.

        offset:
            distance to move current position.
        whence:
            initial position of seeking.
            0: seek from the head.
            1: seek from current position.
            2: seek from eof.
        """
        if whence == 1:
            # adjust offset base on current position.
            offset += self._gfile.tell()
        elif whence == 2:
            # adjust offset base on eof.
            offset += self._gfile.size()

        return self._gfile.seek(offset)

    def read(self, size):
        """
        Read data.

        size:
            how much byte to read.
        """
        return self._gfile.read(size)

    def readline(self):
        """
        Read a line.
        """
        return self._gfile.readline()


class VggNet(object):
    """
    Implement a class to represent a vgg net.

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556
    """
    @staticmethod
    def mean_color_bgr():
        """
        mean color from training data in B/G/R order.
        """
        return [103.939, 116.779, 123.68]

    @staticmethod
    def mean_color_rgb():
        """
        mean color from training data in R/G/B order.
        """
        return [123.68, 116.779, 103.939]

    @staticmethod
    def build_16(path_npy, batch_tensors, end_layer='@_@', reuse=None):
        """
        Virtual constructor to construct a VGG 16 (D) net.
        """
        return VggNet('vgg_16', path_npy, batch_tensors, end_layer, reuse)

    @staticmethod
    def build_19(path_npy, batch_tensors, end_layer='@_@', reuse=None):
        """
        Virtual constructor to construct a VGG 19 (E) net.
        """
        return VggNet('vgg_19', path_npy, batch_tensors, end_layer, reuse)

    @staticmethod
    def pool(tensors, name):
        """
        Connect to a pool layer and return the result.
        """
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]

        return tf.nn.max_pool(
            tensors, ksize=ksize, strides=strides, padding='SAME', name=name)

    @staticmethod
    def conv(weights, tensors, name, reuse=None):
        """
        Connect to a conv2d layer and return the result.
        """
        with tf.variable_scope(name, reuse=reuse):
            const_filter = tf.constant(weights[name][0], name='filter')
            const_biases = tf.constant(weights[name][1], name='biases')

            tensors = tf.nn.conv2d(
                tensors, const_filter, [1, 1, 1, 1], padding='SAME')
            tensors = tf.nn.bias_add(tensors, const_biases)
            tensors = tf.nn.relu(tensors)

            return tensors

    @staticmethod
    def fc(weights, tensors, name, reuse=None):
        """
        Connect to a fully connected layer and return the result.
        """
        with tf.variable_scope(name, reuse=reuse):
            const_weights = tf.constant(weights[name][0], name='weights')
            const_biases = tf.constant(weights[name][1], name='biases')

            dim = np.prod(tensors.get_shape()[1:])

            tensors = tf.reshape(tensors, [-1, dim])
            tensors = tf.matmul(tensors, const_weights)
            tensors = tf.nn.bias_add(tensors, const_biases)

            return tensors

    def __init__(self, model, path_npy, batch_tensors, end_layer, reuse=None):
        """
        Initialize the VGG net.

        model:
            'vgg_16' or 'vgg_19'.
        path_npy:
            path to a npy file which has all the weights for the net.
        batch_tensors:
            a upstream tensors. The shape must be [-1, 224, 224, 3].
            color channels should be in B/G/R order.
            color range should be between 0.0 and 255.0.
            mean colors should have been subtracted.
        end_layer:
            The end layer name. The net will be build with layers before the
            end layer.
        """
        # sanity check: model
        if model not in ['vgg_16', 'vgg_19']:
            raise Exception('invalid model: {}'.format(model))

        # sanity check: checkpoint
        if path_npy is None or not tf.gfile.Exists(path_npy):
            raise Exception('invalid checkpoint: {}'.format(path_npy))

        # sanity check: input tensors
        if batch_tensors is None:
            raise Exception('invalid tensors: {}'.format(batch_tensors))

        # if batch_tensors.get_shape().as_list()[1:] != [224, 224, 3]:
        #     raise Exception('invalid tensors: {}'.format(batch_tensors))

        # choose architecture:
        if model == 'vgg_16':
            layer_names = [
                'conv1_1', 'conv1_2', 'pool1',
                'conv2_1', 'conv2_2', 'pool2',
                'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
                'conv5_1', 'conv5_2', 'conv5_3', 'pool5',
                'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'prop']
        else:
            layer_names = [
                'conv1_1', 'conv1_2', 'pool1',
                'conv2_1', 'conv2_2', 'pool2',
                'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
                'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
                'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5',
                'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'prop']

        # load the weights
        checkpoint = GFileAdapter(path_npy)

        weights = np.load(checkpoint, encoding='latin1').item()

        # keep layers here
        self._layers = {'upstream': batch_tensors}

        # build
        for layer_name in layer_names:
            if layer_name == end_layer:
                break
            elif layer_name.startswith('conv'):
                batch_tensors = VggNet.conv(
                    weights, batch_tensors, layer_name, reuse=reuse)
            elif layer_name.startswith('pool'):
                batch_tensors = VggNet.pool(batch_tensors, layer_name)
            elif layer_name.startswith('fc'):
                batch_tensors = VggNet.fc(
                    weights, batch_tensors, layer_name, reuse=reuse)
            elif layer_name.startswith('relu'):
                batch_tensors = tf.nn.relu(batch_tensors)
            elif layer_name.startswith('prop'):
                batch_tensors = tf.nn.softmax(batch_tensors, name=layer_name)
            else:
                raise Exception('-_-')

            self._layers[layer_name] = batch_tensors

        self._layers['downstream'] = batch_tensors

    def __getattr__(self, name):
        """
        The properties of this net.
        """
        if name not in self._layers:
            raise Exception('invalid property: {}'.format(name))

        return self._layers[name]
