"""
"""
import tensorflow as tf

from six.moves import range
from vggnet import VggNet


def instance_norm(flow, name):
    """
    arXiv:1607.08022v2
    """
    with tf.variable_scope(name):
        depth = flow.get_shape()[3]
        scale = tf.get_variable(
            'scale',
            [depth],
            initializer=tf.random_normal_initializer(1.0, 0.02))
        shift = tf.get_variable(
            'shift',
            [depth],
            initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(flow, axes=[1, 2], keep_dims=True)

        flow = (flow - mean) / tf.sqrt(variance + 1e-5)

        return scale * flow + shift


def build_vgg(
        vgg19_path, upstream, level_content=2, level_style=2, reuse=None):
    """
    """
    content_end_layers = \
        ['', 'pool1', 'pool2', 'conv3_3', 'conv4_3', 'conv5_3']
    style_end_layers = \
        ['', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']
    # ['', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5']

    # ignore unnecessary layers
    if level_content >= level_style:
        end_layer = content_end_layers[level_content]
    else:
        end_layer = style_end_layers[level_style]

    return VggNet.build_19(vgg19_path, upstream, end_layer, reuse=reuse)


def build_style_transfer_image_transform_net(upstream):
    """
    """
    # model as jcjohnson
    # https://github.com/jcjohnson/fast-neural-style/blob/master/train.lua
    # c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    batch_tensors = upstream

    # batch_tensors = tf.pad(
    #     tensor=batch_tensors,
    #     paddings=[[0, 0], [16, 16], [16, 16], [0, 0]],
    #     mode='symmetric')

    # c9s1-32
    batch_tensors = tf.contrib.layers.convolution2d(
        inputs=batch_tensors,
        num_outputs=32,
        kernel_size=9,
        stride=1,
        padding='SAME',
        activation_fn=None,
        # normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_fn=None,
        weights_initializer=weights_initializer,
        scope='conv_c9s1_32')

    batch_tensors = instance_norm(batch_tensors, 'conv_c9s1_32_norm')

    batch_tensors = tf.nn.relu(batch_tensors)

    # d64, d128
    for num_outputs in [64, 128]:
        batch_tensors = tf.contrib.layers.convolution2d(
            inputs=batch_tensors,
            num_outputs=num_outputs,
            kernel_size=4,
            stride=2,
            padding='SAME',
            activation_fn=None,
            # normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_fn=None,
            weights_initializer=weights_initializer,
            scope='conv_d{}'.format(num_outputs))

        batch_tensors = instance_norm(
            batch_tensors, 'conv_d{}_norm'.format(num_outputs))

        batch_tensors = tf.nn.relu(batch_tensors)

    # R128 x 5
    for i in range(1):
        temp_batch_tensors = batch_tensors

        batch_tensors = tf.contrib.layers.convolution2d(
            inputs=batch_tensors,
            num_outputs=128,
            kernel_size=4,
            stride=1,
            padding='SAME',
            activation_fn=None,
            # normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_fn=None,
            weights_initializer=weights_initializer,
            scope='conv_R_{}_1'.format(i))

        batch_tensors = instance_norm(
            batch_tensors, 'conv_R_{}_1_norm'.format(i))

        batch_tensors = tf.nn.relu(batch_tensors)

        batch_tensors = temp_batch_tensors + tf.contrib.layers.convolution2d(
            inputs=batch_tensors,
            num_outputs=128,
            kernel_size=4,
            stride=1,
            padding='SAME',
            weights_initializer=weights_initializer,
            scope='conv_R_{}_2'.format(i))

    # u64, u32
    for num_outputs in [64, 32]:
        batch_tensors = tf.contrib.layers.convolution2d_transpose(
            inputs=batch_tensors,
            num_outputs=num_outputs,
            kernel_size=4,
            stride=2,
            padding='SAME',
            activation_fn=None,
            # normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_fn=None,
            weights_initializer=weights_initializer,
            scope='conv_u_{}'.format(num_outputs))

        batch_tensors = instance_norm(
            batch_tensors, 'conv_u_{}_norm'.format(num_outputs))

        batch_tensors = tf.nn.relu(batch_tensors)

    # c9s1-3
    batch_tensors = tf.contrib.layers.convolution2d(
        inputs=batch_tensors,
        num_outputs=3,
        kernel_size=9,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.tanh,
        weights_initializer=weights_initializer,
        scope='conv_c9s1_3')

    # batch_tensors = tf.slice(
    #     batch_tensors, [0, 16, 16, 0], [-1, 256, 256, -1])

    return batch_tensors


def style_features(vgg_style, style_level=2):
    """
    """
    features = []

    for i in range(1, style_level + 1):
        layer = vgg_style.__getattr__('conv{}_1'.format(i))

        # transpose to [batch, channel, height, width]
        layer = tf.transpose(layer, [0, 3, 1, 2])

        shape = tf.shape(layer)

        count, channels, length = shape[0], shape[1], shape[2] * shape[3]

        # flatten pixels of each style channel.
        layer = tf.reshape(layer, (count, channels, length))

        # gram matrix of style
        gram = tf.matmul(layer, layer, transpose_b=True)

        features.append([gram, channels, length])

    return features


def content_loss(vgg_styled, vgg_content, content_level=2, content_weight=1.0):
    """
    """
    # reconstruct only style if content level is 0.
    if content_level == 0:
        return 0.0

    # get the feature of the image we are reconstructing.
    last_layer = 'conv{}_2'.format(content_level)

    target_features = vgg_content.__getattr__(last_layer)

    source_features = vgg_styled.__getattr__(last_layer)

    return content_weight * tf.nn.l2_loss(source_features - target_features)


def style_loss(vgg_styled, vgg_style, style_level=2, style_weight=1000.0):
    """
    """
    # reconstruct only high level content if style level is 0.
    if style_level == 0:
        return 0.0

    # we want to reconstruct a new image with simular style features.
    target_features = style_features(vgg_style, style_level)

    source_features = style_features(vgg_styled, style_level)

    with tf.Session() as session:
        # get the style features (gram matrix) from each layer.
        features = [f[0] for f in target_features]
        parameters = [f[1:] for f in target_features]

        features = session.run(features)

        target_features = [[f] + parameters[i] for i, f in enumerate(features)]

    #
    all_loss = 0

    for i in range(style_level):
        t_gram, channels, length = target_features[i]
        s_gram, channels, length = source_features[i]

        length = tf.cast(length, tf.float32)

        channels = tf.cast(channels, tf.float32)

        m = tf.cast(length * length * channels, tf.float32)

        all_loss += tf.nn.l2_loss(s_gram - t_gram) / m

    return style_weight * all_loss


def transfer_loss(
        vgg_styled,
        vgg_content,
        vgg_style,
        style_level=2,
        style_weight=1000.0,
        content_level=2,
        content_weight=1.0):
    """
    """
    lc = content_loss(vgg_styled, vgg_content, content_level, content_weight)
    ls = style_loss(vgg_styled, vgg_style, style_level, style_weight)

    return lc, ls


def build_super_resolution_network(
        image_sd,
        image_hd,
        vgg16_path,
        checkpoint_path,
        training=True):
    """
    """


def build_style_transfer_network(
        image_content,
        image_style,
        vgg16_path,
        content_weight=1.0,
        content_level=2,
        style_weight=1000.0,
        style_level=2,
        training=True):
    """
    """
    image_styled = build_style_transfer_image_transform_net(image_content)

    network = {
        'image_content': image_content,
        'image_styled': image_styled,
    }

    if training:
        # FIXME: load vgg16 multiple times?

        # to 0 ~ 255
        image_content = image_content * 127.5 + 127.5

        image_styled = image_styled * 127.5 + 127.5

        image_style = image_style * 127.5 + 127.5

        image_content = tf.subtract(image_content, VggNet.mean_color_bgr())

        image_styled = tf.subtract(image_styled, VggNet.mean_color_bgr())

        image_style = tf.subtract(image_style, VggNet.mean_color_bgr())

        vgg_styled = build_vgg(
            vgg16_path, image_styled, content_level, style_level, reuse=None)

        vgg_style = build_vgg(
            vgg16_path, image_style, content_level, style_level, reuse=True)

        vgg_content = build_vgg(
            vgg16_path, image_content, content_level, style_level, reuse=True)

        # global step
        step = tf.get_variable(
            'global_step',
            [],
            trainable=False,
            initializer=tf.constant_initializer(0, dtype=tf.int64),
            dtype=tf.int64)

        loss_content, loss_style = transfer_loss(
            vgg_styled,
            vgg_content,
            vgg_style,
            style_level=style_level,
            style_weight=style_weight,
            content_level=content_level,
            content_weight=content_weight)

        train_content = tf.train.AdamOptimizer(1e-3)
        train_content = train_content.minimize(loss_content, global_step=step)

        train_transfer = tf.train.AdamOptimizer(1e-3)
        train_transfer = train_transfer.minimize(
            loss_content + loss_style, global_step=step)

        network['step'] = step
        network['loss_content'] = loss_content
        network['loss_transfer'] = loss_content + loss_style
        network['train_content'] = train_content
        network['train_transfer'] = train_transfer

    return network
