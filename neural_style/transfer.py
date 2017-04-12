"""
"""
import tensorflow as tf

from six.moves import range
from vggnet import VggNet


tf.app.flags.DEFINE_float('content-weight', 1.0, '')
tf.app.flags.DEFINE_float('style-weight', 1000.0, '')
tf.app.flags.DEFINE_string('content-path', './image.jpg', '')
tf.app.flags.DEFINE_string('style-path', './style.jpg', '')
tf.app.flags.DEFINE_string('vgg19-path', '@_@', '')
tf.app.flags.DEFINE_string('log-path', '@_@', '')
tf.app.flags.DEFINE_integer('content-level', 2, '0 ~ 5')
tf.app.flags.DEFINE_integer('style-level', 2, '0 ~ 5')


def sanity_check():
    """
    """
    path_content = tf.app.flags.FLAGS.content_path
    path_style = tf.app.flags.FLAGS.style_path
    path_vgg19 = tf.app.flags.FLAGS.vgg19_path
    path_log = tf.app.flags.FLAGS.log_path

    # check path to content image.
    if path_content is None or not tf.gfile.Exists(path_content):
        raise Exception('invalid content path: {}'.format(path_content))

    # check path to style image.
    if path_style is None or not tf.gfile.Exists(path_style):
        raise Exception('invalid style path: {}'.format(path_style))

    # check path to vgg 19 weights
    if path_vgg19 is None or not tf.gfile.Exists(path_vgg19):
        raise Exception('invalid vgg19 path: {}'.format(path_vgg19))

    #
    if path_log is None or not tf.gfile.Exists(path_log):
        raise Exception('invalid log path: {}'.format(path_log))

    level_content = tf.app.flags.FLAGS.content_level
    level_style = tf.app.flags.FLAGS.style_level

    # check level of content
    if level_content is None or level_content < 0 or level_content > 5:
        raise Exception('invalid content level: {}'.format(level_content))

    # check level of style
    if level_style is None or level_style < 0 or level_style > 5:
        raise Exception('invalid style level: {}'.format(level_style))

    # content and style level can not be zero and the same time.
    if level_content + level_style == 0:
        raise Exception('content_level + style_level must not be zero')

    weight_content = tf.app.flags.FLAGS.content_weight
    weight_style = tf.app.flags.FLAGS.style_weight

    if weight_content is None or weight_content < 0.0:
        raise Exception('invalid content weight: {}'.format(weight_content))

    if weight_style is None or weight_style < 0.0:
        raise Exception('invalid style weight: {}'.format(weight_style))


def load_image(path):
    """
    """
    file_names = tf.train.string_input_producer([path])

    _, image = tf.WholeFileReader().read(file_names)

    # Decode byte data, no gif please.
    # NOTE: tf.image.decode_image can decode both jpeg and png. However, the
    #       shape (height and width) is unknown.
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)[:2]
    image = tf.image.resize_images(image, [224, 224])
    image = tf.reshape(image, [1, 224, 224, 3])

    # for VggNet, subtract the mean color of it's training data.
    image = tf.subtract(image, VggNet.mean_color_rgb())

    # R/G/B to B/G/R
    image = tf.reverse(image, [3])

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image, shape = session.run([image, shape])

        coord.request_stop()
        coord.join(threads)

        return image, shape


def build_vgg():
    """
    """
    level_content = tf.app.flags.FLAGS.content_level
    level_style = tf.app.flags.FLAGS.style_level

    content_end_layers = \
        ['', 'pool1', 'pool2', 'conv3_3', 'conv4_3', 'conv5_3']
    style_end_layers = \
        ['', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

    # ignore unnecessary layers
    if level_content >= level_style:
        end_layer = content_end_layers[level_content]
    else:
        end_layer = style_end_layers[level_style]

    # a 'trainable' variable for:
    #   variable of content image to get content features.
    #   variable of style image to get style features.
    #   variable of result image.
    upstream = tf.get_variable(
        'upstream',
        [1, 224, 224, 3],
        initializer=tf.random_uniform_initializer(0.0, 1.0))

    return VggNet.build_19(tf.app.flags.FLAGS.vgg19_path, upstream, end_layer)


def content_features(vgg, content):
    """
    """
    level_content = tf.app.flags.FLAGS.content_level

    last_layer = 'conv{}_2'.format(level_content)

    with tf.Session() as session:
        # assign the content image to the upstream of the vgg.
        set_upstream = vgg.upstream.assign(content)

        session.run(set_upstream)

        # get the content features from the last layers.
        return session.run(vgg.__getattr__(last_layer))


def style_features(vgg, style):
    """
    """
    level_style = tf.app.flags.FLAGS.style_level

    features = []

    for i in range(1, level_style + 1):
        layer = vgg.__getattr__('conv{}_1'.format(i))

        # transpose to [batch, channel, height, width]
        layer = tf.transpose(layer, [0, 3, 1, 2])

        shape = tf.shape(layer)

        count, channels, length = shape[0], shape[1], shape[2] * shape[3]

        # flatten pixels of each style channel.
        layer = tf.reshape(layer, (count, channels, length))

        # gram matrix of style
        gram = tf.matmul(layer, layer, transpose_b=True)

        features.append(gram)

    with tf.Session() as session:
        # assign the style image to the upstream of vgg.
        set_upstream = vgg.upstream.assign(style)

        session.run(set_upstream)

        # get the style features (gram matrix) from each layer.
        return session.run(features)


def content_loss(vgg, content):
    """
    """
    level_content = tf.app.flags.FLAGS.content_level

    # reconstruct only style if content level is 0.
    if level_content == 0:
        return 0.0

    # we want to reconstruct a new image with simular content.
    target_features = content_features(vgg, content)

    target_features = tf.constant(target_features)

    # get the feature of the image we are reconstructing.
    last_layer = 'conv{}_2'.format(level_content)

    source_features = vgg.__getattr__(last_layer)

    #
    weight_content = tf.app.flags.FLAGS.content_weight

    return weight_content * tf.nn.l2_loss(source_features - target_features)


def style_loss(vgg, style):
    """
    """
    level_style = tf.app.flags.FLAGS.style_level

    # reconstruct only high level content if style level is 0.
    if level_style == 0:
        return 0.0

    # we want to reconstruct a new image with simular style features.
    target_features = style_features(vgg, style)

    #
    all_loss = 0

    for i in range(1, level_style + 1):
        layer = vgg.__getattr__('conv{}_1'.format(i))

        # transpose to [batch, channel, height, width]
        layer = tf.transpose(layer, [0, 3, 1, 2])

        shape = tf.shape(layer)

        count, channels, length = shape[0], shape[1], shape[2] * shape[3]

        # flatten pixels of each layer.
        layer = tf.reshape(layer, (count, channels, length))

        # the gram matrix
        gram = tf.matmul(layer, layer, transpose_b=True)

        # length * length: Ml (height * width)
        # level_style: weight of style loss for each layer.
        m = tf.cast(length * length * level_style, tf.float32)

        all_loss += tf.nn.l2_loss(gram - target_features[i - 1]) / m

    weight_style = tf.app.flags.FLAGS.style_weight

    return weight_style * all_loss


def transfer_loss(vgg, content, style):
    """
    """
    return content_loss(vgg, content) + style_loss(vgg, style)


def transfer_trainer(vgg, loss):
    """
    """
    return tf.train.AdamOptimizer(1e-2).minimize(loss)


def transfer_reporter():
    """
    """
    return tf.summary.FileWriter(tf.app.flags.FLAGS.log_path)


def transfer_summary(vgg, loss, content_shape):
    """
    summaries of loss and result image.
    """
    image = tf.add(vgg.upstream, VggNet.mean_color_bgr())
    image = tf.image.resize_images(image, content_shape)
    image = tf.saturate_cast(image, tf.uint8)
    image = tf.reverse(image, [3])

    summary_image = tf.summary.image('generated image', image, max_outputs=1)

    summary_loss = tf.summary.scalar('transfer loss', loss)

    return tf.summary.merge([summary_image, summary_loss])


def main(_):
    """
    arXiv:1508.06576

    A Neural Algorithm of Artistic Style
    """
    sanity_check()

    content, content_shape = load_image(tf.app.flags.FLAGS.content_path)

    style, style_shape = load_image(tf.app.flags.FLAGS.style_path)

    vgg = build_vgg()

    loss = transfer_loss(vgg, content, style)

    trainer = transfer_trainer(vgg, loss)

    reporter = transfer_reporter()

    summary = transfer_summary(vgg, loss, content_shape)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # start optimizing from a random image.
        noise = tf.random_uniform([1, 224, 224, 3], 0.0, 255.0)

        # subtract mean color for vgg.
        # reversing is not necessary since the image is a random one.
        noise = tf.subtract(noise, VggNet.mean_color_bgr())

        initialize_upstream = vgg.upstream.assign(noise)

        # assign the random image to the input tensor of vgg.
        # the result image will be generated on the upstream of vgg.
        session.run(initialize_upstream)

        for step in range(100000000):
            _loss, _summary, _ = session.run([loss, summary, trainer])

            if step % 500 == 0:
                print('step: {}, loss: {}'.format(step, _loss))

                reporter.add_summary(_summary, step)


if __name__ == '__main__':
    tf.app.run()
