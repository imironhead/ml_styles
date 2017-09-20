"""
"""
import glob
import os
import tensorflow as tf

from model import build_style_transfer_network

tf.app.flags.DEFINE_string('contents-path', None, '')
tf.app.flags.DEFINE_string('content-path', None, '')
tf.app.flags.DEFINE_string('style-path', None, '')
tf.app.flags.DEFINE_float('content-weight', 1.0, '')
tf.app.flags.DEFINE_float('style-weight', 1000.0, '')
tf.app.flags.DEFINE_integer('content-level', 2, '0 ~ 5')
tf.app.flags.DEFINE_integer('style-level', 2, '0 ~ 5')

tf.app.flags.DEFINE_string('vgg16-path', None, '')
tf.app.flags.DEFINE_string('ckpt-path', None, '')
tf.app.flags.DEFINE_string('logs-path', None, '')
tf.app.flags.DEFINE_boolean('train', True, '')
tf.app.flags.DEFINE_integer('batch-size', 4, '')
tf.app.flags.DEFINE_integer('padding', 16, '')

FLAGS = tf.app.flags.FLAGS


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
    image = tf.image.resize_images(image, [256, 256])
    image = tf.reshape(image, [1, 256, 256, 3])

    # for VggNet, subtract the mean color of it's training data.
    # image = tf.subtract(image, VggNet.mean_color_rgb())

    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    # R/G/B to B/G/R
    image = tf.reverse(image, [3])

    padding = [FLAGS.padding, FLAGS.padding]

    image = tf.pad(
        tensor=image,
        paddings=[[0, 0], padding, padding, [0, 0]],
        mode='symmetric')

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image, shape = session.run([image, shape])

        coord.request_stop()
        coord.join(threads)

        return image, shape


def build_style():
    """
    """
    image = load_image(FLAGS.style_path)

    return tf.constant(image[0], name='style_image')


def build_contents_reader():
    """
    """
    if FLAGS.train:
        paths_jpg_wildcards = os.path.join(FLAGS.contents_path, '*.jpg')

        paths_images = glob.glob(paths_jpg_wildcards)
    else:
        paths_images = [FLAGS.content_path]

    file_name_queue = tf.train.string_input_producer(paths_images)

    reader = tf.WholeFileReader()

    reader_key, reader_val = reader.read(file_name_queue)

    image = tf.image.decode_jpeg(reader_val, channels=3)

    if FLAGS.train:
        image = tf.random_crop(image, size=[256, 256, 3])

        image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    # R/G/B to B/G/R
    image = tf.reverse(image, [2])

    padding = [FLAGS.padding, FLAGS.padding]

    image = tf.pad(
        tensor=image,
        paddings=[padding, padding, [0, 0]],
        mode='symmetric')

    if FLAGS.train:
        return tf.train.batch(
            tensors=[image],
            batch_size=FLAGS.batch_size,
            capacity=FLAGS.batch_size)
    else:
        return tf.reshape(image, [1, 256, 256, 3])


def build_summaries(network):
    """
    """
    # summary_loss = tf.summary.scalar('transfer loss', network['loss'])

    images_c = network['image_content']
    images_s = network['image_styled']

    images_c = tf.slice(
        images_c,
        [0, FLAGS.padding, FLAGS.padding, 0],
        [-1, 256, 256, -1])

    images_s = tf.slice(
        images_s,
        [0, FLAGS.padding, FLAGS.padding, 0],
        [-1, 256, 256, -1])

    images_c = tf.reshape(images_c, [1, FLAGS.batch_size * 256, 256, 3])
    images_s = tf.reshape(images_s, [1, FLAGS.batch_size * 256, 256, 3])

    images_a = tf.concat([images_c, images_s], axis=2)
    images_a = images_a * 127.5 + 127.5
    # images_a = tf.add(images_a, VggNet.mean_color_bgr())
    images_a = tf.reverse(images_a, [3])
    images_a = tf.saturate_cast(images_a, tf.uint8)

    summary_image = tf.summary.image('all', images_a, max_outputs=4)

    # summary_plus = tf.summary.merge([summary_image, summary_loss])

    return {
        # 'summary_part': summary_loss,
        'summary_plus': summary_image,
    }


def train():
    """
    """
    ckpt_source_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    ckpt_target_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    image_contents = build_contents_reader()
    image_style = build_style()

    network = build_style_transfer_network(
        image_contents,
        image_style,
        FLAGS.vgg16_path,
        FLAGS.content_weight,
        FLAGS.content_level,
        FLAGS.style_weight,
        FLAGS.style_level,
        training=True)

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    summaries = build_summaries(network)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if ckpt_source_path is not None:
            tf.train.Saver().restore(session, ckpt_source_path)

        # give up overlapped old data
        step = session.run(network['step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            fetches = {
                'step': network['step'],
                'train': network['train_transfer'],
            }

            if step % 100 == 0:
                print 'step: {}'.format(step)

            if step % 50 == 0:
                fetches['summary'] = summaries['summary_plus']

            fetched = session.run(fetches)

            step = fetched['step']

            if 'summary' in fetched:
                reporter.add_summary(fetched['summary'], step)

            if step % 10000 == 0:
                tf.train.Saver().save(
                    session, ckpt_target_path, global_step=network['step'])

        coord.request_stop()
        coord.join(threads)


def transfer():
    """
    """


def main(_):
    """
    """
    # train() if FLAGS.train else transfer()
    train()


if __name__ == '__main__':
    tf.app.run()
