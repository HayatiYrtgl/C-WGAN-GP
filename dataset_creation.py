import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_the_image(img_file):
    img = tf.io.read_file(img_file)
    img = tf.io.decode_jpeg(img)

    w = tf.shape(img)[1]
    width = w // 2
    scratch = img[:, :width, :]
    transformed = img[:, width:, :]

    """plt.subplot(1, 2, 1)
    plt.imshow(scratch)

    plt.subplot(1, 2, 2)
    plt.imshow(transformed)

    plt.show()"""

    if tf.random.uniform(()) > 0.5:
        scratch = tf.image.flip_left_right(scratch)
        transformed = tf.image.flip_left_right(transformed)

    scratch = (tf.cast(scratch, tf.float32) / 127.5) - 1
    transformed = (tf.cast(transformed, tf.float32) / 127.5) - 1

    return scratch, transformed


dataset = tf.data.Dataset.list_files("../../DATASETS/portraits_dataset/*.jpg")
train_data = dataset.map(preprocess_the_image, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1024).batch(1)

