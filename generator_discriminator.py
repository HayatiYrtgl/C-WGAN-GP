from keras.layers import *
import tensorflow as tf
from keras.models import *


# encoder
def encode(filters, kernel_size, bath_norm=True, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    # encoder section
    encoder = Sequential()
    encoder.add(Conv2D(filters=filters, kernel_size=kernel_size, padding="same", strides=strides,
                       use_bias=False, kernel_initializer=initializer))

    if bath_norm:
        encoder.add(BatchNormalization())

    encoder.add(LeakyReLU(0.25))
    return encoder


# decoder
def decode(filters, kernel_size, dropout=False, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    decoder = Sequential()

    decoder.add(Conv2DTranspose(filters=filters, kernel_initializer=initializer,
                                kernel_size=kernel_size, strides=strides, use_bias=False, padding="same"))

    decoder.add(BatchNormalization())

    if dropout:
        decoder.add(Dropout(0.5))

    decoder.add(ReLU())

    return decoder


# generator
def generator():
    inputs = Input(shape=(256, 256, 3))

    # downsample
    downsample = [
        encode(64, 4, bath_norm=False),
        encode(128, 4, False),
        encode(256, 4),
        encode(256, 4, strides=1),
        encode(512, 4),
        encode(512, 4),
        encode(512, 4),
        encode(512, 4),
        encode(512, 4),

    ]

    upsample = [
        decode(512, 4, dropout=True),
        decode(512, 4, dropout=True),
        decode(512, 4, dropout=True),
        decode(512, 4),
        decode(512, 4),
        decode(256, 4, strides=1),
        decode(256, 4),
        decode(128, 4),
        decode(64, 4),
    ]

    output_channel = 3
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channel, kernel_size=4, padding="same", kernel_initializer=initializer, strides=2,
                           activation="tanh",
                           use_bias=False)

    x = inputs
    skips = []

    for d in downsample:
        x = d(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(upsample, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)
    return Model(inputs, x, name="Generator")


def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    scratch = Input(shape=(256, 256, 3), name="scratch")
    transformed = Input(shape=(256, 256, 3), name="transfromed")

    x = concatenate([scratch, transformed])

    d1 = encode(256, 4, bath_norm=False)(x)
    d2 = encode(512, 4)(d1)
    d3 = encode(512, 4)(d2)
    zeropad1 = ZeroPadding2D()(d3)
    conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                  use_bias=False)(zeropad1)
    bn = BatchNormalization()(conv)
    leaky_relu = LeakyReLU(0.25)(bn)

    zeropad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, 1, kernel_initializer=initializer)(zeropad2)
    return  Model([scratch, transformed], last, name="discriminator")
