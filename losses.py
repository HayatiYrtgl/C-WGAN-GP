import tensorflow as tf
from keras.optimizers import Adam


def generator_loss(fake_output, g_output, target):
    gan_loss = -tf.reduce_mean(fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target - g_output))
    total_loss = gan_loss + (l1_loss*100)
    return total_loss, l1_loss


def discriminator_loss(fake, real):
    real_loss = tf.reduce_mean(real)
    fake_loss = tf.reduce_mean(fake)

    return fake_loss - real_loss


optimizer_gen = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
optimizer_disc = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
