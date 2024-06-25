from keras.models import Model
from keras.metrics import Mean
import tensorflow as tf
from dataset_creation import train_data
from losses import generator_loss, discriminator_loss, optimizer_disc, optimizer_gen
from generator_discriminator import generator, discriminator
from keras.callbacks import Callback
from generate_image import generate_image

tf.config.run_functions_eagerly(True)


class CWGAN_GP(Model):
    def __init__(self, generator, discriminator):
        super(CWGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

        self.scratch = None
        self.transformed = None
        self.gp_weight = 10
        self.d_extra_steps = 5
        self.batch_size = 1

    def compile(self, gen_opt, disc_opt, disc_loss, gen_loss):
        super().compile()
        self.g_opt = gen_opt
        self.d_opt = disc_opt
        self.d_loss = disc_loss
        self.g_loss = gen_loss

        self.g_metric = Mean(name="Generator Total Loss")
        self.d_metric = Mean(name="Discriminator_gp Loss")
        self.l1_metric = Mean(name="L1_Loss")

    # gradient penalty
    def gradient_penalty(self, real_images, fake_images, scratch):
        # alpha value
        batch_size = self.batch_size

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images

        interpolation = real_images + alpha * diff

        # gradient calculation
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolation)

            # pred
            prediction = self.discriminator([scratch, interpolation], training=True)

        # calculate
        grads = gp_tape.gradient(prediction, [interpolation])[0]

        # norm
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads), axis=[1, 2, 3]))

        # apply
        gradient_penalty = tf.reduce_mean((norm - 1) ** 2)

        return gradient_penalty

    def train_step(self, data):
        scratch, transformed = data
        self.scratch, self.transformed = scratch, transformed

        for i in range(self.d_extra_steps):
            with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                generated = self.generator(scratch, training=True)

                disc_real = self.discriminator([scratch, transformed], training=True)
                disc_fake = self.discriminator([scratch, generated], training=True)

                gen_loss, l1_loss = self.g_loss(disc_fake, generated, transformed)

                d_cost = self.d_loss(disc_fake, disc_real)

                gp = self.gradient_penalty(real_images=transformed, fake_images=generated, scratch=scratch)

                d_loss = d_cost + gp * self.gp_weight

            grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            grad_gen = g_tape.gradient(gen_loss, self.generator.trainable_weights)
            self.g_opt.apply_gradients(zip(grad_gen, self.generator.trainable_weights))

            self.g_metric.update_state(gen_loss)
            self.l1_metric.update_state(l1_loss)
            self.d_metric.update_state(d_loss)

            return {"Disc loss: ": self.d_metric.result(),
                    "Gen Loss: ": self.g_metric.result(),
                    "L1_LOSS: ": self.l1_metric.result()}


gen = generator()
disc = discriminator()
gen_loss = generator_loss
disc_loss = discriminator_loss
g_opt = optimizer_gen
d_opt = optimizer_disc


class Monitor(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            self.model.generator.save("../completed_models/generator_edges_to.h5")
            self.model.discriminator.save("../completed_models/discriminator_edges_to.h5")
            print("Models Saved")

        generate_image(model=self.model.generator, step=epoch, original=self.model.scratch,
                       transformed=self.model.transformed)


c = CWGAN_GP(generator=gen, discriminator=disc)
c.compile(gen_loss=gen_loss, disc_loss=disc_loss, gen_opt=g_opt, disc_opt=d_opt)
c.fit(train_data, epochs=300, callbacks=[Monitor()])
