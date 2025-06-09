from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class GAN:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(8*8*128, input_dim=100))
        model.add(layers.Reshape((8, 8, 128)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh'))
        return model




    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Dense(1, activation='sigmoid'))
        return model


    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model = models.Sequential([self.generator, self.discriminator])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    
    def generate(self, num_images):
        noise = np.random.normal(0, 1, (num_images, 100))
        generated_images = self.generator.predict(noise)
        # Rescale from [-1, 1] to [0, 1] for displaying
        generated_images = 0.5 * generated_images + 0.5
        return generated_images


    def train(self, X_train, epochs, batch_size=128):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_images = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            fake_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 100 == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

            if epoch % 1000 == 0:
                samples = self.generate(9)
                plt.figure(figsize=(6, 6))
                for i in range(9):
                    plt.subplot(3, 3, i + 1)
                    plt.imshow((samples[i] + 1) / 2)  # Rescale from [-1,1] to [0,1]
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

