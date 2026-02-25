import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, backend as K

tf.random.set_seed(42)

ds_train, ds_test = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    batch_size=-1,
    as_supervised=True,
)

x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

x_train = tf.image.rot90(x_train, k=3).numpy()
x_test = tf.image.rot90(x_test, k=3).numpy()

latent_dim = 16

encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_input)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = models.Model(decoder_input, decoder_output, name="decoder")

outputs = decoder(z)
vae = models.Model(encoder_input, outputs, name="vae")

reconstruction_loss = tf.keras.losses.binary_crossentropy(
    K.flatten(encoder_input), K.flatten(outputs)
)
reconstruction_loss *= 28 * 28
kl_loss = -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer="adam")

vae.fit(x_train, x_train, epochs=30, batch_size=256, validation_split=0.1)

z_mean_train, _, _ = encoder.predict(x_train, batch_size=256)
z_mean_test, _, _ = encoder.predict(x_test, batch_size=256)

x_data = np.concatenate([z_mean_train, z_mean_test], axis=0)
y_data = np.concatenate([y_train, y_test], axis=0)
np.save("x_data.npy", x_data)
np.save("y_data.npy", y_data)