import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

tf.random.set_seed(42)

x_all = np.load("embeddings_barlow_train.npy")
for i in range(len(x_all)):
    x_all[i] = (x_all[i] - np.min(x_all[i]))/(np.max(x_all[i]) - np.min(x_all[i]))
    
x_all = x_all.astype("float32")
input_dim = x_all.shape[1]

latent_dim = 64

encoder_input = layers.Input(shape=(input_dim,))
x = layers.Dense(256, activation="relu")(encoder_input)
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
x = layers.Dense(128, activation="relu")(decoder_input)
x = layers.Dense(256, activation="relu")(x)
decoder_output = layers.Dense(input_dim, activation="sigmoid")(x)
decoder = models.Model(decoder_input, decoder_output, name="decoder")

outputs = decoder(z)
vae = models.Model(encoder_input, outputs, name="vae")

reconstruction_loss = tf.keras.losses.binary_crossentropy(
    K.flatten(encoder_input), K.flatten(outputs)
)
reconstruction_loss *= input_dim
kl_loss = -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer="adam")

vae.fit(x_all, x_all, epochs=20, batch_size=32, validation_split=0.1)

y_all = np.load("labels_train.npy")
x_test = np.load("embeddings_barlow_test.npy")
for i in range(len(x_test)):
    x_test[i] = (x_test[i] - np.min(x_test[i]))/(np.max(x_test[i]) - np.min(x_test[i]))
    
z_mean_train, _, _ = encoder.predict(x_all, batch_size=256)
z_mean_test, _, _ = encoder.predict(x_test, batch_size=256)

y_train = np.load("labels_train.npy")
y_test = np.load("labels_test.npy")

x_data = np.concatenate([z_mean_train, z_mean_test], axis=0)
y_data = np.concatenate([y_all,y_test],axis=0)
np.save("x_data.npy", x_data)
np.save("y_data.npy", y_data)


