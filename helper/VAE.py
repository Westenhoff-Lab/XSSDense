"""
Shared model definitions for the beta-VAE on voxelized electron densities.

Imported by the training script (to build the model) and by the analysis and
reconstruction scripts (to load the saved encoder/decoder). Importing this module
registers ElectronConservation with Keras, so `load_model` finds it.
"""

import tensorflow as tf

GRID_SIZE = 52
LATENT_GRID = GRID_SIZE // 4  # spatial size after the encoder's two 2x poolings
INPUT_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1)


@tf.keras.utils.register_keras_serializable()
class ElectronConservation(tf.keras.layers.Layer):
    """Rescales each density grid so that its voxel sum equals `target_sum`."""

    def __init__(self, target_sum, eps=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.target_sum = float(target_sum)
        self.eps = eps

    def call(self, rho):
        total = tf.reduce_sum(rho, axis=[1, 2, 3, 4], keepdims=True)
        return rho * (self.target_sum / (total + self.eps))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(target_sum=self.target_sum, eps=self.eps)
        return cfg


def build_encoder(z_dim):
    conv = dict(padding="same", activation="relu", kernel_initializer="he_normal")
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv3D(64, 3, **conv)(inputs)
    x = tf.keras.layers.Conv3D(64, 3, **conv)(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
    x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
    x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_dim * 2, activation=None, dtype="float32")(x)  # mean and logvar
    return tf.keras.Model(inputs, z, name="encoder")


def build_decoder(z_dim, target_sum):
    conv = dict(padding="same", activation="relu", kernel_initializer="he_normal")
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(LATENT_GRID**3 * 32, activation="relu", kernel_initializer="he_normal")(latent_inputs)
    x = tf.keras.layers.Reshape((LATENT_GRID, LATENT_GRID, LATENT_GRID, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, **conv)(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, **conv)(x)
    rho = tf.keras.layers.Conv3D(1, 3, padding="same", activation="softplus", dtype="float32")(x)
    outputs = ElectronConservation(target_sum, dtype="float32")(rho)
    return tf.keras.Model(latent_inputs, outputs, name="decoder")


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, encoder, decoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def get_config(self):
        return {"latent_dim": self.latent_dim}

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(0.5 * logvar) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)


def build_vae(z_dim, target_sum):
    """New, untrained VAE (for training)."""
    vae = VAE(z_dim, build_encoder(z_dim), build_decoder(z_dim, target_sum))
    vae(tf.zeros((1, *INPUT_SHAPE)))  # create all variables
    return vae


def load_vae(model_dir):
    """Trained VAE from the encoder_model.keras / decoder_model.keras in `model_dir`."""
    encoder = tf.keras.models.load_model(f"{model_dir}/encoder_model.keras")
    decoder = tf.keras.models.load_model(f"{model_dir}/decoder_model.keras", custom_objects={"ElectronConservation": ElectronConservation})
    latent_dim = encoder.output_shape[-1] // 2
    return VAE(latent_dim, encoder, decoder)
