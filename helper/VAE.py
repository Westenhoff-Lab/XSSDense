"""
Shared model definitions and data loading (train / validation / test) for the beta-VAE
"""

import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_dilation

AUTOTUNE = tf.data.AUTOTUNE
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


def reconstruct(vae, x):
    """Encode to the latent mean and decode again (deterministic)."""
    mean, _ = vae.encode(x)
    return vae.decode(mean)


# =========================================================
# Data: train / validation / test
# =========================================================
def parse_tfrecord(example_proto):
    features = {"data": tf.io.FixedLenFeature(GRID_SIZE**3, tf.float32)}
    parsed = tf.io.parse_single_example(example_proto, features)
    return tf.reshape(parsed["data"], INPUT_SHAPE)


def count_records(path):
    return sum(1 for _ in tf.data.TFRecordDataset(path))


def split_train_val(num_samples, block=100, val_every=10):
    """Masks for the records of train.tfrecords: (is_train, is_val).
    Every `val_every`-th block of `block` records is validation, the rest is training.
    """
    pos = np.arange(num_samples)
    is_val = (pos // block) % val_every == 0
    is_train = ~is_val
    print(f"split: {is_train.sum()} train, {is_val.sum()} validation")
    return is_train, is_val


def _select(path, mask):
    """Records of `path` where `mask` is True, parsed."""
    mask = tf.constant(mask)

    def keep(i, rec):
        return mask[i]

    def drop_index(i, rec):
        return rec

    return tf.data.TFRecordDataset(path).enumerate().filter(keep).map(drop_index).map(parse_tfrecord, num_parallel_calls=AUTOTUNE)


def make_train_dataset(path, is_train, batchsize, seed, shuffle_buffer=5000):
    """Training data for one epoch: shuffled with `seed`, batched."""
    return _select(path, is_train).shuffle(shuffle_buffer, seed=seed).batch(batchsize, drop_remainder=True).prefetch(AUTOTUNE)


def make_val_dataset(path, is_val, batchsize):
    """Validation data: fixed, unshuffled, cached after the first pass."""
    return _select(path, is_val).batch(batchsize).cache().prefetch(AUTOTUNE)


def make_test_dataset(path, batchsize):
    """Test data (its own file): unshuffled, read once."""
    return tf.data.TFRecordDataset(path).map(parse_tfrecord, num_parallel_calls=AUTOTUNE).batch(batchsize).prefetch(AUTOTUNE)
