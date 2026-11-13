
import tensorflow as tf
import numpy as np
import sys
import os

# =========================================================
# Config
# =========================================================
input_shape = (52, 52, 52, 1)

# =========================================================
# VAE model
# =========================================================
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, encoder, decoder):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        out = self.encoder(x)
        # If encoder is a TFSMLayer, it returns a dict
        if isinstance(out, dict):
            out = list(out.values())[0]  # grab the first tensor
        mean, logvar = tf.split(out, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(0.5 * logvar) + mean

    def decode(self, z, batch_size=32):
        outputs = []
        for i in range(0, len(z), batch_size):
            batch = z[i:i + batch_size]
            out = self.decoder(batch)
            outputs.append(out)
        return tf.concat(outputs, axis=0)


# =========================================================
# Encoder & Decoder
# =========================================================
def build_encoder(z_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_dim * 2, activation=None)(x)
    return tf.keras.Model(inputs, z, name='encoder')

def build_decoder(z_dim):
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(8 * 8 * 8 * 32, activation='relu', kernel_initializer='he_normal')(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    outputs = tf.keras.layers.Conv3D(1, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# =========================================================
# Load model and weights
# =========================================================
def load_vae(weights_path, z_dim):
    """
    Build VAE model and load pre-trained weights.
    Returns a VAE instance ready for inference.
    """
    encoder = build_encoder(z_dim)
    decoder = build_decoder(z_dim)
    vae = VAE(z_dim, encoder, decoder)


    # Now all layers have been “built” and weights can be loaded
    vae.load_weights(weights_path)
    print(f"VAE weights loaded from {weights_path}")
    return vae
