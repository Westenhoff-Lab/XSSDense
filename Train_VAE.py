import tensorflow as tf
import numpy as np
import sys
import os

"""
Beta-VAE with beta scheduling, beta value can be increased over epochs to enforce good reconstruction in early epochs
and stronger regularised latent space in later epochs. Scheduling is linear over epochs.  
"""

# =========================================================
# GPU memory: enable growth (prevents full VRAM pre-allocation)
# =========================================================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)

# =========================================================
# Config
# =========================================================
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

training_data = sys.argv[1]  # Path to TFRecord file
test_data_path = sys.argv[2]  # Path to TFRecord file
save_name = sys.argv[3]
mode = sys.argv[4]
beta_input = float(sys.argv[5])

input_shape = (52,52,52, 1)
z_dim = 8 #8
learning_rate = 3e-4
epochs = 80
batchsize = 64

# A scalar beta variable read inside @tf.function (prevents retracing when beta changes)
beta_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)

# =========================================================
# TFRecord Parser
# =========================================================
def parse_tfrecord(example_proto):
    features = {'data': tf.io.FixedLenFeature([52*52*52], tf.float32)}
    parsed = tf.io.parse_single_example(example_proto, features)
    data = tf.reshape(parsed['data'], input_shape)
    return data, data

# =========================================================
# Datasets
# =========================================================
def load_dataset(path, batchsize, seed):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    # Reduce RAM and keep batch shapes static
    buffer = min(3200, batchsize * 16)  # e.g., 1024 for batch=64
    ds = ds.shuffle(buffer_size=buffer, seed=seed)
    ds = ds.batch(batchsize, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    return ds

train_ds = load_dataset(training_data, batchsize, seed)
test_ds = load_dataset(test_data_path, batchsize, seed)

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
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
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

# =========================================================
# Encoder & Decoder
# =========================================================
def build_encoder():
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

def build_decoder():
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(13 * 13 * 13 * 32, activation='relu', kernel_initializer='he_normal')(latent_inputs)
    x = tf.keras.layers.Reshape((13, 13, 13, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    outputs = tf.keras.layers.Conv3D(1, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# =========================================================
# Loss Functions
# =========================================================
def compute_KL_prior_latent(latent_mean, latent_std, epsilon_loss=1e-8):
    latent_std = tf.maximum(latent_std, epsilon_loss)
    kl_div = 0.5 * tf.reduce_sum(
        tf.square(latent_mean) + tf.square(latent_std) - tf.math.log(tf.square(latent_std)) - 1,
        axis=1,
    )
    return tf.reduce_mean(kl_div)

def compute_loss(model, x, beta=0.0):
    mean, logvar = model.encode(x)
    latent_std = tf.exp(0.5 * logvar)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)
    #reconstruction_loss = tf.reduce_mean(tf.square(x - x_pred), axis=[1, 2, 3, 4])
    reconstruction_loss = tf.reduce_sum(tf.square(x - x_pred), axis=[1, 2, 3, 4])

    beta_t = tf.cast(beta, tf.float32)  # ensure tensor scalar
    kl_loss = compute_KL_prior_latent(mean, latent_std) * beta_t
    total_loss = reconstruction_loss + kl_loss

    return tf.reduce_mean(total_loss), tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss)

# Single train step that reads beta_var inside the graph (prevents retracing)
@tf.function
def train_step_any(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, mse, KL = compute_loss(model, x, beta_var)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, mse, KL

# =========================================================
# Build and Train
# =========================================================
encoder = build_encoder()
decoder = build_decoder()
vae = VAE(z_dim, encoder, decoder)
vae.build((None, 52,52,52, 1))

optimizer = tf.keras.optimizers.Adam(learning_rate)

num_train_samples = sum(1 for _ in tf.data.TFRecordDataset(training_data))
steps_per_epoch = num_train_samples // batchsize
total_steps = steps_per_epoch * epochs  # total number of iterations (for annealing)

os.makedirs(f'{save_name}_log', exist_ok=True)
log_path = f"{save_name}_log/log.txt"

global_step = 0  # track total iterations for annealing

with open(log_path, 'a') as log_file:
    val_fraction = 0.1

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ─────────────────────────────
        # Shuffle dataset *fresh each epoch*
        # ─────────────────────────────
        ds_epoch = tf.data.TFRecordDataset(training_data)
        ds_epoch = ds_epoch.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        ds_epoch = ds_epoch.shuffle(buffer_size=5000, seed=seed + epoch)

        # Compute number of elements
        num_samples = num_train_samples
        val_size = int(val_fraction * num_samples)

        # Split
        val_ds = ds_epoch.take(val_size)
        train_ds_epoch = ds_epoch.skip(val_size)

        # Batch them
        train_ds_epoch = train_ds_epoch.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        epoch_losses, epoch_mses, epoch_kls  = [], [], []

        # ─────────────────────────────
        # TRAINING
        # ─────────────────────────────
        for step, (x_batch, _) in enumerate(train_ds_epoch):

            if mode == "constant":
                beta = beta_input
            elif mode == "late":
                progress = min(global_step / total_steps, 1.0)
                beta = beta_input * progress
            else:
                raise ValueError(f"Unknown mode: {mode}")

            beta_var.assign(beta)

            loss, mse, KL  = train_step_any(vae, x_batch, optimizer)

            loss_f = float(loss.numpy())
            mse_f = float(mse.numpy())
            kl_f = float(KL.numpy())
            

            epoch_losses.append(loss_f)
            epoch_mses.append(mse_f)
            epoch_kls.append(kl_f)
           

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss_f:.5f}, MSE: {mse_f:.5f}, KL: {kl_f:.5f}")

            global_step += 1

        # ─────────────────────────────
        # VALIDATION
        # ─────────────────────────────
        val_loss, val_mse, val_kl = 0.0, 0.0, 0.0
        val_steps = 0

        for x_val, _ in val_ds:
            l, m, k = compute_loss(vae, x_val, beta_var)

            val_loss += float(l.numpy())
            val_mse += float(m.numpy())
            val_kl += float(k.numpy())
       
            val_steps += 1

        val_loss /= val_steps
        val_mse /= val_steps
        val_kl /= val_steps
        

        # ─────────────────────────────
        # LOGGING
        # ─────────────────────────────
        print(
            f"Epoch {epoch + 1} done | "
            f"TrainLoss: {np.mean(epoch_losses):.5f}, MSE: {mse_f:.5f}, KL: {kl_f:.5f}, Beta: {float(beta_var.numpy()):.5f} | "
            f"ValLoss: {val_loss:.5f} | Beta: {float(beta_var.numpy()):.5f}"
        )

        log_file.write(
            f"Epoch {epoch + 1} | "
            f"TrainLoss: {np.mean(epoch_losses):.5f}, MSE: {mse_f:.5f}, KL: {kl_f:.5f}, Beta: {float(beta_var.numpy()):.5f} | "
            f"ValLoss: {val_loss:.5f}, ValMSE: {val_mse:.5f}, "
            f"ValKL: {val_kl:.5f}, Beta: {float(beta_var.numpy()):.5f},\n"
        )

        log_file.flush()
        os.fsync(log_file.fileno())

        vae.save_weights(f"{save_name}_log/vae_epoch_{epoch + 1}.weights.h5")


print("\nFinal test evaluation...")

test_loss, test_mse, test_kl = 0.0, 0.0, 0.0
test_steps = 20

for x_test, _ in test_ds.take(test_steps):
    l, m, k = compute_loss(vae, x_test, beta_var)

    test_loss += float(l.numpy())
    test_mse += float(m.numpy())
    test_kl += float(k.numpy())


test_loss /= test_steps
test_mse /= test_steps
test_kl /= test_steps


print(f"Final test: Loss={test_loss:.5f}, MSE={test_mse:.5f}, KL={test_kl:.5f} ")

# =========================================================
# Final Model Save
# =========================================================
vae.encoder.save(f'{save_name}_log/encoder_model.keras')
vae.decoder.save(f'{save_name}_log/decoder_model.keras')
vae.save_weights(f'{save_name}_log/vae_final.weights.h5')
print(f"Models saved in {save_name}_log")

