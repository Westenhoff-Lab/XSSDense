"""
Train a beta-VAE on voxelized electron densities stored as TFRecords.
"""

import os
from helper import redirect_tf_cpp_logs

redirect_tf_cpp_logs.redirect()
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow as tf
from helper.VAE import GRID_SIZE, INPUT_SHAPE, build_vae

# =========================================================
# Constants
# =========================================================
SEED = 42
GRID_SIZE = 52
LATENT_GRID = GRID_SIZE // 4  # spatial size after the encoder's two 2x poolings
INPUT_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1)
AUTOTUNE = tf.data.AUTOTUNE
WARMUP_FRACTION = 0.1  # fraction of training steps for the beta warm-up ("late" mode)


# =========================================================
# Setup
# =========================================================
def configure_gpu(mixed_precision=True):
    """Enable memory growth on all GPUs and optionally the mixed_float16 policy."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: No GPU detected by TensorFlow. Running on CPU.")
    else:
        print(f"GPU DETECTED: TensorFlow found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"  Could not set memory growth: {e}")
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


# =========================================================
# Config
# =========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a beta-VAE on voxelized electron densities.")
    parser.add_argument("--datapath", help="Converter output directory (expects tfrecords/train.tfrecords, tfrecords/test.tfrecords, meta/meta.json)")
    parser.add_argument("--train_datapath", help="Path to training TFRecord file")
    parser.add_argument("--test_datapath", help="Path to test TFRecord file")
    parser.add_argument("--meta_datapath", help="Path to meta.json")
    parser.add_argument("--savepath", default=".", help="Directory for run outputs (default: current directory)")
    parser.add_argument("--mode", required=True, choices=["constant", "late"], help="Beta schedule")
    parser.add_argument("--beta_input", type=float, required=True, help="Beta value")
    parser.add_argument("--z_dim", type=int, default=8, help="Latent dimension (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial Adam learning rate (default: 3e-4)")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--batchsize", type=int, default=64, help="Batch size (default: 64)")
    args = parser.parse_args()

    explicit = [args.train_datapath, args.test_datapath, args.meta_datapath]
    usage_error = "use either --datapath or all three of --train_datapath, --test_datapath and --meta_datapath"
    if args.datapath:
        if any(explicit):
            parser.error(usage_error)
        args.train_datapath = os.path.join(args.datapath, "tfrecords", "train.tfrecords")
        args.test_datapath = os.path.join(args.datapath, "tfrecords", "test.tfrecords")
        args.meta_datapath = os.path.join(args.datapath, "meta", "meta.json")
    elif not all(explicit):
        parser.error(usage_error)
    return args


def load_meta(meta_path):
    with open(meta_path) as f:
        return json.load(f)


# =========================================================
# Data
# =========================================================
def parse_tfrecord(example_proto):
    features = {"data": tf.io.FixedLenFeature(GRID_SIZE**3, tf.float32)}
    parsed = tf.io.parse_single_example(example_proto, features)
    return tf.reshape(parsed["data"], INPUT_SHAPE)


def count_records(path):
    return sum(1 for _ in tf.data.TFRecordDataset(path))


def make_train_val_datasets(path, num_samples, batchsize, val_fraction=0.1):
    """Fixed split: the first `val_fraction` of the file is validation, the rest training.

    Returns the unbatched training dataset (shuffled per epoch by make_epoch_dataset),
    the batched and cached validation dataset, and the number of training samples.
    """
    ds = tf.data.TFRecordDataset(path).map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    val_size = int(val_fraction * num_samples)
    val_ds = ds.take(val_size).batch(batchsize).cache().prefetch(AUTOTUNE)
    train_base = ds.skip(val_size)
    return train_base, val_ds, num_samples - val_size


def make_epoch_dataset(train_base, batchsize, seed, shuffle_buffer=5000):
    return train_base.shuffle(shuffle_buffer, seed=seed).batch(batchsize, drop_remainder=True).prefetch(AUTOTUNE)


def make_test_dataset(path, batchsize):
    return tf.data.TFRecordDataset(path).map(parse_tfrecord, num_parallel_calls=AUTOTUNE).batch(batchsize).prefetch(AUTOTUNE)


# # =========================================================
# # Model
# # =========================================================
# @tf.keras.utils.register_keras_serializable()
# class ElectronConservation(tf.keras.layers.Layer):
#     """Rescales each density grid so that its voxel sum equals `target_sum`."""

#     def __init__(self, target_sum, eps=1e-7, **kwargs):
#         super().__init__(**kwargs)
#         self.target_sum = float(target_sum)
#         self.eps = eps

#     def call(self, rho):
#         total = tf.reduce_sum(rho, axis=[1, 2, 3, 4], keepdims=True)
#         return rho * (self.target_sum / (total + self.eps))

#     def get_config(self):
#         cfg = super().get_config()
#         cfg.update(target_sum=self.target_sum, eps=self.eps)
#         return cfg


# def build_encoder(z_dim):
#     conv = dict(padding="same", activation="relu", kernel_initializer="he_normal")
#     inputs = tf.keras.Input(shape=INPUT_SHAPE)
#     x = tf.keras.layers.Conv3D(64, 3, **conv)(inputs)
#     x = tf.keras.layers.Conv3D(64, 3, **conv)(x)
#     x = tf.keras.layers.MaxPool3D(2)(x)
#     x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
#     x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
#     x = tf.keras.layers.MaxPool3D(2)(x)
#     x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
#     x = tf.keras.layers.Conv3D(128, 3, **conv)(x)
#     x = tf.keras.layers.Flatten()(x)
#     z = tf.keras.layers.Dense(z_dim * 2, activation=None, dtype="float32")(x)  # mean and logvar
#     return tf.keras.Model(inputs, z, name="encoder")


# def build_decoder(z_dim, target_sum):
#     conv = dict(padding="same", activation="relu", kernel_initializer="he_normal")
#     latent_inputs = tf.keras.Input(shape=(z_dim,))
#     x = tf.keras.layers.Dense(LATENT_GRID**3 * 32, activation="relu", kernel_initializer="he_normal")(latent_inputs)
#     x = tf.keras.layers.Reshape((LATENT_GRID, LATENT_GRID, LATENT_GRID, 32))(x)
#     x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, **conv)(x)
#     x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, **conv)(x)
#     rho = tf.keras.layers.Conv3D(1, 3, padding="same", activation="softplus", dtype="float32")(x)
#     outputs = ElectronConservation(target_sum, dtype="float32")(rho)
#     return tf.keras.Model(latent_inputs, outputs, name="decoder")


# class VAE(tf.keras.Model):
#     def __init__(self, latent_dim, encoder, decoder):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.encoder = encoder
#         self.decoder = decoder

#     def get_config(self):
#         return {"latent_dim": self.latent_dim}

#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         logvar = tf.clip_by_value(logvar, -10.0, 10.0)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=tf.shape(mean))
#         return eps * tf.exp(0.5 * logvar) + mean

#     def decode(self, z):
#         return self.decoder(z)

#     def call(self, x):
#         mean, logvar = self.encode(x)
#         z = self.reparameterize(mean, logvar)
#         return self.decode(z)


# def build_vae(z_dim, target_sum):
#     vae = VAE(z_dim, build_encoder(z_dim), build_decoder(z_dim, target_sum))
#     vae(tf.zeros((1, *INPUT_SHAPE)))  # create all variables
#     return vae


# =========================================================
# Losses
# =========================================================
def kl_divergence(mean, std, eps=1e-8):
    """KL(q(z|x) || N(0, I)), summed over latent dims, averaged over the batch."""
    std = tf.maximum(std, eps)
    kl = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(std) - tf.math.log(tf.square(std)) - 1, axis=1)
    return tf.reduce_mean(kl)


def compute_loss(model, x, beta):
    """Returns (total loss, reconstruction SSE, raw unweighted KL), all batch means."""
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)
    reconstruction = tf.reduce_sum(tf.square(x - x_pred), axis=[1, 2, 3, 4])
    kl_raw = kl_divergence(mean, tf.exp(0.5 * logvar))
    total = reconstruction + tf.cast(beta, tf.float32) * kl_raw
    return tf.reduce_mean(total), tf.reduce_mean(reconstruction), kl_raw


# =========================================================
# Training
# =========================================================
def beta_schedule(mode, beta_input, step, total_steps):
    if mode == "constant":
        return beta_input
    if mode == "late":
        return beta_input * min(step / (WARMUP_FRACTION * total_steps), 1.0)
    raise ValueError(f"Unknown mode: {mode}")


def make_optimizer(learning_rate, total_steps, warmup_fraction=0.05):
    warmup_steps = int(warmup_fraction * total_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup_steps,
        alpha=0.05,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )
    adam = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0, jit_compile=False)
    return tf.keras.mixed_precision.LossScaleOptimizer(adam)


# def make_optimizer(learning_rate, total_steps):
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps=total_steps, alpha=0.05)
# adam = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0, jit_compile=False)
# return tf.keras.mixed_precision.LossScaleOptimizer(adam)


def make_train_step(model, optimizer, beta_var):
    """Compiled train step. Reads beta from `beta_var`, so changing beta doesn't retrace."""

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss, mse, kl = compute_loss(model, x, beta_var)
            scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse, kl

    return train_step


def evaluate(model, ds, beta):
    """Average (loss, mse, kl) over all batches of `ds`."""
    loss = mse = kl = 0.0
    steps = 0
    for x in ds:
        l, m, k = compute_loss(model, x, beta)
        loss += float(l.numpy())
        mse += float(m.numpy())
        kl += float(k.numpy())
        steps += 1
    return loss / steps, mse / steps, kl / steps


def train(vae, train_base, val_ds, steps_per_epoch, args, log_dir):
    """Runs the epoch loop. Returns the beta variable (holding the final beta)."""
    total_steps = steps_per_epoch * args.epochs
    beta_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    optimizer = make_optimizer(args.learning_rate, total_steps)
    train_step = make_train_step(vae, optimizer, beta_var)

    global_step = 0
    with open(os.path.join(log_dir, "log.txt"), "w") as log_file:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_ds_epoch = make_epoch_dataset(train_base, args.batchsize, seed=SEED + epoch)

            epoch_losses, epoch_mses, epoch_kls = [], [], []
            for step, x_batch in enumerate(train_ds_epoch):
                beta_var.assign(beta_schedule(args.mode, args.beta_input, global_step, total_steps))
                loss, mse, kl = train_step(x_batch)

                loss_f, mse_f, kl_f = float(loss.numpy()), float(mse.numpy()), float(kl.numpy())
                epoch_losses.append(loss_f)
                epoch_mses.append(mse_f)
                epoch_kls.append(kl_f)

                if step % 10 == 0:
                    print(f"Step {step}, Loss: {loss_f:.5f}, MSE: {mse_f:.5f}, KL: {kl_f:.5f}")
                global_step += 1

            val_loss, val_mse, val_kl = evaluate(vae, val_ds, beta_var)

            log_message = (
                f"Epoch {epoch + 1} | "
                f"TrainLoss: {np.mean(epoch_losses):.5f}, MSE: {np.mean(epoch_mses):.5f}, KL: {np.mean(epoch_kls):.5f} | "
                f"ValLoss: {val_loss:.5f}, ValMSE: {val_mse:.5f}, ValKL: {val_kl:.5f} | "
                f"Beta: {float(beta_var.numpy()):.5f}"
            )
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())

            vae.save_weights(os.path.join(log_dir, f"vae_epoch_{epoch + 1}.weights.h5"))

    return beta_var


def save_models(vae, log_dir):
    vae.encoder.save(os.path.join(log_dir, "encoder_model.keras"))
    vae.decoder.save(os.path.join(log_dir, "decoder_model.keras"))
    vae.save_weights(os.path.join(log_dir, "vae_final.weights.h5"))
    print(f"Models saved in {log_dir}")


def make_log_dir(args):
    """<savepath>/X.vae, where X is the --datapath directory name or the training file name."""
    src = Path(args.datapath) if args.datapath else Path(args.train_datapath)
    name = src.resolve().with_suffix(".vae").name
    log_dir = Path(args.savepath) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


# =========================================================
# Main
# =========================================================
def main():
    configure_gpu(mixed_precision=True)
    set_seeds(SEED)

    args = parse_arguments()
    meta = load_meta(args.meta_datapath)
    target_sum = meta["electron_conservation"]["target_sum_normalized"]

    vae = build_vae(args.z_dim, target_sum)

    num_samples = count_records(args.train_datapath)
    train_base, val_ds, num_train = make_train_val_datasets(args.train_datapath, num_samples, args.batchsize)
    steps_per_epoch = num_train // args.batchsize

    log_dir = make_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    beta_var = train(vae, train_base, val_ds, steps_per_epoch, args, log_dir)

    print("\nFinal test evaluation...")
    test_ds = make_test_dataset(args.test_datapath, args.batchsize)
    test_loss, test_mse, test_kl = evaluate(vae, test_ds, beta_var)
    print(f"Final test: Loss={test_loss:.5f}, MSE={test_mse:.5f}, KL={test_kl:.5f}")

    save_models(vae, log_dir)


if __name__ == "__main__":
    main()
