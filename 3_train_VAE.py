"""
Train a beta-VAE on voxelized electron densities stored as TFRecords.
"""

import os
from helper import redirect_tf_cpp_logs

redirect_tf_cpp_logs.redirect()
import argparse
import json
import numpy
import tensorflow as tf
from pathlib import Path

from helper.VAE import build_vae, count_records, split_train_val, make_train_dataset, make_val_dataset, make_test_dataset

# =========================================================
# Constants
# =========================================================
SEED = 42
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
    numpy.random.seed(seed)


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
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without improvement in validation MSE (default: 5)")
    parser.add_argument("--min_improvement", type=float, default=0.01, help="Relative drop in the ValMSE that counts as improvement")
    parser.add_argument("--save_every_epoch", type=bool, default=False, help="Saves weights from every epoch (default: False)")
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
    # sse = tf.reduce_sum(tf.square(x - x_pred), axis=[1, 2, 3, 4])
    # l1 = tf.reduce_sum(tf.abs(x - x_pred), axis=[1, 2, 3, 4])
    # kl_raw = kl_divergence(mean, tf.exp(0.5 * logvar))
    # total = l1 + tf.cast(beta, tf.float32) * kl_raw
    # return tf.reduce_mean(total), tf.reduce_mean(sse), kl_raw
    


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


def train(vae, is_train, val_ds, steps_per_epoch, args, log_dir):
    """Runs the epoch loop. Returns the beta variable (holding the final beta)."""
    total_steps = steps_per_epoch * args.epochs
    beta_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    optimizer = make_optimizer(args.learning_rate, total_steps)
    train_step = make_train_step(vae, optimizer, beta_var)
    best_val = float("inf")
    best_path = os.path.join(log_dir, "vae_best.weights.h5")
    epochs_without_improvement = 0

    global_step = 0
    with open(os.path.join(log_dir, "log.txt"), "w") as log_file:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_ds_epoch = make_train_dataset(args.train_datapath, is_train, args.batchsize, seed=SEED + epoch)

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
                f"TrainLoss: {numpy.mean(epoch_losses):.5f}, MSE: {numpy.mean(epoch_mses):.5f}, KL: {numpy.mean(epoch_kls):.5f} | "
                f"ValLoss: {val_loss:.5f}, ValMSE: {val_mse:.5f}, ValKL: {val_kl:.5f} | "
                f"Beta: {float(beta_var.numpy()):.5e}"
            )
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())

            if args.save_every_epoch:
                vae.save_weights(os.path.join(log_dir, f"vae_epoch_{epoch + 1}.weights.h5"))

            if val_mse < best_val * (1 - args.min_improvement):
                best_val = val_mse
                epochs_without_improvement = 0
                vae.save_weights(best_path)
                print(f"  new best ValMSE {best_val:.5f}, saved")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping: no improvement for {args.patience} epochs (best ValMSE {best_val:.5f})")
                    break

    vae.load_weights(best_path)
    print(f"Restored best weights (ValMSE {best_val:.5f})")
    return beta_var


def save_models(vae, log_dir):
    vae.encoder.save(os.path.join(log_dir, "encoder_model.keras"))
    vae.decoder.save(os.path.join(log_dir, "decoder_model.keras"))
    print(f"Models saved in {log_dir}")


def make_log_dir(args):
    """<savepath>/X.vae, where X is the --datapath directory name or the training file name."""
    src = Path(args.datapath) if args.datapath else Path(args.train_datapath)
    name = src.resolve().with_suffix(".vae").name
    log_dir = Path(args.savepath) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


if __name__ == "__main__":
    configure_gpu(mixed_precision=True)
    set_seeds(SEED)

    args = parse_arguments()
    meta = load_meta(args.meta_datapath)
    target_sum = meta["electron_conservation"]["target_sum_normalized"]

    vae = build_vae(args.z_dim, target_sum)
    num_samples = count_records(args.train_datapath)

    is_train, is_val = split_train_val(num_samples)
    val_ds = make_val_dataset(args.train_datapath, is_val, args.batchsize)
    steps_per_epoch = int(is_train.sum()) // args.batchsize

    log_dir = make_log_dir(args)
    beta_var = train(vae, is_train, val_ds, steps_per_epoch, args, log_dir)
    print("Final test evaluation...")
    test_ds = make_test_dataset(args.test_datapath, args.batchsize)
    test_loss, test_mse, test_kl = evaluate(vae, test_ds, beta_var)
    print(f"Final test: Loss={test_loss:.5f}, MSE={test_mse:.5f}, KL={test_kl:.5f}")
    save_models(vae, log_dir)
