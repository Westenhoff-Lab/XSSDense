"""
Saves voxel density as TFRecord with global min-max normalization [0, 1]
Usage:
    python 2_generate_tfrecord.py \
    --voxels_dir "data/1crn_41bb_1d61dv_5co.h5" \
    --output_dir "data/1crn_41bb_1d61dv_5co.tfrecord"
"""

import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import h5py
import argparse


# ─────────────────────────────────────────────
# TFRecord helpers
# ─────────────────────────────────────────────
def serialize_example(flat_array, index):
    return tf.train.Example(features=tf.train.Features(feature={"data": tf.train.Feature(float_list=tf.train.FloatList(value=flat_array)), "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))})).SerializeToString()


def serialize_frame(args):
    idx, flat = args
    return idx, serialize_example(flat, idx)


def frame_jobs(voxels):
    for idx in range(len(voxels)):
        yield idx, voxels[idx].reshape(-1)


def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {os.path.basename(filename)}"):
            writer.write(ex)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxels_dir", help="Directory containing voxel .h5 files", required=True)
    parser.add_argument("--output_dir", help="Directory to save TFRecords and metadata", required=True)
    parser.add_argument("--grid_size", help="Size of the voxel grid", default=52, type=int)
    parser.add_argument("--train_split_mod", help="Modulo for train/test split (every Nth sample goes to test)", default=10, type=int)
    parser.add_argument("--num_proc", help="Number of processes for parallel processing", default=8, type=int)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────
def run_workflow(voxels_dir, output_dir, grid_size, train_split_mod, num_proc):
    output_dir = os.path.abspath(output_dir)
    tf_out = os.path.join(output_dir, "tfrecords")
    meta_out = os.path.join(output_dir, "meta")
    os.makedirs(tf_out, exist_ok=True)
    os.makedirs(meta_out, exist_ok=True)
    # ------------------------------------------------
    # Load voxel files
    # ------------------------------------------------
    with h5py.File(voxels_dir, "r") as voxels_file:
        print(f"Loaded {voxels_file['rho'].shape[0]} samples")
        raw = voxels_file["rho"][()]
        dv = voxels_file["dv"][()]

    n_electrons_per_sample = np.sum(raw, axis=tuple(range(1, raw.ndim))) * dv**3
    n_electrons = np.mean(n_electrons_per_sample)
    n_electrons_std = np.std(n_electrons_per_sample)
    rel_spread = n_electrons_std / n_electrons
    print(f"num electrons: mean={n_electrons:.6g}, rel. std={rel_spread:.2%}")
    if rel_spread > 0.01:
        print("WARNING: sums vary >1% — electron count may not be constant, or density is clipped at the box edge")
    # ------------------------------------------------
    # Global min-max normalization to [0, 1]
    # ------------------------------------------------
    X_MIN = raw.min()
    X_MAX = raw.max()

    if X_MAX <= X_MIN:
        raise RuntimeError(f"Invalid normalization range: min={X_MIN}, max={X_MAX}")

    voxels = raw.astype(np.float32)
    voxels = (voxels - X_MIN) / (X_MAX - X_MIN)

    sample_sums = np.sum(voxels, axis=tuple(range(1, voxels.ndim)))
    target_sum = float(sample_sums.mean())
    del raw  # free memory

    # Sanity check
    print(f"Normalization:")
    print(f"  global min = {X_MIN:.6g}")
    print(f"  global max = {X_MAX:.6g}")
    print(f"  after norm → min={voxels.min():.3f}, max={voxels.max():.3f}, density sum: {target_sum:.3f}")
    # ------------------------------------------------
    # Serialize to TFRecords
    # ------------------------------------------------
    examples_train = []
    examples_test = []

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        results = list(tqdm(executor.map(serialize_frame, frame_jobs(voxels), chunksize=16), total=len(voxels), desc="Serializing"))

    for idx, ex in results:
        if idx % train_split_mod == 0:
            examples_test.append(ex)
        else:
            examples_train.append(ex)

    write_tfrecord(os.path.join(tf_out, "train.tfrecords"), examples_train)
    write_tfrecord(os.path.join(tf_out, "test.tfrecords"), examples_test)
    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------
    meta = {
        "grid_size": grid_size,
        "voxel_shape": [grid_size, grid_size, grid_size],
        "voxel_count": len(voxels),
        "train_samples": len(examples_train),
        "test_samples": len(examples_test),
        "normalized": True,
        "normalization": {
            "type": "minmax",
            "min": float(X_MIN),
            "max": float(X_MAX),
        },
        "electron_conservation": {
            "n_electrons": float(n_electrons),
            "target_sum_normalized": float(target_sum),
            "n_electrons_std": float(n_electrons_std),
        },
        "source_dir": voxels_dir,
    }

    with open(os.path.join(meta_out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("TFRecord generation complete")
    print(f"  Train TFRecord: {os.path.join(tf_out, 'train.tfrecords')}")
    print(f"  Test TFRecord : {os.path.join(tf_out, 'test.tfrecords')}")
    print(f"  Meta          : {os.path.join(meta_out, 'meta.json')}")


if __name__ == "__main__":
    args = parse_arguments()
    run_workflow(args.voxels_dir, args.output_dir, args.grid_size, args.train_split_mod, args.num_proc)
