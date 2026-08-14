# coding: utf-8

"""
Voxel-only TFRecord pipeline with global min–max normalization [0, 1]

Input:
- Directory of .npy files, each with shape (32, 32, 32)

Output:
- TFRecords (train/test)
- Metadata JSON with normalization parameters

Usage:
  python voxels_to_tfrecords_minmax.py <voxels_dir> <output_dir>
"""

import os
import sys
import glob
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import h5py
import hashlib

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GRID_SIZE = 52
TRAIN_SPLIT_MOD = 10      # every 5th sample → test
NUM_THREADS = 8

# ─────────────────────────────────────────────
# TFRecord helpers
# ─────────────────────────────────────────────
def serialize_example(flat_array, index):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "data": tf.train.Feature(
                    float_list=tf.train.FloatList(value=flat_array)
                ),
                "index": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[index])
                ),
            }
        )
    ).SerializeToString()

def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {os.path.basename(filename)}"):
            writer.write(ex)

# ─────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────
def run_workflow(voxels_dir, output_dir):

    output_dir = os.path.abspath(output_dir)

    tf_out = os.path.join(output_dir, "tfrecords")
    meta_out = os.path.join(output_dir, "meta")

    os.makedirs(tf_out, exist_ok=True)
    os.makedirs(meta_out, exist_ok=True)

    # ------------------------------------------------
    # Load voxel files
    # ------------------------------------------------
    with h5py.File(voxels_dir, "r") as voxels_file:
        print(f"Loaded {len(voxels_file)} samples")
        voxels = voxels_file['rho'][()]
        # ------------------------------------------------
        # Global min–max normalization to [0, 1]
        # ------------------------------------------------
        X_MIN = voxels.min()
        X_MAX = voxels.max()

        if X_MAX <= X_MIN:
            raise RuntimeError(
                f"Invalid normalization range: min={X_MIN}, max={X_MAX}"
            )

        voxels = (voxels - X_MIN) / (X_MAX - X_MIN)
        voxels = voxels.astype(np.float32)

    # Sanity check
    print(f"Normalization:")
    print(f"  global min = {X_MIN:.6g}")
    print(f"  global max = {X_MAX:.6g}")
    print(f"  after norm → min={voxels.min():.3f}, max={voxels.max():.3f}")

    # ------------------------------------------------
    # Serialize to TFRecords
    # ------------------------------------------------
    examples_train = []
    examples_test = []

    def process_sample(idx):
        flat = voxels[idx].reshape(-1)
        return idx, serialize_example(flat, idx)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(
            tqdm(
                executor.map(process_sample, range(len(voxels))),
                total=len(voxels),
                desc="Serializing"
            )
        )

    for idx, ex in results:
        if idx % TRAIN_SPLIT_MOD == 0:
            examples_test.append(ex)
        else:
            examples_train.append(ex)

    write_tfrecord(os.path.join(tf_out, "train.tfrecords"), examples_train)
    write_tfrecord(os.path.join(tf_out, "test.tfrecords"), examples_test)

    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------
    meta = {
        "grid_size": GRID_SIZE,
        "voxel_shape": [GRID_SIZE, GRID_SIZE, GRID_SIZE],
        "voxel_count": len(voxels),
        "train_samples": len(examples_train),
        "test_samples": len(examples_test),
        "normalized": True,
        "normalization": {
            "type": "minmax",
            "min": float(X_MIN),
            "max": float(X_MAX),
        },
        "source_dir": voxels_dir,
    }

    with open(os.path.join(meta_out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("TFRecord generation complete")
    print(f"  Train TFRecord: {os.path.join(tf_out, 'train.tfrecords')}")
    print(f"  Test TFRecord : {os.path.join(tf_out, 'test.tfrecords')}")
    print(f"  Meta          : {os.path.join(meta_out, 'meta.json')}")

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python voxels_to_tfrecords_minmax.py <voxels_dir> <output_dir>")
        sys.exit(1)

    voxels_dir = sys.argv[1]
    output_dir = sys.argv[2]

    run_workflow(voxels_dir, output_dir)

