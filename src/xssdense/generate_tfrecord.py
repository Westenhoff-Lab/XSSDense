"""
Save voxel densities as TFRecords with global min-max normalization to [0, 1].
Usage:
    xssdense-gen-tfrecord \
        --h5 "data/1crn_41bb_1d61dv_5co.h5" \
        --output_dir "data/1crn_41bb_1d61dv_5co.tfrecord"
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# =========================================================
# TFRecord helpers
# =========================================================
def serialize_example(flat_array, index):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "data": tf.train.Feature(float_list=tf.train.FloatList(value=flat_array)),
                "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            }
        )
    ).SerializeToString()


def serialize_frame(args):
    idx, flat = args
    return idx, serialize_example(flat, idx)


def frame_jobs(voxels):
    for idx in range(len(voxels)):
        yield idx, voxels[idx].reshape(-1)


def serialize_all(voxels, num_proc):
    """Serialized examples of all frames, in frame order."""
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        results = list(tqdm(executor.map(serialize_frame, frame_jobs(voxels), chunksize=16), total=len(voxels), desc="Serializing"))
    return [ex for _, ex in results]


def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {os.path.basename(filename)}"):
            writer.write(ex)


# =========================================================
# Data steps
# =========================================================
def load_rho(path):
    """Densities, voxel size and bounding box from the voxelisation h5 (bbmax None in old files)."""
    with h5py.File(path, "r") as f:
        rho = f["rho"][()]
        dv = float(f["dv"][()])
        bbmax = float(f["bbmax"][()]) if "bbmax" in f else None
    print(f"Loaded {rho.shape[0]} frames of {rho.shape[1:]}, voxel size {dv:.4f} Å, bounding box {bbmax} Å")
    return rho, dv, bbmax


def electron_count(rho, dv):
    """Mean and std of the electron count per frame.
    Warn if it varies by more than 1%."""
    n = rho.sum(axis=tuple(range(1, rho.ndim)), dtype=np.float64) * dv**3
    mean, std = n.mean(), n.std()
    print(f"Electrons per frame: mean {mean:.6g}, rel. std {std / mean:.2%}")
    if std / mean > 0.01:
        print("WARNING: electron count varies >1% (not constant, or density clipped at the box edge)")
    return float(mean), float(std)


def normalize(rho):
    """Global min-max normalization to [0, 1]. Returns (rho_norm, min, max)."""
    x_min, x_max = float(rho.min()), float(rho.max())
    if x_max <= x_min:
        raise RuntimeError(f"Invalid normalization range: min={x_min}, max={x_max}")
    rho_norm = ((rho - x_min) / (x_max - x_min)).astype(np.float32)
    print(f"Normalization: min {x_min:.6g}, max {x_max:.6g}")
    return rho_norm, x_min, x_max


def split_blocks(n_frames, block, test_every):
    """Every `test_every`-th block of `block` consecutive frames is test, the rest is training.
    Not needed for independent structures (e.g. AFsample2), but important for MD,
    where neighbouring frames are correlated."""
    pos = np.arange(n_frames)
    is_test = (pos // block) % test_every == 0
    is_train = ~is_test
    print(f"Split: {is_train.sum()} train, {is_test.sum()} test")
    return is_train, is_test


def mask_to_ranges(mask):
    """Boolean mask -> half-open [start, stop) runs of True, e.g. [[0, 100], [1000, 1100]]."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]


def compact_json(obj):
    """json.dumps with indent=2, but lists of plain numbers kept on one line.
    Only lists spanning several lines are collapsed; strings never contain raw newlines, so they are untouched."""
    text = json.dumps(obj, indent=2)
    return re.sub(r"\[(\s*\n[-+\d.eE,\s]*?)\]",lambda m: "[" + ", ".join(s.strip() for s in m.group(1).split(",")) + "]", text)


# =========================================================
# Main
# =========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Save voxel densities as normalized TFRecords with a block split")
    parser.add_argument("--h5", required=True, help=".h5 with datasets 'rho' and 'dv'")
    parser.add_argument("--output_dir", required=True, help="Directory for the TFRecords and metadata")
    parser.add_argument("--block", type=int, default=100, help="Frames per block (default: 100)")
    parser.add_argument("--test_every", type=int, default=10, help="Every n-th block is test (default: 10, i.e. 10%%)")
    parser.add_argument("--num_proc", type=int, default=8, help="Processes for serialization (default: 8)")
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args


def main():
    args = parse_arguments()
    output_dir = os.path.abspath(args.output_dir)
    tf_out = os.path.join(output_dir, "tfrecords")
    meta_out = os.path.join(output_dir, "meta")
    os.makedirs(tf_out, exist_ok=True)
    os.makedirs(meta_out, exist_ok=True)

    rho, dv, bbmax = load_rho(args.h5)
    grid_size = rho.shape[1]
    n_electrons, n_electrons_std = electron_count(rho, dv)

    is_train, is_test = split_blocks(len(rho), args.block, args.test_every)
    train_indices, test_indices = np.flatnonzero(is_train), np.flatnonzero(is_test)

    rho_norm, x_min, x_max = normalize(rho)
    del rho  # free memory
    target_sum = float(rho_norm.sum(axis=tuple(range(1, rho_norm.ndim)), dtype=np.float64).mean())

    examples = serialize_all(rho_norm, args.num_proc)
    write_tfrecord(os.path.join(tf_out, "train.tfrecords"), [examples[i] for i in train_indices])
    write_tfrecord(os.path.join(tf_out, "test.tfrecords"), [examples[i] for i in test_indices])

    meta = {
        "grid_size": grid_size,
        "voxel_shape": [grid_size] * 3,
        "dv": dv,  # copied from the voxelisation h5
        "bbmax": bbmax,  # copied from the voxelisation h5
        "n_frames": len(rho_norm),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        # frame ranges into the source h5, half-open [start, stop); expand with range(start, stop)
        "split": {
            "type": "blocks",
            "block": args.block,
            "test_every": args.test_every,
            "train_ranges": mask_to_ranges(is_train),
            "test_ranges": mask_to_ranges(is_test),
        },
        "normalized": True,
        "normalization": {"type": "minmax", "min": x_min, "max": x_max},
        "electron_conservation": {
            "n_electrons": n_electrons,
            "n_electrons_std": n_electrons_std,
            "target_sum_normalized": target_sum,
        },
        "source_dir": args.h5,
    }
    with open(os.path.join(meta_out, "meta.json"), "w") as f:
        f.write(compact_json(meta) + "\n")

    print("TFRecord generation complete")
    for name in ("train.tfrecords", "test.tfrecords"):
        print(f"  {os.path.join(tf_out, name)}")
    print(f"  {os.path.join(meta_out, 'meta.json')}")


if __name__ == "__main__":
    main()
