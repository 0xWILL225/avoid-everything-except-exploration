#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

INDEX_SPECIAL_TOKEN = "_index"
LENGTHS_SPECIAL_TOKEN = "_lengths"
WELL_INDEXED = "well_indexed"


def find_expert_keys(f: h5py.File) -> List[str]:
    """
    Detect expert trajectory dataset names by the presence of a companion
    '<key>_lengths' dataset. Returns base keys (without suffixes).
    """
    keys = set(f.keys())
    expert_keys: List[str] = []
    for k in keys:
        if k.endswith(LENGTHS_SPECIAL_TOKEN):
            base = k[: -len(LENGTHS_SPECIAL_TOKEN)]
            if base in keys:
                expert_keys.append(base)
    if not expert_keys:
        raise RuntimeError("No expert keys found (no '<key>_lengths' present).")
    return sorted(expert_keys)


def copy_dataset(src: h5py.Dataset, dst_group: h5py.Group, name: str):
    """
    Copy a dataset from src file to destination group with same shape/dtype.
    Uses chunked copying to avoid high memory, and preserves attributes.
    """
    dst = dst_group.create_dataset(name, data=src, dtype=src.dtype, shape=src.shape)
    # Copy attributes
    for a in src.attrs.keys():
        dst.attrs[a] = src.attrs[a]


def sparsify_trajectory_row(
    row: np.ndarray, original_length: int, stride: int
) -> Tuple[np.ndarray, int]:
    """
    Given a single trajectory row of shape [H, D] and the true original length T,
    return a new row (same H, D) sparsified by 'stride', padded by repeating the
    last kept state, and the new true length T'.
    """
    if original_length < 1:
        raise ValueError("Encountered a trajectory with length < 1, which is invalid.")
    if stride < 1:
        raise ValueError("Stride must be >= 1")

    H = row.shape[0]
    # Indices to keep: take every 'stride' step starting at 0; ensure we include the last frame
    keep = list(range(0, original_length, stride))
    if keep[-1] != original_length - 1:
        keep.append(original_length - 1)

    kept = row[keep]  # [T', D]
    new_len = kept.shape[0]

    # Pad to H by repeating the last kept state (edge padding semantics)
    out = np.empty_like(row)
    out[:new_len] = kept
    if new_len < H:
        out[new_len:] = kept[-1]
    return out, new_len


def rebuild_index_for_key(
    dst_file: h5py.File, key: str, lengths: np.ndarray
) -> np.ndarray:
    """
    Build '<key>_index' dataset as an array of shape [sum(lengths), 2], mapping
    (pidx, t) for t in [0, length[pidx]). Returns the built index array.
    """
    total_states = int(lengths.sum())
    index = np.zeros((total_states, 2), dtype=np.int64)
    offset = 0
    for pidx, T in enumerate(lengths.tolist()):
        if T < 1:
            raise ValueError(
                f"Rebuilt length {T} for pidx={pidx} is invalid (must be >= 1)."
            )
        index[offset : offset + T, 0] = pidx
        index[offset : offset + T, 1] = np.arange(T, dtype=np.int64)
        offset += T

    # Create or replace index dataset
    idx_name = f"{key}{INDEX_SPECIAL_TOKEN}"
    if idx_name in dst_file.keys():
        del dst_file[idx_name]
    ds = dst_file.create_dataset(idx_name, index.shape, dtype=index.dtype)
    ds[...] = index
    return index


def sparsify_hdf5(
    input_path: Path, output_path: Path, stride: int, keys: List[str] | None
):
    if input_path == output_path:
        raise ValueError("Output path must be different from input path.")
    if stride < 1:
        raise ValueError("Stride must be >= 1.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(input_path), "r") as fin:
        expert_keys = keys if keys is not None and len(keys) > 0 else find_expert_keys(fin)

        # Validate expert datasets exist and shapes are consistent
        lengths_map: Dict[str, np.ndarray] = {}
        shapes_map: Dict[str, Tuple[int, ...]] = {}
        for k in expert_keys:
            if k not in fin.keys():
                raise KeyError(f"Expert dataset '{k}' not found in input file.")
            len_key = f"{k}{LENGTHS_SPECIAL_TOKEN}"
            if len_key not in fin.keys():
                raise KeyError(f"Lengths dataset '{len_key}' not found for expert '{k}'.")
            lengths = np.asarray(fin[len_key][...], dtype=np.int64)
            data = fin[k]
            if data.ndim < 2:
                raise ValueError(
                    f"Expert dataset '{k}' must be at least 2D (N, H, ...). Got shape {data.shape}."
                )
            N = data.shape[0]
            if lengths.shape[0] != N:
                raise ValueError(
                    f"Lengths for '{k}' have shape {lengths.shape}, expected ({N},)."
                )
            if not np.all((lengths >= 1) & (lengths <= data.shape[1])):
                raise ValueError(
                    f"Invalid lengths for '{k}'. Must be in [1, H]. Found min={lengths.min()}, max={lengths.max()}, H={data.shape[1]}"
                )
            lengths_map[k] = lengths
            shapes_map[k] = data.shape

        # Create destination file and copy non-expert datasets as-is
        with h5py.File(str(output_path), "w") as fout:
            for name in fin.keys():
                # Skip expert data and their index/lengths; we will rebuild
                skip = False
                for k in expert_keys:
                    if name == k:
                        skip = True
                        break
                    if name == f"{k}{LENGTHS_SPECIAL_TOKEN}" or name == f"{k}{INDEX_SPECIAL_TOKEN}":
                        skip = True
                        break
                if skip:
                    continue
                copy_dataset(fin[name], fout, name)

            # For each expert dataset, sparsify rows and rebuild lengths and index
            for k in expert_keys:
                src = fin[k]
                lengths = lengths_map[k]
                shape = shapes_map[k]  # (N, H, D...)
                N, H = shape[0], shape[1]

                dst = fout.create_dataset(k, shape, dtype=src.dtype)
                # Copy attributes, but we'll override WELL_INDEXED at the end
                for a in src.attrs.keys():
                    dst.attrs[a] = src.attrs[a]

                new_lengths = np.empty_like(lengths)

                # Chunk over problems to avoid loading whole arrays into RAM
                indices = np.arange(N)
                chunk_size = 1000
                nchunks = int(np.ceil(N / chunk_size))
                for ci in range(nchunks):
                    sl = slice(ci * chunk_size, min((ci + 1) * chunk_size, N))
                    for pidx in range(sl.start, sl.stop):
                        T = int(lengths[pidx])
                        row = src[pidx, :H, ...]
                        new_row, new_T = sparsify_trajectory_row(row, T, stride)
                        dst[pidx, :H, ...] = new_row
                        new_lengths[pidx] = new_T

                # Write lengths dataset
                len_name = f"{k}{LENGTHS_SPECIAL_TOKEN}"
                len_ds = fout.create_dataset(len_name, new_lengths.shape, dtype=np.int64)
                len_ds[...] = new_lengths

                # Rebuild index and set WELL_INDEXED
                rebuild_index_for_key(fout, k, new_lengths)
                dst.attrs[WELL_INDEXED] = True

            # Ensure persistent key exists (sanity)
            # Not strictly needed here; we avoid silent fixes per instructions.


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sparsify expert trajectories in an HDF5 dataset.")
    ap.add_argument("--input", dest="input_path", type=Path, required=True, help="Path to input .hdf5 file")
    ap.add_argument("--output", dest="output_path", type=Path, required=True, help="Path to output .hdf5 file (will be created)")
    ap.add_argument("--stride", dest="stride", type=int, default=2, help="Keep every 'stride' waypoint; ensure last is kept")
    ap.add_argument(
        "--keys",
        dest="keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of expert dataset keys to sparsify (defaults to all detected)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sparsify_hdf5(args.input_path, args.output_path, args.stride, args.keys)
    print(f"Wrote sparsified dataset to {args.output_path}")


if __name__ == "__main__":
    main()
