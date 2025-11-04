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
    Uses HDF5 direct copy to avoid high memory, and preserves attributes.
    """
    dst = dst_group.create_dataset(name, data=src, dtype=src.dtype, shape=src.shape)
    for a in src.attrs.keys():
        dst.attrs[a] = src.attrs[a]


def build_keep_indices(T: int, offset: int, stride: int) -> List[int]:
    """
    Compute indices to keep for a given original length T, stride, and phase offset.
    Always include the final index T-1 even if it is not congruent to the offset.
    """
    if T < 1:
        raise ValueError("Encountered a trajectory with length < 1, which is invalid.")
    if stride < 1:
        raise ValueError("Stride must be >= 1")
    keep = list(range(offset, T, stride)) if offset < T else []
    if (T - 1) not in keep:
        keep.append(T - 1)
    return keep


def sparsify_row_multi(row: np.ndarray, T: int, stride: int, offset: int) -> Tuple[np.ndarray, int]:
    """
    From a single trajectory row [H, D...], take every 'stride' element starting at 'offset',
    always including the last element (T-1). Return padded row of length H and the new true length.
    """
    H = row.shape[0]
    keep = build_keep_indices(T, offset, stride)
    kept = row[keep]
    new_len = kept.shape[0]
    out = np.empty_like(row)
    out[:new_len] = kept
    if new_len < H:
        out[new_len:] = kept[-1]
    return out, new_len


def rebuild_index_for_key(dst_file: h5py.File, key: str, lengths: np.ndarray) -> np.ndarray:
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

    idx_name = f"{key}{INDEX_SPECIAL_TOKEN}"
    if idx_name in dst_file.keys():
        del dst_file[idx_name]
    ds = dst_file.create_dataset(idx_name, index.shape, dtype=index.dtype)
    ds[...] = index
    return index


def replicate_non_expert(fin: h5py.File, fout: h5py.File, expert_N: int, stride: int, expert_keys: List[str]):
    """
    Copy non-expert datasets. If a dataset has leading dimension equal to the number
    of problems (expert_N), replicate each row 'stride' times along axis 0. Otherwise copy as-is.
    Skip expert datasets and their *_lengths and *_index counterparts.
    """
    for name in fin.keys():
        # Skip expert data and their index/lengths; experts handled separately
        skip = False
        for k in expert_keys:
            if name == k or name == f"{k}{LENGTHS_SPECIAL_TOKEN}" or name == f"{k}{INDEX_SPECIAL_TOKEN}":
                skip = True
                break
        if skip:
            continue

        src = fin[name]
        if hasattr(src, "shape") and len(src.shape) >= 1 and src.shape[0] == expert_N:
            # Replicate along axis 0
            shape = (expert_N * stride, *src.shape[1:])
            dst = fout.create_dataset(name, shape, dtype=src.dtype)
            for a in src.attrs.keys():
                dst.attrs[a] = src.attrs[a]

            chunk = 2048
            for s in range(0, expert_N, chunk):
                e = min(s + chunk, expert_N)
                data_chunk = src[s:e, ...]
                rep = np.repeat(data_chunk, repeats=stride, axis=0)
                dst[s * stride : e * stride, ...] = rep
        else:
            copy_dataset(src, fout, name)


def sparsify_hdf5(input_path: Path, output_path: Path, stride: int, keys: List[str] | None):
    if input_path == output_path:
        raise ValueError("Output path must be different from input path.")
    if stride < 1:
        raise ValueError("Stride must be >= 1.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(input_path), "r") as fin:
        expert_keys = keys if keys is not None and len(keys) > 0 else find_expert_keys(fin)

        # Validate and gather base N (number of problems) from the first expert
        base_key = expert_keys[0]
        base_lengths = np.asarray(fin[f"{base_key}{LENGTHS_SPECIAL_TOKEN}"][...], dtype=np.int64)
        expert_N = fin[base_key].shape[0]
        if base_lengths.shape[0] != expert_N:
            raise ValueError(f"Lengths for '{base_key}' have shape {base_lengths.shape}, expected ({expert_N},).")

        # Validate other experts match base N
        for k in expert_keys[1:]:
            if fin[k].shape[0] != expert_N:
                raise ValueError(f"Expert '{k}' has N={fin[k].shape[0]} != {expert_N} of '{base_key}'.")
            if fin[f"{k}{LENGTHS_SPECIAL_TOKEN}"].shape[0] != expert_N:
                raise ValueError(f"Lengths for '{k}' do not match N={expert_N}.")

        new_N = expert_N * stride

        with h5py.File(str(output_path), "w") as fout:
            # First, copy and replicate non-expert datasets
            replicate_non_expert(fin, fout, expert_N, stride, expert_keys)

            # Now, expand each expert dataset
            for k in expert_keys:
                src = fin[k]
                lengths = np.asarray(fin[f"{k}{LENGTHS_SPECIAL_TOKEN}"][...], dtype=np.int64)
                if not np.all((lengths >= 1) & (lengths <= src.shape[1])):
                    raise ValueError(
                        f"Invalid lengths for '{k}'. Must be in [1, H]. Found min={lengths.min()}, max={lengths.max()}, H={src.shape[1]}"
                    )

                H = src.shape[1]
                dst_shape = (new_N, *src.shape[1:])
                dst = fout.create_dataset(k, dst_shape, dtype=src.dtype)
                for a in src.attrs.keys():
                    dst.attrs[a] = src.attrs[a]

                new_lengths = np.empty((new_N,), dtype=np.int64)

                # Process in chunks of original problems
                chunk = 512
                for s in range(0, expert_N, chunk):
                    e = min(s + chunk, expert_N)
                    rows = src[s:e, :H, ...]
                    len_chunk = lengths[s:e]

                    for i in range(e - s):
                        pidx = s + i
                        T = int(len_chunk[i])
                        row = rows[i]
                        for offset in range(stride):
                            out_idx = pidx * stride + offset
                            new_row, new_T = sparsify_row_multi(row, T, stride, offset)
                            dst[out_idx, :H, ...] = new_row
                            new_lengths[out_idx] = new_T

                # Write new lengths and rebuild index
                len_name = f"{k}{LENGTHS_SPECIAL_TOKEN}"
                len_ds = fout.create_dataset(len_name, new_lengths.shape, dtype=np.int64)
                len_ds[...] = new_lengths

                rebuild_index_for_key(fout, k, new_lengths)
                dst.attrs[WELL_INDEXED] = True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Split each expert trajectory into 'stride' sub-trajectories.")
    ap.add_argument("--input", dest="input_path", type=Path, required=True, help="Path to input .hdf5 file")
    ap.add_argument("--output", dest="output_path", type=Path, required=True, help="Path to output .hdf5 file (will be created)")
    ap.add_argument("--stride", dest="stride", type=int, default=2, help="Number of splits per trajectory; also the stride between kept waypoints")
    ap.add_argument(
        "--keys",
        dest="keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of expert dataset keys to split (defaults to all detected)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sparsify_hdf5(args.input_path, args.output_path, args.stride, args.keys)
    print(f"Wrote expanded dataset to {args.output_path}")


if __name__ == "__main__":
    main()
