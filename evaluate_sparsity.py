#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

INDEX_SPECIAL_TOKEN = "_index"
LENGTHS_SPECIAL_TOKEN = "_lengths"


def find_expert_keys(f: h5py.File) -> List[str]:
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


def evaluate_key(fin: h5py.File, key: str, chunk_size: int = 512) -> Dict[str, float]:
    data = fin[key]
    lengths = np.asarray(fin[f"{key}{LENGTHS_SPECIAL_TOKEN}"][...], dtype=np.int64)

    N = int(data.shape[0])
    H = int(data.shape[1])

    if lengths.shape[0] != N:
        raise ValueError(f"Lengths for '{key}' have shape {lengths.shape}, expected ({N},).")
    if not np.all((lengths >= 1) & (lengths <= H)):
        raise ValueError(
            f"Invalid lengths for '{key}'. Must be in [1, H]. Found min={lengths.min()}, max={lengths.max()}, H={H}"
        )

    mean_len = float(lengths.mean())

    # Accumulate distances across all valid consecutive pairs
    total_pairs = 0
    total_dist_sum = 0.0

    # Process in chunks of problems
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        rows = data[s:e, :H, ...]  # [B, H, D]
        len_chunk = lengths[s:e]

        for i in range(e - s):
            T = int(len_chunk[i])
            if T <= 1:
                # No consecutive pairs in this trajectory
                continue
            traj = rows[i, :T, ...]
            diffs = traj[1:] - traj[:-1]  # [T-1, D]
            # Flatten trailing dims into D vector if needed
            if diffs.ndim > 2:
                diffs = diffs.reshape(diffs.shape[0], -1)
            dists = np.linalg.norm(diffs, axis=1)  # [T-1]
            total_dist_sum += float(dists.sum())
            total_pairs += (T - 1)

    avg_step_dist = float(total_dist_sum / total_pairs) if total_pairs > 0 else 0.0

    return {
        "num_trajectories": float(N),
        "avg_waypoints_per_trajectory": mean_len,
        "avg_consecutive_step_distance": avg_step_dist,
        "total_pairs": float(total_pairs),
    }


def evaluate_file(path: Path, keys: List[str] | None = None, per_key: bool = True) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    with h5py.File(str(path), "r") as fin:
        expert_keys = keys if (keys is not None and len(keys) > 0) else find_expert_keys(fin)
        agg_pairs = 0
        agg_dist_sum = 0.0
        agg_traj = 0
        agg_len_sum = 0.0

        for k in expert_keys:
            res = evaluate_key(fin, k)
            if per_key:
                results[k] = res
            # Aggregate
            agg_traj += int(res["num_trajectories"])
            agg_len_sum += res["avg_waypoints_per_trajectory"] * res["num_trajectories"]
            agg_pairs += int(res["total_pairs"])
            agg_dist_sum += res["avg_consecutive_step_distance"] * res["total_pairs"]

        # Combined metrics (weighted)
        results["__aggregate__"] = {
            "num_trajectories": float(agg_traj),
            "avg_waypoints_per_trajectory": float(agg_len_sum / agg_traj) if agg_traj > 0 else 0.0,
            "avg_consecutive_step_distance": float(agg_dist_sum / agg_pairs) if agg_pairs > 0 else 0.0,
            "total_pairs": float(agg_pairs),
        }
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate sparsity metrics for HDF5 expert datasets.")
    ap.add_argument("--input", dest="input_path", type=Path, required=True, help="Path to input .hdf5 file")
    ap.add_argument(
        "--keys",
        dest="keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of expert dataset keys to evaluate (defaults to all detected)",
    )
    ap.add_argument("--no-per-key", dest="per_key", action="store_false", help="Hide per-key metrics and only show aggregate")
    return ap.parse_args()


def main():
    args = parse_args()
    results = evaluate_file(args.input_path, keys=args.keys, per_key=args.per_key)

    # Pretty print
    print(f"File: {args.input_path}")
    for k, res in results.items():
        label = "aggregate" if k == "__aggregate__" else k
        print(
            f"[{label}] N={int(res['num_trajectories'])}, avg_len={res['avg_waypoints_per_trajectory']:.4f}, avg_step_dist={res['avg_consecutive_step_distance']:.6f}"
        )


if __name__ == "__main__":
    main()
