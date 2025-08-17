import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import h5py

try:
    from pxr import Usd, UsdGeom, Gf
except Exception as e:  # pragma: no cover - environment dependent
    Usd = None
    UsdGeom = None
    Gf = None


def _read_usd_file(usda_path: Path) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, float, float, np.ndarray]], List[np.ndarray]]:
    """
    Parse a single USDA file and extract:
    - cuboids: list of (center[3], dims[3], quat_wxyz[4])
    - cylinders: list of (center[3], radius, height, quat_wxyz[4])
    - trajectories: list of arrays with shape (T, D)
    """
    if Usd is None:
        return _read_usd_file_ascii(usda_path)

    stage = Usd.Stage.Open(str(usda_path))

    # Obstacles
    cuboids: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cylinders: List[Tuple[np.ndarray, float, float, np.ndarray]] = []
    obstacles_prim = stage.GetPrimAtPath("/Data/Obstacles")
    if obstacles_prim:
        children = obstacles_prim.GetChildren()
        for i in range(len(children)):
            prim = stage.GetPrimAtPath(f"/Data/Obstacles/primitive_{i}")
            if not prim:
                continue
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            # Expect translate, orient, scale as in sample USD
            translate = None
            orient = None
            scale = None
            for op in ops:
                name = op.GetOpName()
                if name.endswith(":translate"):
                    translate = np.array(op.Get(), dtype=float)
                elif name.endswith(":orient"):
                    orient = op.Get()
                elif name.endswith(":scale"):
                    scale = np.array(op.Get(), dtype=float)
            if translate is None:
                translate = np.zeros(3, dtype=float)
            if orient is None:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                q = Gf.Quatf(orient)
                quat_wxyz = np.array([q.GetReal(), *list(q.GetImaginary())], dtype=float)
            if prim.GetTypeName() == "Cube":
                # size default is 2, dims are 2*scale
                if scale is None:
                    scale = np.ones(3, dtype=float)
                dims = 2.0 * scale
                cuboids.append((translate.astype(float), dims.astype(float), quat_wxyz.astype(float)))
            elif prim.GetTypeName() == "Cylinder":
                radius_attr = prim.GetAttribute("radius")
                height_attr = prim.GetAttribute("height")
                radius = float(radius_attr.Get()) if radius_attr else 0.0
                height = float(height_attr.Get()) if height_attr else 0.0
                cylinders.append((translate.astype(float), radius, height, quat_wxyz.astype(float)))

    # Trajectories
    trajectories: List[np.ndarray] = []
    trajectories_prim = stage.GetPrimAtPath("/Data/Trajectories")
    if trajectories_prim:
        children = trajectories_prim.GetChildren()
        for i in range(len(children)):
            traj_prim = stage.GetPrimAtPath(f"/Data/Trajectories/trajectory_{i}")
            if not traj_prim:
                continue
            length_attr = traj_prim.GetAttribute("length")
            length = int(length_attr.Get()) if length_attr else 0
            if length <= 0:
                continue
            configs: List[np.ndarray] = []
            for t in range(length):
                cfg_prim = stage.GetPrimAtPath(f"/Data/Trajectories/trajectory_{i}/config_{t}")
                if not cfg_prim:
                    break
                values_attr = cfg_prim.GetAttribute("values")
                values = values_attr.Get() if values_attr else None
                if values is None:
                    break
                configs.append(np.asarray(values, dtype=float))
            if len(configs) > 0:
                trajectories.append(np.stack(configs, axis=0))

    return cuboids, cylinders, trajectories


def _parse_tuple(s: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in s.split(',') if x.strip() != ''], dtype=float)


def _read_usd_file_ascii(usda_path: Path) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, float, float, np.ndarray]], List[np.ndarray]]:
    """
    Lightweight parser for the specific USDA structure used in this dataset.
    """
    import re

    with open(usda_path, 'r') as f:
        lines = f.readlines()

    cuboids: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cylinders: List[Tuple[np.ndarray, float, float, np.ndarray]] = []
    trajectories: List[np.ndarray] = []

    i = 0
    n = len(lines)

    # Parse Obstacles
    while i < n:
        line = lines[i]
        if 'def "Obstacles"' in line or 'def "Obstacles"' in line:
            i += 1
            depth = 1
            while i < n and depth > 0:
                l = lines[i]
                if '{' in l:
                    depth += l.count('{')
                if '}' in l:
                    depth -= l.count('}')
                # Cube block
                m_cube = re.search(r"Cube\s+\"primitive_", l)
                m_cyl = re.search(r"Cylinder\s+\"primitive_", l)
                if m_cube or m_cyl:
                    prim_type = 'Cube' if m_cube else 'Cylinder'
                    # enter block
                    i += 1
                    pdepth = 1
                    translate = np.zeros(3, dtype=float)
                    scale = np.ones(3, dtype=float)
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                    radius = 0.0
                    height = 0.0
                    while i < n and pdepth > 0:
                        pline = lines[i]
                        if '{' in pline:
                            pdepth += pline.count('{')
                        if '}' in pline:
                            pdepth -= pline.count('}')
                        m_t = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", pline)
                        if m_t:
                            translate = _parse_tuple(m_t.group(1))
                        m_s = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", pline)
                        if m_s:
                            scale = _parse_tuple(m_s.group(1))
                        m_o = re.search(r"xformOp:orient\s*=\s*\(([^)]*)\)", pline)
                        if m_o:
                            quat = _parse_tuple(m_o.group(1))
                        m_r = re.search(r"\b(radius)\b\s*=\s*([\d\.eE\-]+)", pline)
                        if m_r:
                            try:
                                radius = float(m_r.group(2))
                            except Exception:
                                pass
                        m_h = re.search(r"\b(height)\b\s*=\s*([\d\.eE\-]+)", pline)
                        if m_h:
                            try:
                                height = float(m_h.group(2))
                            except Exception:
                                pass
                        i += 1
                    # After block
                    if prim_type == 'Cube':
                        dims = 2.0 * scale
                        cuboids.append((translate.astype(float), dims.astype(float), quat.astype(float)))
                    else:
                        cylinders.append((translate.astype(float), float(radius), float(height), quat.astype(float)))
                    continue  # continue without extra i++
                i += 1
        else:
            i += 1

    # Parse Trajectories
    i = 0
    while i < n:
        line = lines[i]
        if re.search(r"def\s+\"Trajectories\"", line):
            i += 1
            depth = 1
            while i < n and depth > 0:
                l = lines[i]
                if '{' in l:
                    depth += l.count('{')
                if '}' in l:
                    depth -= l.count('}')
                m_tr = re.search(r"def\s+\"trajectory_(\d+)\"", l)
                if m_tr:
                    # enter trajectory block
                    i += 1
                    tdepth = 1
                    seq: List[np.ndarray] = []
                    while i < n and tdepth > 0:
                        tl = lines[i]
                        if '{' in tl:
                            tdepth += tl.count('{')
                        if '}' in tl:
                            tdepth -= tl.count('}')
                        m_vals = re.search(r"custom\s+double\[\]\s+values\s*=\s*\[([^\]]*)\]", tl)
                        if m_vals:
                            arr = np.array([float(x.strip()) for x in m_vals.group(1).split(',') if x.strip() != ''], dtype=float)
                            seq.append(arr)
                        i += 1
                    if len(seq) > 0:
                        # ensure consistent order by occurrence
                        trajectories.append(np.stack(seq, axis=0))
                    continue
                i += 1
        else:
            i += 1

    return cuboids, cylinders, trajectories


def _collect_from_directory(input_dir: Path, max_files: int | None, max_trajectories: int | None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Iterate USD files and collect per-trajectory obstacle arrays and trajectories.
    Returns lists matching per-trajectory:
      cuboid_centers, cuboid_dims, cuboid_quats, cylinder_centers, cylinder_radii, cylinder_heights, cylinder_quats, trajectories
    """
    usda_files = sorted([p for p in input_dir.glob("*.usda")])
    if max_files is not None:
        usda_files = usda_files[:max_files]

    cuboid_centers_list: List[np.ndarray] = []
    cuboid_dims_list: List[np.ndarray] = []
    cuboid_quats_list: List[np.ndarray] = []
    cylinder_centers_list: List[np.ndarray] = []
    cylinder_radii_list: List[np.ndarray] = []
    cylinder_heights_list: List[np.ndarray] = []
    cylinder_quats_list: List[np.ndarray] = []
    trajectories_list: List[np.ndarray] = []

    for usda in usda_files:
        cuboids, cylinders, trajectories = _read_usd_file(usda)
        for traj in trajectories:
            # attach the same obstacles to each trajectory in the file
            if cuboids:
                cub_centers = np.stack([c for c, _, _ in cuboids], axis=0)
                cub_dims = np.stack([d for _, d, _ in cuboids], axis=0)
                cub_quats = np.stack([q for _, _, q in cuboids], axis=0)
            else:
                cub_centers = np.array([[0.0, 0.0, 0.0]], dtype=float)
                cub_dims = np.array([[0.0, 0.0, 0.0]], dtype=float)
                cub_quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)

            if cylinders:
                cyl_centers = np.stack([c for c, _, _, _ in cylinders], axis=0)
                cyl_radii = np.stack([[r] for _, r, _, _ in cylinders], axis=0)
                cyl_heights = np.stack([[h] for _, _, h, _ in cylinders], axis=0)
                cyl_quats = np.stack([q for _, _, _, q in cylinders], axis=0)
            else:
                cyl_centers = np.array([[0.0, 0.0, 0.0]], dtype=float)
                cyl_radii = np.array([[0.0]], dtype=float)
                cyl_heights = np.array([[0.0]], dtype=float)
                cyl_quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)

            cuboid_centers_list.append(cub_centers)
            cuboid_dims_list.append(cub_dims)
            cuboid_quats_list.append(cub_quats)
            cylinder_centers_list.append(cyl_centers)
            cylinder_radii_list.append(cyl_radii)
            cylinder_heights_list.append(cyl_heights)
            cylinder_quats_list.append(cyl_quats)
            trajectories_list.append(traj)

            if max_trajectories is not None and len(trajectories_list) >= max_trajectories:
                return (
                    cuboid_centers_list,
                    cuboid_dims_list,
                    cuboid_quats_list,
                    cylinder_centers_list,
                    cylinder_radii_list,
                    cylinder_heights_list,
                    cylinder_quats_list,
                    trajectories_list,
                )

    return (
        cuboid_centers_list,
        cuboid_dims_list,
        cuboid_quats_list,
        cylinder_centers_list,
        cylinder_radii_list,
        cylinder_heights_list,
        cylinder_quats_list,
        trajectories_list,
    )


def _pad_and_stack(list_of_arrays: List[np.ndarray], target_shape_tail: Tuple[int, ...], pad_values: np.ndarray) -> np.ndarray:
    """
    Given a list of arrays with potentially varying first dimension, pad to target shape.
    target_shape_tail excludes the batch dimension.
    pad_values should have shape target_shape_tail and provide default fill values.
    """
    n = len(list_of_arrays)
    out = np.zeros((n, *target_shape_tail), dtype=float)
    for i, arr in enumerate(list_of_arrays):
        slices = [slice(0, min(arr.shape[d], target_shape_tail[d])) for d in range(arr.ndim)]
        assign_to = tuple([i, *slices])
        out[assign_to] = arr[tuple(slices)]
        # Fill remaining with pad_values automatically by initialization to zeros, then overwrite quat w
        # For quaternions, caller will set default w=1 later if needed
    return out


def _write_hdf5(destination: Path, key_name: str, cuboid_centers: List[np.ndarray], cuboid_dims: List[np.ndarray], cuboid_quats: List[np.ndarray], cylinder_centers: List[np.ndarray], cylinder_radii: List[np.ndarray], cylinder_heights: List[np.ndarray], cylinder_quats: List[np.ndarray], trajectories: List[np.ndarray]) -> None:
    """
    Create an HDF5 file with persistent obstacle columns and an expert dataset with lengths and index.
    """
    from avoid_everything.dataset import Dataset  # late import to avoid heavy deps at module load
    from robofin.robots import Robot

    # Infer shapes
    n = len(trajectories)
    max_cuboids = max((x.shape[0] for x in cuboid_centers), default=1)
    max_cylinders = max((x.shape[0] for x in cylinder_centers), default=1)
    max_len = max(int(t.shape[0]) for t in trajectories)
    dof = int(trajectories[0].shape[1]) if trajectories else 0

    # Prepare arrays
    cub_centers_arr = np.zeros((n, max_cuboids, 3), dtype=float)
    cub_dims_arr = np.zeros((n, max_cuboids, 3), dtype=float)
    cub_quats_arr = np.zeros((n, max_cuboids, 4), dtype=float)

    cyl_centers_arr = np.zeros((n, max_cylinders, 3), dtype=float)
    cyl_radii_arr = np.zeros((n, max_cylinders, 1), dtype=float)
    cyl_heights_arr = np.zeros((n, max_cylinders, 1), dtype=float)
    cyl_quats_arr = np.zeros((n, max_cylinders, 4), dtype=float)

    expert_arr = np.zeros((n, max_len, dof), dtype=float)
    lengths = np.zeros((n,), dtype=int)

    # Fill
    for i in range(n):
        ccent = cuboid_centers[i]
        cdims = cuboid_dims[i]
        cquat = cuboid_quats[i]
        m1 = min(max_cuboids, ccent.shape[0])
        cub_centers_arr[i, :m1, :] = ccent[:m1, :]
        cub_dims_arr[i, :m1, :] = cdims[:m1, :]
        cub_quats_arr[i, :m1, :] = cquat[:m1, :]
        # Ensure valid quaternions for empty slots
        if m1 < max_cuboids:
            cub_quats_arr[i, m1:, 0] = 1.0

        ycent = cylinder_centers[i]
        yr = cylinder_radii[i]
        yh = cylinder_heights[i]
        yq = cylinder_quats[i]
        m2 = min(max_cylinders, ycent.shape[0])
        cyl_centers_arr[i, :m2, :] = ycent[:m2, :]
        cyl_radii_arr[i, :m2, :] = yr[:m2, :]
        cyl_heights_arr[i, :m2, :] = yh[:m2, :]
        cyl_quats_arr[i, :m2, :] = yq[:m2, :]
        if m2 < max_cylinders:
            cyl_quats_arr[i, m2:, 0] = 1.0

        t = trajectories[i]
        L = t.shape[0]
        expert_arr[i, :L, :] = t
        # pad by repeating last as expected by readers (they handle via indexing/edge mode later)
        if L < max_len:
            expert_arr[i, L:, :] = t[-1, :][None, :]
        lengths[i] = L

    # Write using Dataset helper to build index
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Create file and columns
    with h5py.File(str(destination), "w-") as f:
        f.create_dataset("cuboid_centers", data=cub_centers_arr)
        f.create_dataset("cuboid_dims", data=cub_dims_arr)
        f.create_dataset("cuboid_quaternions", data=cub_quats_arr)
        f.create_dataset("cylinder_centers", data=cyl_centers_arr)
        f.create_dataset("cylinder_radii", data=cyl_radii_arr)
        f.create_dataset("cylinder_heights", data=cyl_heights_arr)
        f.create_dataset("cylinder_quaternions", data=cyl_quats_arr)

    # Now open via Dataset to add expert and build index
    # Use GP7 URDF which exists in repo for indexing routines that require Robot
    urdf_path = "/workspace/assets/gp7/gp7.urdf"
    robot = Robot(urdf_path)
    with Dataset(robot, destination, mode="r+") as ds:
        ds.add_expert(key_name, expert_arr, lengths, build_index=True)


def _split_indices(n: int, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    if n <= 0:
        return [], [], []
    if n == 1:
        return idxs[:1], [], []
    if n == 2:
        return [idxs[0]], [idxs[1]], []
    # n >= 3
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * val_ratio)))  # use same ratio for test by default
    # leave rest for train
    n_train = max(1, n - n_val - n_test)
    # If rounding caused overflow, adjust
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _subset_list(lst: List[np.ndarray], indices: List[int]) -> List[np.ndarray]:
    return [lst[i] for i in indices]


def convert_usd_directory(
    input_dir: str,
    output_root: str,
    key_name: str = "global_solutions",
    max_files: int | None = None,
    max_trajectories: int | None = None,
):
    input_path = Path(input_dir)
    output_root_path = Path(output_root)
    (
        cub_centers,
        cub_dims,
        cub_quats,
        cyl_centers,
        cyl_radii,
        cyl_heights,
        cyl_quats,
        trajectories,
    ) = _collect_from_directory(input_path, max_files, max_trajectories)

    # filter out empty trajectories defensively and keep all lists aligned
    aligned = []
    for cc, cd, cq, yc, yr, yh, yq, t in zip(
        cub_centers, cub_dims, cub_quats, cyl_centers, cyl_radii, cyl_heights, cyl_quats, trajectories
    ):
        if t is None or t.size == 0:
            continue
        aligned.append((cc, cd, cq, yc, yr, yh, yq, t))
    if len(aligned) == 0:
        raise RuntimeError("No valid trajectories found in input directory")
    cub_centers, cub_dims, cub_quats, cyl_centers, cyl_radii, cyl_heights, cyl_quats, trajectories = map(list, zip(*aligned))

    n = len(trajectories)
    if n == 0:
        raise RuntimeError("No valid trajectories found in input directory")

    train_idx, val_idx, test_idx = _split_indices(n, 0.8, 0.1, seed=42)

    # Train
    if len(train_idx) > 0:
        train_path = output_root_path / "train" / "train.hdf5"
        _write_hdf5(
            train_path,
            key_name,
            _subset_list(cub_centers, train_idx),
            _subset_list(cub_dims, train_idx),
            _subset_list(cub_quats, train_idx),
            _subset_list(cyl_centers, train_idx),
            _subset_list(cyl_radii, train_idx),
            _subset_list(cyl_heights, train_idx),
            _subset_list(cyl_quats, train_idx),
            _subset_list(trajectories, train_idx),
        )

    # Val
    if len(val_idx) > 0:
        val_path = output_root_path / "val" / "val.hdf5"
        _write_hdf5(
            val_path,
            key_name,
            _subset_list(cub_centers, val_idx),
            _subset_list(cub_dims, val_idx),
            _subset_list(cub_quats, val_idx),
            _subset_list(cyl_centers, val_idx),
            _subset_list(cyl_radii, val_idx),
            _subset_list(cyl_heights, val_idx),
            _subset_list(cyl_quats, val_idx),
            _subset_list(trajectories, val_idx),
        )

    # Test
    if len(test_idx) > 0:
        test_path = output_root_path / "test" / "test.hdf5"
        _write_hdf5(
            test_path,
            key_name,
            _subset_list(cub_centers, test_idx),
            _subset_list(cub_dims, test_idx),
            _subset_list(cub_quats, test_idx),
            _subset_list(cyl_centers, test_idx),
            _subset_list(cyl_radii, test_idx),
            _subset_list(cyl_heights, test_idx),
            _subset_list(cyl_quats, test_idx),
            _subset_list(trajectories, test_idx),
        )


def main():
    parser = argparse.ArgumentParser(description="Convert OpenUSD dataset to Avoid-Everything HDF5 format")
    parser.add_argument("input_dir", type=str, help="Directory containing .usda files")
    parser.add_argument("output_root", type=str, help="Root output directory (will create train/ val/ test/)")
    parser.add_argument("--key", type=str, default="global_solutions", help="Expert key name to store trajectories under")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of USD files for a quick conversion test")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Limit total number of trajectories for a quick conversion test")
    args = parser.parse_args()

    convert_usd_directory(
        input_dir=args.input_dir,
        output_root=args.output_root,
        key_name=args.key,
        max_files=args.max_files,
        max_trajectories=args.max_trajectories,
    )


if __name__ == "__main__":
    main()


