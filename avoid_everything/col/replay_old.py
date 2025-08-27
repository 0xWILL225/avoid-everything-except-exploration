import numpy as np
import torch
from termcolor import cprint

from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler
from avoid_everything.data_loader import Base


class ReplayBuffer:
    """
    Replay buffer for the COL algorithm.
    Stores minimal transitions (idx, q, a, q_next, r, done) and reconstructs the 
    scene state using the idx when sampling.
    """

    def __init__(self, capacity: int, urdf_path: str, robot_dof: int, num_robot_points: int, num_target_points: int, dataset: Base, device: torch.device | str | None = None):
        self.capacity = int(capacity)
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.urdf_path = urdf_path
        # self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.robot = None
        self.robot_sampler = None
        self.dataset = dataset # reference to the dataset (for extracting scene info)
        # self.idx   = np.zeros((capacity,), dtype=np.int64) # indices in the dataset (for reconstructing scene)
        # self.q     = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized configurations
        # self.a     = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized actions
        # self.qnext = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized next configurations
        # self.r     = np.zeros((capacity, 1), dtype=np.float32) # rewards
        # self.done  = np.zeros((capacity, 1), dtype=np.uint8) # done flags
        self.idx   = torch.empty(capacity, dtype=torch.int64,  pin_memory=True)
        self.q     = torch.empty(capacity, robot_dof, dtype=torch.float16, pin_memory=True)
        self.a     = torch.empty_like(self.q)
        self.qnext = torch.empty_like(self.q)
        self.r     = torch.empty(capacity, 1, dtype=torch.float32, pin_memory=True)
        self.done  = torch.empty(capacity, 1, dtype=torch.uint8,  pin_memory=True)
        self.ptr = 0
        self.full = False

    # def set_device(self, device: torch.device | str):
    #     """Update target device and rebuild robot/sampler accordingly."""
    #     cprint(f"Warning: ReplayBuffer changing device to {str(device)}", "yellow")
    #     self.device = torch.device(device)
    #     self.robot = Robot(self.urdf_path, device=self.device)
    #     self.robot_sampler = TorchRobotSampler(
    #         self.robot,
    #         num_robot_points=self.num_robot_points,
    #         num_eef_points=self.num_target_points,
    #         use_cache=True,
    #         with_base_link=True,
    #         device=self.device,
    #     )

    def _ensure_robot_sampler(self, device: torch.device):
        if self.robot_sampler is None:
            self.robot = Robot(self.urdf_path, device=device)
            self.robot_sampler = TorchRobotSampler(
                self.robot,
                num_robot_points=self.num_robot_points,
                num_eef_points=self.num_target_points,
                with_base_link=True,
                use_cache=True,
                device=device,
            )
        assert self.robot_sampler.device == device, "Sampler device mismatch"

    # def push(self, idx: np.ndarray, q: np.ndarray, a: np.ndarray, q_next: np.ndarray, r: np.ndarray, done: np.ndarray):
    #     """
    #     Push a batch of transitions to the replay buffer. Implements a circular
    #     buffer (FIFO).

    #     Args:
    #         idx: (B,) int32
    #         q: (B, DOF) float16
    #         a: (B, DOF) float16
    #         q_next: (B, DOF) float16
    #         r: (B, 1) float32
    #         done: (B, 1) uint8
    #     """
    #     B = idx.shape[0]
    #     end = self.ptr + B
    #     if end <= self.capacity:
    #         sl = slice(self.ptr, end)
    #         self.ptr = end % self.capacity
    #         self.full = self.full or self.ptr == 0
    #     else:
    #         # wrap-around
    #         first = self.capacity - self.ptr
    #         self.push(idx[:first], q[:first], a[:first], q_next[:first], r[:first], done[:first])
    #         self.push(idx[first:], q[first:], a[first:], q_next[first:], r[first:], done[first:])
    #         return
    #     self.idx[sl] = idx
    #     self.q[sl] = q
    #     self.a[sl] = a
    #     self.qnext[sl] = q_next
    #     self.r[sl] = r
    #     self.done[sl] = done

    def push(self, idx, q, a, q_next, r, done):
        # expect tensors on GPU or CPU; move to CPU pinned asynchronously
        idx    = idx.to('cpu', non_blocking=True, dtype=torch.int64)
        q      = q.to('cpu', non_blocking=True, dtype=torch.float16)
        a      = a.to('cpu', non_blocking=True, dtype=torch.float16)
        q_next = q_next.to('cpu', non_blocking=True, dtype=torch.float16)
        r      = r.to('cpu', non_blocking=True, dtype=torch.float32)
        done   = done.to('cpu', non_blocking=True, dtype=torch.uint8)

        B = idx.shape[0]
        end = self.ptr + B
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.idx[sl].copy_(idx, non_blocking=True)
            self.q[sl].copy_(q, non_blocking=True)
            self.a[sl].copy_(a, non_blocking=True)
            self.qnext[sl].copy_(q_next, non_blocking=True)
            self.r[sl].copy_(r, non_blocking=True)
            self.done[sl].copy_(done, non_blocking=True)
            self.ptr = end % self.capacity
            self.full = self.full or self.ptr == 0
        else:
            first = self.capacity - self.ptr
            self.push(idx[:first], q[:first], a[:first], q_next[:first], r[:first], done[:first])
            self.push(idx[first:], q[first:], a[first:], q_next[first:], r[first:], done[first:])

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        """
        # resolve target device
        # target_device = torch.device(device) if device is not None else self.device
        # if target_device != self.device:
        #     # keep logic simple: rebuild on the requested device
        #     self.set_device(target_device)
        # n = len(self)
        # assert n >= batch_size
        # ids = np.random.randint(0, n, size=batch_size)

        # idx  = torch.from_numpy(self.idx[ids]).long().to(target_device, non_blocking=True)   # [B]
        # q    = torch.from_numpy(self.q[ids]).to(torch.float32).to(target_device, non_blocking=True)   # [B, DOF]
        # a    = torch.from_numpy(self.a[ids]).to(torch.float32).to(target_device, non_blocking=True)   # [B, DOF]
        # qn   = torch.from_numpy(self.qnext[ids]).to(torch.float32).to(target_device, non_blocking=True) # [B, DOF]
        # r    = torch.from_numpy(self.r[ids]).to(torch.float32).to(target_device, non_blocking=True)    # [B, 1]
        # done = torch.from_numpy(self.done[ids]).to(torch.float32).to(target_device, non_blocking=True) # [B, 1]

        # deduplicate scene fetches, then map back to batch order
        # uniq, inv = np.unique(self.idx[ids], return_inverse=True)  # uniq[k] -> scene k, inv[i] = which uniq row
        # scenes = [self.dataset.scene_by_idx(int(u)) for u in uniq]  # each is CPU dict of tensors

        # # stack per-sample tensors in batch order using inv mapping
        # def stack_from_scenes(key):
        #     return torch.stack([scenes[j][key] for j in inv], dim=0).to(device, non_blocking=True)

        # cuboid_centers   = stack_from_scenes("cuboid_centers")
        # cuboid_dims      = stack_from_scenes("cuboid_dims")
        # cuboid_quats     = stack_from_scenes("cuboid_quats")
        # cylinder_centers = stack_from_scenes("cylinder_centers")
        # cylinder_radii   = stack_from_scenes("cylinder_radii")
        # cylinder_heights = stack_from_scenes("cylinder_heights")
        # cylinder_quats   = stack_from_scenes("cylinder_quats")
        # target_position  = stack_from_scenes("target_position")
        # target_orientation = stack_from_scenes("target_orientation")
        # scene_points     = stack_from_scenes("scene_points")    # [B, N_scene, 3]
        # target_points    = stack_from_scenes("target_points")   # [B, N_target, 3]

        # labels = torch.cat([
        #     torch.zeros((B, robot_points.size(1), 1), device=device),
        #     torch.ones((B, scene_points.size(1), 1), device=device),
        #     2*torch.ones((B, target_points.size(1), 1), device=device),
        # ], dim=1)

        device = torch.device(device) if device is not None else torch.device('cpu')
        n = len(self)
        ids = torch.randint(n, (batch_size,), device='cpu')

        idx  = self.idx[ids].to(device, non_blocking=True)
        q    = self.q[ids].to(dtype=torch.float32, device=device, non_blocking=True)
        a    = self.a[ids].to(dtype=torch.float32, device=device, non_blocking=True)
        qn   = self.qnext[ids].to(dtype=torch.float32, device=device, non_blocking=True)
        r    = self.r[ids].to(device, non_blocking=True)
        done = self.done[ids].to(dtype=torch.float32, device=device, non_blocking=True)

        uniq, inv = torch.unique(idx, sorted=True, return_inverse=True)  # inv on device
        sc = self.dataset.batch_scenes_by_idx(uniq.cpu())                # CPU pinned tensors
        inv_cpu = inv.cpu()
        def to_dev(x):
            return x[inv_cpu].to(device, non_blocking=True)  # CPU index → CPU slice → async H2D

        cuboid_centers    = to_dev(sc["cuboid_centers"])
        cuboid_dims       = to_dev(sc["cuboid_dims"])
        cuboid_quats      = to_dev(sc["cuboid_quats"])
        cylinder_centers  = to_dev(sc["cylinder_centers"])
        cylinder_radii    = to_dev(sc["cylinder_radii"])
        cylinder_heights  = to_dev(sc["cylinder_heights"])
        cylinder_quats    = to_dev(sc["cylinder_quats"])
        target_position   = to_dev(sc["target_position"])
        target_orientation= to_dev(sc["target_orientation"])
        scene_points      = to_dev(sc["scene_points"])
        target_points     = to_dev(sc["target_points"])

        assert self.robot is not None
        assert self.robot_sampler is not None
        q_unn = self.robot.unnormalize_joints(q)
        self._ensure_robot_sampler(q_unn.device)
        robot_points = self.robot_sampler.sample(q_unn)[..., :3]  # [B, N_robot, 3]
        assert robot_points.dim() == 3, "Expecting batch dimension"

        pc = torch.cat([robot_points, scene_points, target_points], dim=1)  # [B, N_total, 3]
        B = pc.size(0)
        if not hasattr(self, "_labels"):
            R, S, T = self.num_robot_points, self.dataset.num_obstacle_points, self.num_target_points
            self._labels = torch.cat([torch.zeros(R,1), torch.ones(S,1), 2*torch.ones(T,1)], dim=0).pin_memory()

        labels = self._labels.expand(B, -1, -1).to(device, non_blocking=True)

        batch = {
            "idx": idx,
            "configuration": q,
            "action": a,
            "next_configuration": qn,
            "reward": r,
            "done": done,
            "point_cloud": pc,
            "point_cloud_labels": labels,
            "cuboid_centers": cuboid_centers,
            "cuboid_dims": cuboid_dims,
            "cuboid_quats": cuboid_quats,
            "cylinder_centers": cylinder_centers,
            "cylinder_radii": cylinder_radii,
            "cylinder_heights": cylinder_heights,
            "cylinder_quats": cylinder_quats,
            "target_position": target_position,
            "target_orientation": target_orientation,
            "is_expert": torch.zeros(B, 1, dtype=torch.float32, device=device),
        }
        return batch
