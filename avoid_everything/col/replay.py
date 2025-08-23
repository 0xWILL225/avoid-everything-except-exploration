import numpy as np
import torch
from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler
from avoid_everything.data_loader import Base

class ReplayBuffer:
    """
    Replay buffer for the COL algorithm.
    Stores minimal transitions (idx, q, a, q_next, r, done) and reconstructs the 
    scene state using the idx when sampling.
    """

    def __init__(self, capacity: int, urdf_path: str, num_robot_points: int, num_target_points: int, dataset: Base):
        self.capacity = int(capacity)
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.robot = Robot(urdf_path)
        self.robot_sampler = TorchRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=self.num_target_points,
            use_cache=True,
            with_base_link=True,
        )
        self.dataset = dataset # reference to the dataset (for extracting scene info)
        self.idx   = np.zeros((capacity,), dtype=np.int32) # indices in the dataset (for reconstructing scene)
        self.q     = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized configurations
        self.a     = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized actions
        self.qnext = np.zeros((capacity, self.robot.MAIN_DOF), dtype=np.float16) # normalized next configurations
        self.r     = np.zeros((capacity, 1), dtype=np.float32) # rewards
        self.done  = np.zeros((capacity, 1), dtype=np.uint8) # done flags
        self.ptr = 0
        self.full = False

    def push(self, idx: np.ndarray, q: np.ndarray, a: np.ndarray, q_next: np.ndarray, r: np.ndarray, done: np.ndarray):
        """
        Push a batch of transitions to the replay buffer. Implements a circular
        buffer (FIFO).

        Args:
            idx: (B,) int32
            q: (B, DOF) float16
            a: (B, DOF) float16
            q_next: (B, DOF) float16
            r: (B, 1) float32
            done: (B, 1) uint8
        """
        B = idx.shape[0]
        end = self.ptr + B
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.ptr = end % self.capacity
            self.full = self.full or self.ptr == 0
        else:
            # wrap-around
            first = self.capacity - self.ptr
            self.push(idx[:first], q[:first], a[:first], q_next[:first], r[:first], done[:first])
            self.push(idx[first:], q[first:], a[first:], q_next[first:], r[first:], done[first:])
            return
        self.idx[sl] = idx
        self.q[sl] = q
        self.a[sl] = a
        self.qnext[sl] = q_next
        self.r[sl] = r
        self.done[sl] = done

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        """
        n = len(self)
        assert n >= batch_size
        ids = np.random.randint(0, n, size=batch_size)

        idx  = torch.from_numpy(self.idx[ids]).long()                  # [B]
        q    = torch.from_numpy(self.q[ids]).to(torch.float32)         # [B, DOF]
        a    = torch.from_numpy(self.a[ids]).to(torch.float32)         # [B, DOF]
        qn   = torch.from_numpy(self.qnext[ids]).to(torch.float32)     # [B, DOF]
        r    = torch.from_numpy(self.r[ids]).to(torch.float32)         # [B, 1]
        done = torch.from_numpy(self.done[ids]).to(torch.float32)      # [B, 1]

        # deduplicate scene fetches, then map back to batch order
        idx_np = idx.numpy()
        uniq, inv = np.unique(idx_np, return_inverse=True)  # uniq[k] -> scene k, inv[i] = which uniq row
        scenes = [self.dataset.scene_by_idx(int(u)) for u in uniq]  # each is CPU dict of tensors

        # stack per-sample tensors in batch order using inv mapping
        def stack_from_scenes(key):
            return torch.stack([scenes[j][key] for j in inv], dim=0)

        cuboid_centers   = stack_from_scenes("cuboid_centers")
        cuboid_dims      = stack_from_scenes("cuboid_dims")
        cuboid_quats     = stack_from_scenes("cuboid_quats")
        cylinder_centers = stack_from_scenes("cylinder_centers")
        cylinder_radii   = stack_from_scenes("cylinder_radii")
        cylinder_heights = stack_from_scenes("cylinder_heights")
        cylinder_quats   = stack_from_scenes("cylinder_quats")
        target_position  = stack_from_scenes("target_position")
        target_orientation = stack_from_scenes("target_orientation")
        scene_points     = stack_from_scenes("scene_points")    # [B, N_scene, 3]
        target_points    = stack_from_scenes("target_points")   # [B, N_target, 3]

        q_unn = self.robot.unnormalize_joints(q)
        robot_points = self.robot_sampler.sample(q_unn)[..., :3]  # [B, N_robot, 3]
        assert robot_points.dim() == 3, "Expecting batch dimension"

        pc = torch.cat([robot_points, scene_points, target_points], dim=1)  # [B, N_total, 3]
        B = pc.size(0)
        labels = torch.cat([
            torch.zeros((B, robot_points.size(1), 1)),
            torch.ones((B, scene_points.size(1), 1)),
            2*torch.ones((B, target_points.size(1), 1)),
        ], dim=1)

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
            "is_expert": torch.zeros(B, 1, dtype=torch.float32),
        }
        return batch
