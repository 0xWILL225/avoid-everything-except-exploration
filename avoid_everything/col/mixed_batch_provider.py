"""
Mixed batch sampling utilities for Cycle-of-Learning.

This module streams expert transitions from a DataLoader and mixes them with
actor transitions sampled from a replay sampler. It is designed to:

- Avoid storing the full expert dataset in memory
- Keep batches consistent with the dict[str, torch.Tensor] schema expected by training
- Support a pretraining phase (100% expert) and a CoL phase
  (1 / expert_fraction_denom expert, remainder from actor replay)
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from avoid_everything.col.replay import ReplayBuffer

from avoid_everything.utils.profiling import section

def _to_tensor(x: Any) -> torch.Tensor:
    """Convert a scalar or tensor-like to a torch.Tensor

    This helper standardizes values so that items returned from the expert
    loader can be collated uniformly.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")


def _collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample dictionaries into a batched dictionary.

    Expects each dictionary in `items` to share the same keys and that each
    value is a 1D or multi-D tensor representing a single sample. Stacking
    occurs along a new batch dimension at dim=0 for every key.

    Returns:
        A dict[str, torch.Tensor] with tensors of shape [B, ...].
    """
    if len(items) == 0:
        return {}
    keys = items[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [_to_tensor(it[k]) for it in items]
        batch[k] = torch.stack(vals, dim=0)
    return batch


class MixedBatchProvider:
    """
    Streams expert transitions from a DataLoader and mixes them with actor samples
    drawn from a replay sampler.

    The provider does not pre-load or store the entire expert dataset; instead it
    maintains a small pool to bridge DataLoader batch boundaries. This
    enables precise batch assembly with minimal memory while staying compatible
    with typical PyTorch DataLoader usage.
    """

    def __init__(
        self,
        expert_loader: Iterable[Any],
        actor_replay: ReplayBuffer,
        *,
        key_renames: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a mixed-batch provider.

        Parameters:
            expert_loader: An iterable (e.g., a DataLoader) that yields batched
                dictionaries with consistent keys. These represent expert
                transitions and must match the schema expected by training.
            key_renames: Optional mapping to rename incoming keys (for example,
                {"supervision": "next_configuration"}). Applied per-sample when
                items are pulled from the expert loader.
        """
        self._expert_loader = expert_loader
        self._expert_iter = iter(self._expert_loader)
        self._expert_pool: List[Dict[str, torch.Tensor]] = []
        self._key_renames = key_renames or {}
        self.actor_replay = actor_replay

    def _split_batch_to_items(
        self,
        expert_batch: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Split a batched expert dictionary into a list of per-sample dicts.

        Each tensor value in the input dict is sliced along dim=0 to produce
        one dictionary per sample. Key renames are applied.
        """
        # Determine batch size by first tensor entry
        size = None
        for v in expert_batch.values():
            if isinstance(v, torch.Tensor):
                size = v.shape[0]
                break
        if size is None:
            raise ValueError("Batch must contain at least one tensor with a batch dimension")
        items: List[Dict[str, torch.Tensor]] = []
        for i in range(size):
            item: Dict[str, torch.Tensor] = {}
            for k, v in expert_batch.items():
                vv = v[i] if isinstance(v, torch.Tensor) else _to_tensor(v)
                # Apply key mapping if needed (e.g., supervision -> next_configuration)
                dst_key = self._key_renames.get(k, k)
                item[dst_key] = vv.detach() if isinstance(vv, torch.Tensor) else _to_tensor(vv)
            items.append(item)
        return items

    def _fill_expert_pool(self, required_samples: int) -> int:
        """Fill the internal expert pool up to `required_samples` items.

        Pulls batches from the expert iterator and appends per-sample entries
        to the pool until the requested minimum is satisfied. The iterator is
        reset upon exhaustion (StopIteration), providing an infinite stream if
        desired.

        Returns:
            The number of data loader iterations used to fill the pool.
        """
        data_loader_iterations = 0
        while len(self._expert_pool) < required_samples:
            try:
                batch = next(self._expert_iter)
            except StopIteration:
                self._expert_iter = iter(self._expert_loader)
                batch = next(self._expert_iter)
            data_loader_iterations += 1
            self._expert_pool.extend(self._split_batch_to_items(batch))
        return data_loader_iterations

    def _pop_expert(self, expert_samples: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Remove and return `expert_samples` items from the pool as a batched dict.

        Parameters:
            expert_samples: Number of expert samples to return. If zero, returns an empty dict.

        Returns:
            A dictionary with tensors of shape [expert_samples, ...] and the 
            number of data loader iterations used to fill the pool.
        """
        if expert_samples <= 0:
            return {}, 0
        data_loader_iterations = self._fill_expert_pool(expert_samples)
        items = self._expert_pool[:expert_samples]
        del self._expert_pool[:expert_samples]
        return _collate(items), data_loader_iterations

    def sample(
        self,
        total_batch_size: int,
        expert_fraction: float,
        pretraining: bool,
        device: torch.device | str | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Draw a mixed batch of expert and actor transitions.

        During pretraining (``pretraining=True``) the batch is 100% expert. In
        CoL RL-finetuning mode (``pretraining=False``), this returns 
        expert_fraction of the batch from the expert stream and the 
        remainder from the actor replay.

        Parameters:
            total_batch_size: Exact batch size to return.
            expert_fraction: Fraction of samples to sample from expert loader during fine-tuning.
            pretraining: If True, returns only expert samples.
            device: Optional target device for actor samples (falls back to expert batch device).

        Returns:
            A dictionary of tensors representing a full batch and the number of 
            expert data loader iterations used to fill the batch.
        """
        if pretraining:
            try:
                b = next(self._expert_iter)
            except StopIteration:
                self._expert_iter = iter(self._expert_loader)
                b = next(self._expert_iter)
            return b, 1

        assert expert_fraction >= 0.0 and expert_fraction <= 1.0
        n_expert_samples = int(round(total_batch_size * expert_fraction))
        n_actor_samples = total_batch_size - n_expert_samples

        timings = {}
        with section("MBP._pop_expert", timings):
            expert_batch, data_loader_iterations = self._pop_expert(n_expert_samples) if n_expert_samples > 0 else {}
        
        # choose a target device based on expert batch (if present)
        if device is None:
            if expert_batch:
                for v in expert_batch.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
        with section("MBP.actor_replay.sample", timings):
            actor_batch  = self.actor_replay.sample(n_actor_samples, device=device) if n_actor_samples > 0 else {}

        if n_expert_samples == 0:
            return actor_batch, 0
        if n_actor_samples == 0:
            return expert_batch, data_loader_iterations

        with section("MBP.merge_batches", timings):
            common = expert_batch.keys() & actor_batch.keys()
            merged = {k: torch.cat([expert_batch[k], actor_batch[k]], dim=0) for k in common}

        with section("MBP.permute_batch", timings):
            any_tensor = next(iter(merged.values()))
            perm = torch.randperm(total_batch_size, device=any_tensor.device)
            for k in merged:
                merged[k] = merged[k][perm]

        print(timings)
        return merged, data_loader_iterations
