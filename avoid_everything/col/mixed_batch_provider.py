"""
Mixed batch sampling utilities for Cycle-of-Learning.

This module streams expert transitions from a DataLoader and mixes them with
agent transitions sampled from a replay sampler. It is designed to:

- Avoid storing the full expert dataset in memory
- Keep batches consistent with the dict[str, torch.Tensor] schema expected by training
- Support a pretraining phase (100% expert) and a CoL phase
  (1 / expert_denominator expert, remainder from agent replay)
"""

from typing import Any, Dict, Iterable, List, Optional
import torch
from avoid_everything.col.replay import ReplayBuffer

# class ReplaySampler(Protocol):
#     """Minimal interface required from an agent replay buffer/sampler.

#     Implementations must return a batch of transitions as a dictionary of
#     tensors with a consistent first (batch) dimension across values.

#     Parameters:
#         batch_size: Number of samples to draw.

#     Returns:
#         A dict[str, torch.Tensor] representing a batch of transitions.
#     """

#     def sample(self, batch_size: int) -> Dict[str, torch.Tensor]: ...


def _to_tensor(x: Any) -> torch.Tensor:
    """Convert a scalar or tensor-like to a torch.Tensor (on CPU).

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
    Streams expert transitions from a DataLoader and mixes them with agent samples
    drawn from a replay sampler.

    The provider does not pre-load or store the entire expert dataset; instead it
    maintains a small CPU-side pool to bridge DataLoader batch boundaries. This
    enables precise batch assembly with minimal memory while staying compatible
    with typical PyTorch DataLoader usage.
    """

    def __init__(
        self,
        expert_loader: Iterable[Any],
        agent_replay: ReplayBuffer,
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
        self.agent_replay = agent_replay

    def _split_batch_to_items(self, expert_batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split a batched expert dictionary into a list of per-sample dicts.

        Each tensor value in the input dict is sliced along dim=0 to produce
        one dictionary per sample. Key renames are applied, and tensors are
        stored on CPU to avoid holding GPU memory in the pool.
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
                item[dst_key] = vv.detach().cpu() if isinstance(vv, torch.Tensor) else _to_tensor(vv)
            items.append(item)
        return items

    def _fill_expert_pool(self, required_samples: int) -> None:
        """Fill the internal expert pool up to `required_samples` items.

        Pulls batches from the expert iterator and appends per-sample entries
        to the pool until the requested minimum is satisfied. The iterator is
        reset upon exhaustion (StopIteration), providing an infinite stream if
        desired.
        """
        while len(self._expert_pool) < required_samples:
            try:
                batch = next(self._expert_iter)
            except StopIteration:
                self._expert_iter = iter(self._expert_loader)
                batch = next(self._expert_iter)
            self._expert_pool.extend(self._split_batch_to_items(batch))

    def _pop_expert(self, expert_samples: int) -> Dict[str, torch.Tensor]:
        """Remove and return `expert_samples` items from the pool as a batched dict.

        Parameters:
            expert_samples: Number of expert samples to return. If zero, returns an empty dict.

        Returns:
            A dictionary with tensors of shape [expert_samples, ...].
        """
        if expert_samples <= 0:
            return {}
        self._fill_expert_pool(expert_samples)
        items = self._expert_pool[:expert_samples]
        del self._expert_pool[:expert_samples]
        return _collate(items)

    def sample(
        self,
        total_batch_size: int,
        expert_denominator: int,
        *,
        pretraining: bool,
    ) -> Dict[str, torch.Tensor]:
        """Draw a mixed batch of expert and agent transitions.

        During pretraining (``pretraining=True``) the batch is 100% expert. In
        CoL RL-finetuning mode (``pretraining=False``), this returns 
        1/expert_denominator of the batch from the expert stream and the 
        remainder from the agent replay.

        Parameters:
            total_batch_size: Exact batch size to return.
            expert_denominator: Denominator controlling expert share after
                pretraining (e.g., 4 -> 25% expert, 75% agent).
            pretraining: If True, returns only expert samples.

        Returns:
            A dictionary of tensors representing a full batch, with keys
            identical to those produced by the expert loader (renamed per
            `key_renames`), and matching shapes across sources.
        """
        assert expert_denominator >= 1
        # Always ensure exact total by allocating expert fraction then filling rest with agent
        n_expert_samples = total_batch_size if pretraining else (total_batch_size // expert_denominator)
        # Put any remainder into agent portion so sums to exactly total_batch_size
        n_agent_samples = total_batch_size - n_expert_samples

        expert_batch = self._pop_expert(n_expert_samples) if n_expert_samples > 0 else {}
        agent_batch  = self.agent_replay.sample(n_agent_samples) if n_agent_samples > 0 else {}

        if n_expert_samples == 0:
            agent_batch["is_expert"] = torch.zeros(n_agent_samples, 1)
            return agent_batch
        if n_agent_samples == 0:
            expert_batch["is_expert"] = torch.ones(n_expert_samples, 1)
            return expert_batch

        common = expert_batch.keys() & agent_batch.keys()
        merged = {k: torch.cat([expert_batch[k], agent_batch[k]], dim=0) for k in common}
        is_expert = torch.cat([torch.ones(n_expert_samples,1), torch.zeros(n_agent_samples,1)], dim=0)
        perm = torch.randperm(total_batch_size)
        for k in merged:
            merged[k] = merged[k][perm]
        merged["is_expert"] = is_expert[perm]
        return merged
