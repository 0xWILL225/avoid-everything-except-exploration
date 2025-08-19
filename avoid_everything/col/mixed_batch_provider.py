from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Protocol

import torch


class ReplaySampler(Protocol):
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        ...


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")


def _collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(items) == 0:
        return {}
    keys = items[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [_to_tensor(it[k]) for it in items]
        batch[k] = torch.stack(vals, dim=0)
    return batch


class ExpertReplayBuffer:
    """
    A simple ring-buffer for expert transitions stored as dictionaries of tensors.
    Intended for large capacities; not memory-optimized but simple and robust.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: List[Dict[str, torch.Tensor]] = []
        self.pos: int = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, item: Dict[str, torch.Tensor]) -> None:
        # Store on CPU to avoid holding GPU memory; tensors are moved later by the caller
        cpu_item = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else _to_tensor(v)) for k, v in item.items()}
        if len(self.storage) < self.capacity:
            self.storage.append(cpu_item)
        else:
            self.storage[self.pos] = cpu_item
        self.pos = (self.pos + 1) % self.capacity

    def add_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        # Split by first dimension and push as individual items
        size = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                size = v.shape[0]
                break
        if size is None:
            raise ValueError("Batch must contain at least one tensor with a batch dimension")
        for i in range(size):
            item = {k: (v[i] if isinstance(v, torch.Tensor) else _to_tensor(v)) for k, v in batch.items()}
            self.add(item)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert len(self.storage) > 0, "ExpertReplayBuffer is empty"
        indices = [random.randrange(0, len(self.storage)) for _ in range(batch_size)]
        items = [self.storage[i] for i in indices]
        return _collate(items)


class MixedBatchProvider:
    """
    Provides mixed batches composed of expert transitions (from an expert replay buffer)
    and agent transitions (from a user-provided sampler with a .sample API).
    """

    def __init__(
        self,
        expert_loader: Iterable[Dict[str, torch.Tensor]],
        *,
        expert_capacity: int = 1_000_000,
        preload_max_items: Optional[int] = None,
    ) -> None:
        self.expert_buffer = ExpertReplayBuffer(expert_capacity)
        self._preload_expert(expert_loader, preload_max_items)
        self.agent_replay: Optional[ReplaySampler] = None

    def _preload_expert(
        self, expert_loader: Iterable[Dict[str, torch.Tensor]], preload_max_items: Optional[int]
    ) -> None:
        loaded = 0
        for batch in expert_loader:
            # Accept either individual dict items or batched dicts
            if isinstance(next(iter(batch.values())), torch.Tensor) and next(iter(batch.values())).dim() > 0:
                self.expert_buffer.add_batch(batch)
                # Infer batch size
                bsz = next(iter(batch.values())).shape[0]
                loaded += bsz
            else:
                self.expert_buffer.add(batch)  # type: ignore[arg-type]
                loaded += 1
            if preload_max_items is not None and loaded >= preload_max_items:
                break

    def set_agent_replay(self, agent_replay: ReplaySampler) -> None:
        self.agent_replay = agent_replay

    def sample(self, total_batch_size: int, expert_ratio: float) -> Dict[str, torch.Tensor]:
        assert 0.0 <= expert_ratio <= 1.0
        b = int(total_batch_size)
        b_expert = int(math.ceil(expert_ratio * b))
        b_agent = b - b_expert

        expert_batch = self.expert_buffer.sample(b_expert) if b_expert > 0 else {}

        agent_batch: Dict[str, torch.Tensor]
        if b_agent > 0:
            assert self.agent_replay is not None, "Agent replay not set"
            agent_batch = self.agent_replay.sample(b_agent)
        else:
            agent_batch = {}

        if b_expert == 0:
            return agent_batch
        if b_agent == 0:
            return expert_batch

        # Merge and shuffle
        merged: Dict[str, torch.Tensor] = {}
        for k in expert_batch.keys():
            merged[k] = torch.cat([expert_batch[k], agent_batch[k]], dim=0)
        perm = torch.randperm(b)
        for k in merged.keys():
            merged[k] = merged[k][perm]
        return merged


