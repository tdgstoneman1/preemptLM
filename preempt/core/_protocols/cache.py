# from __future__ import annotations

# from typing import Protocol, runtime_checkable

# from preempt.core.identity import ExpertKey
# from preempt.core.protocols.expert_bank import ExpertPayload


# # TODO add `tensors` method to be consistent with `MlxExpertCache`?
# @runtime_checkable
# class IExpertCache(Protocol):
#     """Interface for caching experts in memory."""

#     def install(
#         self, key: ExpertKey, payload: ExpertPayload
#     ) -> None:  # TODO to cache_expert
#         """Decodes `payload` into weights and caches them under `key`."""
#         ...

#     def evict(self, key: ExpertKey) -> None:  # TODO rename to `drop`
#         """Drops the tensors mapped to `key` from the cache."""
#         ...

#     def is_resident(self, key: ExpertKey) -> bool:  # TODO rename to `is_cached`
#         """Checks if `key` is in the cache."""
#         ...

#     def size(self) -> int:
#         """Total size of the cache in bytes."""
#         ...
