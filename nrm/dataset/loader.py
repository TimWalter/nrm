import re
import random
import bisect
from pathlib import Path

import zarr
import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Int64, Bool

import nrm.dataset.se3 as se3


class Dataset:
    """
    Dataset class to load reachability manifold estimations using Zarr.
    The hierarchy is Keys -> Chunks -> Batches.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, batch_size: int, shuffle: bool, kind: str):
        """
        Initialise dataset.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle batches.
            kind: String indicating which dataset to load.
        """

        self.shuffle = shuffle
        self.root = zarr.open(Path(__file__).parent.parent.parent / 'data' / kind, mode="r")

        file_indices = sorted([
            int(match.group(1))
            for k in self.root.array_keys()
            if (match := re.search(r'^(\d+)_samples$', k))
        ])
        self.morphologies = torch.cat([torch.from_numpy(self.root[f"{idx}_morphologies"][:]) for idx in file_indices], dim=0)
        self.keys = [f"{idx}_samples" for idx in file_indices]

        self.chunk_size = self.root[self.keys[0]].chunks[0]
        self.chunk_offsets = [0]
        self.batch_size = batch_size
        self.batch_offsets = [0]
        for k in self.keys:
            num_samples = self.root[k].shape[0]
            self.chunk_offsets += [self.chunk_offsets[-1] + num_samples // self.chunk_size]
            self.batch_offsets += [self.batch_offsets[-1] + num_samples // self.batch_size]
        self.num_chunks = self.chunk_offsets[-1]
        self.num_batches = self.batch_offsets[-1]

        self.current_chunk_idx = None
        self.current_chunk = None

        self.current_batch_idx = None

        assert self.chunk_size % self.batch_size == 0, \
            f"Chunk size ({self.chunk_size}) must be a multiple of batch size ({self.batch_size})"

    @jaxtyped(typechecker=beartype)
    def __len__(self) -> int:
        return self.num_batches

    @jaxtyped(typechecker=beartype)
    def _cache_chunk(self, chunk_idx: int):
        """
        Loads a chunk into memory.
        Chunks are the atoms of reading and chunk size is optimised for loading speed from disk.

        Args:
            chunk_idx: Index of chunk to load.
        """

        key_idx = bisect.bisect_right(self.chunk_offsets, chunk_idx) - 1
        local_chunk_idx = chunk_idx - self.chunk_offsets[key_idx]

        key = self.keys[key_idx]
        start = local_chunk_idx * self.chunk_size
        end = start + self.chunk_size

        self.current_chunk = torch.from_numpy(self.root[key][start:end])
        self.current_chunk_idx = chunk_idx

    @jaxtyped(typechecker=beartype)
    def _get_batch(self, batch_idx: int):
        """
        Fetch a batch from the dataset. Caches the respective chunk.

        Args:
            batch_idx: Index of batch to fetch.
        Returns:
            Batch
        """

        chunk_idx = batch_idx * self.batch_size // self.chunk_size

        if chunk_idx != self.current_chunk_idx:
            self._cache_chunk(chunk_idx)

        local_batch_idx = batch_idx % (self.chunk_size // self.batch_size)
        start = local_batch_idx * self.batch_size
        end = start + self.batch_size

        batch = self.current_chunk[start:end]

        return batch

    @jaxtyped(typechecker=beartype)
    def _get_morph(self, morph_id: Int[Tensor, " batch_size"]) -> Float[Tensor, " batch dof 3"]:
        """
        Retrieve the morphology for a morphology index.

        Args:
            morph_id: Morphology index.
        Returns:
            Morphology.
        """
        # TODO once we switch to various DOFs
        morph = self.morphologies[morph_id]
        dof = (morph[0].abs().sum(dim=1) != 0).sum().item()
        morph = morph[:, :dof, :]
        # mask = (morph.abs().sum(dim=2) != 0)
        # dofs = mask.sum(dim=1)
        # flat = morph[mask]
        # split_sizes = dofs.tolist()
        # chunks = list(torch.split(flat, split_sizes))
        # morph = torch.nested.nested_tensor(chunks, layout=torch.jagged)
        return morph

    @jaxtyped(typechecker=beartype)
    def _get_pose(self, batch_chunk) -> Float[Tensor, " batch 9"]:
        """
       Retrieve the pose from the batch chunk.

       Args:
           batch_chunk: Batch chunk.
       Returns:
           Pose
       """
        pass

    def __getitem__(self, batch_idx: int) -> tuple[
        Float[Tensor, "batch dof 3"],
        Float[Tensor, "batch 9"],
        Bool[Tensor, "batch"]
    ]:
        batch = self._get_batch(batch_idx)

        morph = self._get_morph(batch[:, 0].long())
        pose = self._get_pose(batch[:, 1:-1])
        label = batch[:, -1].bool()

        return morph.pin_memory(), pose.pin_memory(), label.pin_memory()

    def __iter__(self):
        """
        Iterator that shuffles efficiently two-layered.
        """
        chunk_order = list(range(self.num_chunks))
        if self.shuffle:
            random.shuffle(chunk_order)

        for chunk_idx in chunk_order:
            self._cache_chunk(chunk_idx)

            batches_per_chunk = self.chunk_size // self.batch_size
            batch_order = list(range(batches_per_chunk))
            if self.shuffle:
                random.shuffle(batch_order)
            for local_batch_idx in batch_order:
                self.current_batch_idx = local_batch_idx + chunk_idx * (self.chunk_size // self.batch_size)
                yield self[self.current_batch_idx]

    def get_random_batch(self) -> tuple[
        Float[Tensor, "batch dof 3"],
        Float[Tensor, "batch 9"],
        Float[Tensor, "batch"]
    ]:
        batch_idx = torch.randint(0, self.num_batches, (1,)).item()
        return self[batch_idx]

    def get_semi_random_batch(self) -> tuple[
        Float[Tensor, "batch dof 3"],
        Float[Tensor, "batch 9"],
        Float[Tensor, "batch"]
    ]:
        """
        Get a random batch within the current chunk.
        """
        batch_idx = self.current_chunk_idx * self.chunk_size // self.batch_size
        batch_idx += torch.randint(0, self.num_batches // self.num_chunks, (1,)).item()
        return self[batch_idx]


class TrainingSet(Dataset):
    @jaxtyped(typechecker=beartype)
    def __init__(self, batch_size: int, shuffle: bool):
        super().__init__(batch_size, shuffle, "train")

    @jaxtyped(typechecker=beartype)
    def _get_pose(self, cell_idx: Int64[Tensor, "batch 1"]) -> Float[Tensor, " batch 9"]:
        """
       Retrieve the pose from the batch chunk, which is a cell index in training.

       Args:
           cell_idx: Cell index.
       Returns:
           Pose
       """
        return se3.to_vector(se3.cell_noisy(cell_idx[:, 0]))


class ValidationSet(Dataset):
    @jaxtyped(typechecker=beartype)
    def __init__(self, batch_size: int, shuffle: bool, path: str = "val"):
        self.path = path
        super().__init__(batch_size, shuffle, path)

    @jaxtyped(typechecker=beartype)
    def _get_pose(self, pose: Float[Tensor, "batch 9"]) -> Float[Tensor, " batch 9"]:
        """
       Retrieve the pose from the batch chunk, which is already the pose in validation.

       Args:
           pose: Pose.
       Returns:
           Pose
       """
        return pose
