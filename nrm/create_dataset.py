import re
import argparse
from pathlib import Path

import torch
import zarr
import fasteners
from tqdm import tqdm

import nrm.dataset.se3 as se3
from nrm.dataset.morphology import sample_morph
from nrm.dataset.reachability_manifold import sample_reachability_manifold

CHUNK_SIZE = 100_000  # train: ~2.4MB, val:  ~4.4MB
SHARD_SIZE = CHUNK_SIZE * 1000  # train: ~2.4GB, val:  ~4.4GB

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="train", choices=["train", "val", "test"], help="For which set to sample")
parser.add_argument("--dof", type=int, default=6, choices=[1, 2, 3, 4, 5, 6, 7], help="number of degrees of freedom")
parser.add_argument("--num_robots", type=int, default=1000, help="number of robots to generate")
parser.add_argument("--num_samples", type=int, default=1_000_000, help="number of samples to generate per robot")
args = parser.parse_args()

assert args.num_samples * args.num_robots % CHUNK_SIZE == 0, f"Only full chunks are supported (chunk size {CHUNK_SIZE})"
assert SHARD_SIZE / args.num_samples == SHARD_SIZE // args.num_samples, f"One robot must belong to one shard (shard size {SHARD_SIZE})"

SAFE_FOLDER = Path(__file__).parent.parent / 'data' / args.set
lock = fasteners.InterProcessLock(SAFE_FOLDER.parent / f'{args.set}_lock.file')
compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

with lock:
    SAFE_FOLDER.mkdir(parents=True, exist_ok=True)
root = zarr.open(SAFE_FOLDER, mode="a")

with lock:
    file_indices = [
        int(match.group(1))
        for k in root.array_keys()
        if (match := re.search(r'^(\d+)_samples$', k))
    ]
    file_idx = (max(file_indices) + 1) if file_indices else 0
    morph_offset = sum([root[f"{idx}_morphologies"].shape[0] for idx in file_indices])

    print(f"Working in file {file_idx} with morph offset {morph_offset}")

    morph_filename = str(file_idx) + "_morphologies"
    sample_filename = str(file_idx) + "_samples"

    root.create_array(
        morph_filename,
        shape=(args.num_robots, 8, 3),
        dtype="float32",
        chunks=(args.num_robots, 8, 3),
        compressors=compressor,
        overwrite=False,
    )
    sample_type = "int64" if args.set == "train" else "float32"
    sample_dim = 3 if args.set == "train" else 11
    root.create_array(sample_filename,
                      shape=(args.num_robots * args.num_samples, sample_dim),
                      dtype=sample_type,
                      chunks=(CHUNK_SIZE, sample_dim),
                      shards=(SHARD_SIZE, sample_dim),
                      compressors=compressor)

morphs = sample_morph(args.num_robots, args.dof, False, torch.device("cuda"))
root[morph_filename][0:] = torch.nn.functional.pad(morphs, (0, 0, 0, 8 - morphs.shape[1])).cpu().numpy()

file = root[sample_filename]
file_offset = 0
buffer = torch.zeros(min(args.num_robots * args.num_samples, SHARD_SIZE), sample_dim,
                     dtype=torch.int64 if args.set == "train" else torch.float32)
buffer_offset = 0
for idx, morph in enumerate(tqdm(morphs, desc=f"Generating {args.dof} DOF robots")):
    if args.set == "train":
        cell_indices, labels = sample_reachability_manifold(morph, args.num_samples, seconds=30, use_ik=False)

        poses = cell_indices.unsqueeze(1)
        labels = labels.long().unsqueeze(1)
    else:
        poses, labels = sample_reachability_manifold(morph, args.num_samples, return_poses=True, use_ik=True)
        poses = se3.to_vector(poses)
        labels = labels.float().unsqueeze(1)
    morph_ids = torch.full_like(labels, idx + morph_offset)
    samples = torch.cat([morph_ids, poses, labels], dim=1)

    buffer[buffer_offset: buffer_offset + samples.shape[0]] = samples
    buffer_offset += samples.shape[0]

    if buffer_offset == buffer.shape[0] or idx == morphs.shape[0] - 1:
        active_data = buffer[:buffer_offset]

        active_data = active_data[torch.randperm(active_data.shape[0])]

        file[file_offset: file_offset + active_data.shape[0]] = active_data.cpu().numpy()
        file_offset += active_data.shape[0]
        buffer_offset = 0
