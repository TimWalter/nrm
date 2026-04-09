import warnings

warnings.filterwarnings("ignore", message=".*Dynamo detected a call to a `functools.lru_cache`.*")

from datetime import datetime, timedelta

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Bool, Int64
from tabulate import tabulate

import nrm.dataset.r3 as r3
import nrm.dataset.se3 as se3
from nrm.autotune_batch_size import get_batch_size
from nrm.dataset.kinematics import transformation_matrix, forward_kinematics, \
    inverse_kinematics
from nrm.dataset.self_collision import collision_check
from nrm.dataset.morphology import get_joint_limits

torch.set_float32_matmul_precision("high")


# @jaxtyped(typechecker=beartype)
def sample_reachable_poses(morph: Float[Tensor, "dof 3"], joint_limits: Float[Tensor, "batch_size dof 2"]) -> tuple[
    Float[Tensor, "n_valid 4 4"],
    Int64[Tensor, "n_valid"],
]:
    """
    Sample end effector poses for the robot and compute their discretised cell index.

    Args:
        morph: MDH parameters encoding the robot geometry.
        joint_limits: Joint limits that prevent most self-collisions.

    Returns:
        Reachable end effector poses and their respective cell indices.
    """
    joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
        ..., 1:2]
    poses = forward_kinematics(morph, joints)
    self_collision = collision_check(morph, poses)
    poses = poses[:, -1, :, :][~self_collision]
    cell_indices = se3.index(poses)
    return poses, cell_indices


compiled_sample_reachable_poses = torch.compile(sample_reachable_poses)


# @jaxtyped(typechecker=beartype)
def estimate_reachability_manifold(morph: Float[Tensor, "dofp1 3"], debug: bool = False, seconds: int = 60,
                                   batch_size: int = None) -> \
        Int64[Tensor, "num_samples"] | tuple[Int64[Tensor, "num_samples"], tuple[int, int, float, float, float], int]:
    """
    Estimat the reachability manifold using only forward kinematics, a discretisation of SE(3) and the closed world assumption.
    Fill up the discretised cells using FK until convergence. All unfilled cells are assumed to be unreachable.

    Args:
        morph: MDH parameters encoding the robot geometry.
        debug: Whether to return benchmark parameters.
        seconds: Number of seconds to sample FK.
        batch_size: Batch size if None is determined automatically.

    Returns:
        Cell indices of reachable cells and optionally benchmark parameters and batch size.
    """
    joint_limits = get_joint_limits(morph)
    probe_size = 2048
    if batch_size is None:
        args = [morph.unsqueeze(0).expand(probe_size, -1, -1),
                joint_limits.unsqueeze(0).expand(probe_size, -1, -1)]
        # To exclude any one-time costs from the batch size calculation
        sample_reachable_poses(*args)
        batch_size = get_batch_size(morph.device, sample_reachable_poses, probe_size, args, safety=0.5)
    morph = morph.unsqueeze(0).expand(batch_size, -1, -1)
    joint_limits = joint_limits.unsqueeze(0).expand(batch_size, -1, -1)
    if debug:  # Warm-Up for benchmarking
        _ = compiled_sample_reachable_poses(morph, joint_limits)

    indices = []
    cuda_indices = []
    cuda_memory = 0
    n_batches = 0
    collision_free_samples = 0
    start = datetime.now()
    while datetime.now() - start < timedelta(seconds=seconds):
        _, new_indices = compiled_sample_reachable_poses(morph, joint_limits)
        cuda_indices += [new_indices]
        n_batches += 1
        collision_free_samples += new_indices.shape[0]
        cuda_memory += new_indices.numel() * new_indices.element_size()
        if cuda_memory > 2 * 1024 ** 3:  # Flush every 2 GB
            transfer = torch.cat(cuda_indices).unique()
            pinned = torch.empty(transfer.shape, dtype=transfer.dtype, pin_memory=True)
            pinned.copy_(transfer, non_blocking=True)
            indices += [pinned]
            cuda_indices = []
            cuda_memory = 0
            if len(indices) >= 5:
                indices = [torch.cat(indices).unique()]

    if len(cuda_indices) > 0:
        indices += [torch.cat(cuda_indices).cpu()]

    indices = torch.cat(indices).unique()

    if debug:
        filled_cells = indices.shape[0]
        total_samples = n_batches * batch_size
        total_efficiency = filled_cells / total_samples * 100
        unique_efficiency = filled_cells / collision_free_samples * 100
        collision_efficiency = collision_free_samples / total_samples * 100
        return indices, (filled_cells, total_samples, total_efficiency, unique_efficiency, collision_efficiency), batch_size
    return indices


# @jaxtyped(typechecker=beartype)
def estimate_reachable_ball(morph: Float[Tensor, "*batch dof 3"]) -> tuple[
    Float[Tensor, "*batch 3"],
    Float[Tensor, "*batch"]
]:
    """
    Estimate the reachable ball of a robot.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Center and radius of the reachable ball.
    """
    centre = transformation_matrix(morph[..., 0, 0:1],
                                   morph[..., 0, 1:2],
                                   morph[..., 0, 2:3],
                                   torch.zeros_like(morph[..., 0, 2:3]))[..., :3, 3]
    radius = torch.sqrt(morph[..., 1:, 1] ** 2 + morph[..., 1:, 2] ** 2).sum(dim=-1)

    return centre, radius


def sample_poses_in_reach(num_samples: int, morph: Float[Tensor, "dof 3"]) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample poses that could be reached by the robot, i.e. do not sample outside the reachable ball.

    Args:
        num_samples: Number of samples to generate.
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Poses that could be reached by the robot.
    """
    centre, radius = estimate_reachable_ball(morph[:-1])  # Ignore the EEF
    radius = max(0.0, radius - r3.DISTANCE_BETWEEN_CELLS)  # Robust within discretisation
    last_joint = se3.random_ball(num_samples, centre, radius).to(morph.device)

    eef_transformation = transformation_matrix(morph[-1, 0:1], morph[-1, 1:2], morph[-1, 2:3],
                                               torch.zeros_like(morph[-1, 0:1])).to(morph.device)
    return last_joint @ eef_transformation


# @jaxtyped(typechecker=beartype)
def sample_reachability_manifold(morph: Float[Tensor, "dofp1 3"],
                                 num_samples: int,
                                 return_poses: bool = False,
                                 use_ik: bool = False,
                                 seconds: int = 60) -> \
        tuple[Int64[Tensor, "num_samples"], Bool[Tensor, "num_samples"]] | \
        tuple[Float[Tensor, "num_samples 4 4"], Bool[Tensor, "num_samples"]]:
    """
    Estimate the workspace of a robot solely from forward kinematics.

    Args:
        morph: MDH parameters encoding the robot geometry.
        return_poses: Whether to return poses or only the cell index.
        use_ik: Whether to use inverse kinematics or only forward kinematics.
        num_samples: Number of samples to generate.
        seconds: Number of seconds to sample FK.

    Returns:
        Labels and indices encoding the discretised reachability manifold
    """
    poses = sample_poses_in_reach(num_samples, morph)
    cell_indices = se3.index(poses)

    if not use_ik:
        r_indices = estimate_reachability_manifold(morph, seconds=seconds)
        labels = torch.isin(cell_indices, r_indices)
    else:
        joints, manipulability = inverse_kinematics(morph, poses)
        labels = manipulability.cpu() != -1

    return cell_indices if not return_poses else poses, labels


if __name__ == "__main__":
    from nrm.dataset.morphology import sample_morph

    torch.manual_seed(1)
    morphs = sample_morph(10, 6, True, torch.device("cuda"))
    benchmarks = []
    for morph in morphs:
        morph = morph.to("cuda")
        _, benchmark = estimate_reachability_manifold(morph, True)
        benchmarks += [torch.tensor(benchmark)]

    mean_benchmark = torch.stack(benchmarks).mean(dim=0, keepdim=True).tolist()
    mean_benchmark[0][0] = int(mean_benchmark[0][0])
    mean_benchmark[0][1] = int(mean_benchmark[0][1])
    print(tabulate(mean_benchmark,
                   headers=["Filled Cells", "Total Samples<br>(Speed)", "Efficiency<br>(Total)",
                            "Efficiency<br>(Unique)",
                            "Efficiency<br>(Collision)"], floatfmt=".4f", intfmt=",", tablefmt="github"))
