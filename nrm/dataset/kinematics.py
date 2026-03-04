import torch

from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int, Float64, Bool
from eaik.IK_Homogeneous import HomogeneousRobot

import nrm.dataset.se3 as se3
from nrm.dataset import r3, so3
from nrm.dataset.self_collision import EPS, collision_check
from nrm.dataset.manipulability import geometric_jacobian, yoshikawa_manipulability


# @jaxtyped(typechecker=beartype)
def transformation_matrix(alpha: Float[Tensor, "*batch 1"], a: Float[Tensor, "*batch 1"], d: Float[Tensor, "*batch 1"],
                          theta: Float[Tensor, "*batch 1"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Computes the modified Denavit-Hartenberg transformation matrix.

    Args:
        alpha: Twist angle
        a: Link length
        d: Link offset
        theta: Joint angle

    Returns:
        Transformation matrix.
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)
    return torch.stack([torch.cat([ct, -st, zero, a], dim=-1),
                        torch.cat([st * ca, ct * ca, -sa, -d * sa], dim=-1),
                        torch.cat([st * sa, ct * sa, ca, d * ca], dim=-1),
                        torch.cat([zero, zero, zero, one], dim=-1)], dim=-2)


# @jaxtyped(typechecker=beartype)
def forward_kinematics(mdh: Float[Tensor, "*batch dofp1 3"],
                       theta: Float[Tensor, "*batch dofp1 1"]) -> Float[Tensor, "*batch dofp1 4 4"]:
    """
    Computes forward kinematics for a robot defined by modified Denavit-Hartenberg parameters.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        theta: The joint angle (theta_i) for each joint.

    Returns:
        The transformation matrices from the base frame to each joint frame.
    """
    transforms = transformation_matrix(mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta)

    poses = []
    pose = torch.eye(4, device=mdh.device, dtype=mdh.dtype).expand(*mdh.shape[:-2], 1, 4, 4)
    for i in range(mdh.shape[-2]):
        poses.append(pose := pose @ transforms[..., i:i + 1, :, :])

    return torch.cat(poses, dim=-3)


# @jaxtyped(typechecker=beartype)
def unique_with_index(x: Tensor, dim: int = 0) -> tuple[Tensor, Tensor]:
    """
    Compute the unique version of a tensor and return both this version and the indices of the first appearance of each
    unique element.

    Args:
        x: Tensor to make unique
        dim: Axes of the operation

    Returns:
        Unique tensor and respective indices.
    """

    unique, inverse, counts = torch.unique(x, dim=dim,
                                           sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, index


# @jaxtyped(typechecker=beartype)
def unique_indices(indices: Int[Tensor, "batch"],
                   manipulability: Float[Tensor, "batch"],
                   other: list[Float[Tensor, "batch *rest"]]) \
        -> tuple[
            Int[Tensor, "n_unique"],
            Float[Tensor, "n_unique"],
            list[Float[Tensor, "n_unique *rest"]]
        ]:
    """
    Select unique indices, keeping the ones with the highest manipulability.

    Args:
        indices: Indices corresponding to determine uniqueness
        manipulability: Manipulability values corresponding to indices
        other: Other tensors to be filtered accordingly

    Returns:
        Unique indices, their manipulability, and other tensors filtered accordingly.
    """
    manipulability, sort_indices = torch.sort(manipulability, descending=True)
    indices = indices[sort_indices]
    other = [tensor[sort_indices] for tensor in other]

    indices, unique_indices = unique_with_index(indices)

    manipulability = manipulability[unique_indices]
    other = [tensor[unique_indices] for tensor in other]

    return indices, manipulability, other

def is_analytically_solvable(morph: Float[Tensor, "batch_size dofp1 3"]) -> Bool[Tensor, "batch_size"]:
    """
    Determine whether a robot is solvable by EAIK.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Mask indicating whether the robot is solvable.
    """
    dof = morph.shape[1] - 1
    if dof < 5:
        mask = torch.ones(morph.shape[0], dtype=torch.bool, device=morph.device)
    else:
        mask = torch.zeros(morph.shape[0], dtype=torch.bool, device=morph.device)
        if dof == 5:
            # The last or first two axes intersect
            mask |= (morph[:, 1, 1] == 0) | (morph[:, 4, 1] == 0)
            # One pair of consecutive, intermediate axes intersects while the other is parallel
            mask |= ((morph[:, 2, 1] == 0) & (morph[:, 4, 0] == 0)) | ((morph[:, 4, 1] == 0) & (morph[:, 2, 0] == 0))
            # Any three consecutive axes are parallel
            mask |= ((morph[:, 1:, 0] == 0) & (morph[:, :-1, 0] == 0)).any(dim=1)
        elif dof == 6:
            # Spherical wrist at the beginning or end
            mask |= (morph[:, 1, 1] == 0) & (morph[:, 1, 2] == 0) & (morph[:, 2, 1] == 0)
            mask |= (morph[:, 4, 1] == 0) & (morph[:, 4, 2] == 0) & (morph[:, 5, 1] == 0)
            # 3 Parallel & 2 intersecting axes on opposing ends
            mask |= (morph[:, 1, 0] == 0) & (morph[:, 2, 0] == 0) & (morph[:, 5, 1] == 0)
            mask |= (morph[:, 4, 0] == 0) & (morph[:, 5, 0] == 0) & (morph[:, 1, 1] == 0)
            # 3 Parallel inner axes
            mask |= (morph[:, 3, 0] == 0) & ((morph[:, 2, 0] == 0) | (morph[:, 4, 0] == 0))

    return mask

# @jaxtyped(typechecker=beartype)
def inverse_kinematics(mdh: Float[Tensor, "dofp1 3"],
                       poses: Float[Tensor, "batch 4 4"]) -> tuple[
    Float[Tensor, "batch dofp1 1"],
    Float[Tensor, "batch"]
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint angles (theta_i) for each joint that achieve the desired poses with the highest manipulability and
        the manipulability values.
    """
    if is_analytically_solvable(mdh.unsqueeze(0)):
        try:
            joints, manipulability = analytical_inverse_kinematics(mdh.cpu(), poses.cpu())
        except RuntimeError:
            joints, manipulability = numerical_inverse_kinematics(mdh, poses)
    else:
        joints, manipulability = numerical_inverse_kinematics(mdh, poses)

    return joints, manipulability


# @jaxtyped(typechecker=beartype)
def morph_to_eaik(mdh: Float[Tensor, "dofp1 3"]) -> HomogeneousRobot:
    """
    Transform our description of a morphology (mdh parameters) into one the EAIK tool can understand for
    analytical IK.

    Args:
        mdh: Modified DH parameters [alpha, a, d, theta]

    Returns:
        EAIK class with IK_batched method
    """
    local_coord = transformation_matrix(mdh[:, 0:1], mdh[:, 1:2], mdh[:, 2:3], torch.zeros_like(mdh[:, 2:3]))
    global_coords = torch.empty_like(local_coord)
    global_coords[0] = local_coord[0]
    for i in range(1, len(local_coord)):
        global_coords[i] = global_coords[i - 1] @ local_coord[i]
    return HomogeneousRobot(global_coords.cpu().numpy())


# @jaxtyped(typechecker=beartype)
def pure_analytical_inverse_kinematics(mdh: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> list[
    Float[Tensor, "n_solutions dofp1 1"],
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters analytically via EAIK and
    returns all solutions without checking for self-collisions or whether we actually end up in the correct position.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint solutions
    """
    eaik_bot = morph_to_eaik(mdh)
    if not eaik_bot.hasKnownDecomposition():
        raise RuntimeError(f"Robot is not analytically solvable. {mdh}")
    solutions = eaik_bot.IK_batched(poses.cpu().numpy())
    if torch.tensor([sol.num_solutions() == 0 for sol in solutions]).all():
        raise RuntimeError(f"EAIK bug.")
    joints = [torch.cat([torch.from_numpy(sol.Q.copy()).unsqueeze(-1),
                         torch.zeros(sol.num_solutions(), 1, 1)
                         ],dim=1) if sol.num_solutions() != 0
              else torch.empty(0, mdh.shape[0], 1, dtype=torch.double)
              for sol in solutions]

    return joints


# @jaxtyped(typechecker=beartype)
def analytical_inverse_kinematics(mdh: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> tuple[
    Float64[Tensor, "batch dofp1 1"],
    Float64[Tensor, "batch"]
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters analytically via EAIK.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint angles (theta_i) for each joint that achieve the desired poses with the highest manipulability and
        the manipulability values.
    """
    mdh = mdh.double().cpu()
    poses = poses.double().cpu()

    joints = pure_analytical_inverse_kinematics(mdh, poses)
    pose_indices = torch.cat(
        [torch.full((joint.shape[0], 1), i, dtype=torch.int64) for i, joint in enumerate(joints)], dim=0)
    joints = torch.cat(joints, dim=0)
    if joints.shape[0] != 0:
        bmorph = mdh.unsqueeze(0).expand(joints.shape[0], -1, -1)
        full_poses = forward_kinematics(bmorph, joints)
        self_collision = collision_check(bmorph, full_poses)

        pose_error = se3.distance(full_poses[:, -1, :, :], poses[pose_indices[:, 0]]).squeeze(-1)

        mask = ~self_collision & (pose_error < EPS)
        full_poses = full_poses[mask]
        joints = joints[mask]
        pose_indices = pose_indices[mask]

        jacobian = geometric_jacobian(full_poses)
        manipulability = yoshikawa_manipulability(jacobian)

        pose_indices, manipulability, [joints] = unique_indices(pose_indices[:, 0], manipulability, [joints])
    else:
        pose_indices = torch.zeros((*poses.shape[:-2],), dtype=torch.bool)
        manipulability = torch.empty(0, dtype=torch.double)

    full_joints = torch.zeros((*poses.shape[:-2], mdh.shape[0], 1), dtype=torch.double)
    full_joints[pose_indices] = joints

    full_manipulability = -torch.ones((*poses.shape[:-2],), dtype=torch.double)
    full_manipulability[pose_indices] = manipulability

    return full_joints, full_manipulability


# @jaxtyped(typechecker=beartype)
def numerical_inverse_kinematics(inp_mdh: Float[Tensor, "dofp1 3"], inp_poses: Float[Tensor, "batch 4 4"]) \
        -> tuple[Float[Tensor, "batch dofp1 1"], Float[Tensor, "batch"]]:
    """
    Fast Levenberg-Marquardt IK solver using Cholesky decomposition.
    """
    j = []
    m = []

    for batch_idx in range(0, inp_poses.shape[0], 100000):
        poses = inp_poses[batch_idx:batch_idx + 100000]

        max_iter = 100
        damping = 1e-3
        num_seeds = 10

        device = inp_mdh.device
        dtype = inp_mdh.dtype
        batch_size = poses.shape[0]
        dof = inp_mdh.shape[0] - 1

        total_batch = batch_size * num_seeds

        mdh = inp_mdh.unsqueeze(0).expand(total_batch, -1, -1)
        poses = poses.unsqueeze(1).expand(-1, num_seeds, -1, -1).reshape(total_batch, 4, 4)

        # Initialize joints (randomly)
        joints = (torch.rand(total_batch, dof, 1, device=device, dtype=dtype) * 2 * torch.pi) - torch.pi

        # Pre-allocate damping matrix (Tikhonov regularization)
        damping_vals = torch.full((total_batch, 1, 1), damping, device=device, dtype=dtype)
        prev_error_norm = torch.full((total_batch,), float('inf'), device=device, dtype=dtype)

        for _ in range(max_iter):
            # 1. Forward Kinematics
            full_joints = torch.cat([joints, torch.zeros(total_batch, 1, 1, device=device, dtype=dtype)], dim=1)
            reached_pose = forward_kinematics(mdh, full_joints)

            # 2. Compute Error (6D)
            error = torch.cat([r3.log(reached_pose[..., -1, :3, 3], poses[..., :3, 3]),
                               torch.einsum('bij,bj->bi', reached_pose[..., -1, :3, :3],
                                            so3.log(reached_pose[..., -1, :3, :3], poses[..., :3, :3]))], dim=-1).unsqueeze(
                -1)
            active = error[..., 0].norm(dim=-1) > EPS

            # 3. Jacobian
            jacobian = geometric_jacobian(reached_pose)  # [B, 6, dof]

            # Adaptive Damping
            got_worse = error[..., 0].norm(dim=-1) > prev_error_norm
            damping_vals[got_worse] *= 2.0
            damping_vals[~got_worse & active] *= 0.7
            damping_vals.clamp_(1e-6, 1e3)
            prev_error_norm = error[..., 0].norm(dim=-1).clone()
            diag_damp = torch.eye(dof, device=device, dtype=dtype).unsqueeze(0) * damping_vals

            # 4. Solve Normal Equations: (J^T J + λI) Δθ = J^T e
            lhs = jacobian.transpose(-2, -1) @ jacobian + diag_damp
            rhs = jacobian.transpose(-2, -1) @ error

            # Cholesky solve is numerically stable for Positive Definite matrices (which damped J^T J is)
            # We assume damping is sufficient to avoid singularity.
            delta_theta = torch.linalg.solve(lhs, rhs)

            # 5. Update
            joints[active] = joints[active] + delta_theta[active]

        # Finalize and Wrap
        joints = torch.atan2(torch.sin(joints), torch.cos(joints))
        full_joints = torch.cat([joints, torch.zeros(total_batch, 1, 1, device=device, dtype=dtype)], dim=1)

        # Metrics
        reached_pose = forward_kinematics(mdh, full_joints)
        jacobian = geometric_jacobian(reached_pose)
        manipulability = yoshikawa_manipulability(jacobian)

        self_collision = collision_check(mdh, reached_pose)
        error = se3.distance(reached_pose[:, -1, :, :], poses).squeeze(-1)

        # Gather best solutions
        full_joints = full_joints.reshape(batch_size, num_seeds, dof + 1, 1)
        manipulability = manipulability.reshape(batch_size, num_seeds)
        self_collision = self_collision.reshape(batch_size, num_seeds)
        error = error.reshape(batch_size, num_seeds)

        error_masked = error.clone()
        error_masked[self_collision] = torch.inf
        best_seed_idx = error_masked.argmin(dim=1)  # [batch]

        batch_indices = torch.arange(batch_size, device=device)
        best_joints = full_joints[batch_indices, best_seed_idx]  # [batch, dof+1, 1]
        best_manipulability = manipulability[batch_indices, best_seed_idx]  # [batch]
        best_error = error[batch_indices, best_seed_idx]  # [batch]
        best_collision = self_collision[batch_indices, best_seed_idx]  # [batch]

        # Strict success check
        mask = (best_error < EPS) & ~best_collision
        best_manipulability[~mask] = -1.0


        j.append(best_joints)
        m.append(best_manipulability)

    best_joints = torch.cat(j, dim=0)
    best_manipulability = torch.cat(m, dim=0)

    return best_joints, best_manipulability
