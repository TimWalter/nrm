import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped

from nrm.dataset.kinematics import transformation_matrix, forward_kinematics
from nrm.dataset.self_collision import EPS, LINK_RADIUS, get_capsules, collision_check
from nrm.dataset.manipulability import geometric_jacobian, yoshikawa_manipulability


# @jaxtyped(typechecker=beartype)
def _sample_link_type(batch_size: int, dof: int, device: torch.device) -> Int[Tensor, "batch_size {dof+1}"]:
    """
    Sample link types:
        0 <=> a≠0, d≠0 (General)
        1 <=> a=0, d≠0 (Only link length)
        2 <=> a≠0, d=0 (Only link offset)
        3 <=> a=0, d=0 (Spherical wrist component)
    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots
        device: device to operate on

    Returns:
        Link type sampled uniformly
    """
    link_type = torch.randint(0, 4 if dof > 1 else 3, size=(batch_size, dof + 1), device=device)

    return link_type


# @jaxtyped(typechecker=beartype)
def _reject_link_type(link_type: Int[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link type combinations that are not manufacturable.

    Args:
        link_type: Link types of the robots

    Returns:
        Mask of rejected link types
    """
    rejected = ((link_type[:, 1:] == 3) & (link_type[:, :-1] == 3)).any(dim=1)
    return rejected


# @jaxtyped(typechecker=beartype)
def _sample_link_twist(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1"]:
    """
    Sample link twist angles (α) uniformly from {-π/2, 0, π/2}

    Args:
        link_type: Link types of the robots

    Returns:
        Link twist angles
    """
    link_twist_options = torch.tensor([0, torch.pi / 2, -torch.pi / 2], device=link_type.device)
    link_twist_choice = torch.randint(0, 3, size=link_type.shape, device=link_type.device)
    link_twist = link_twist_options[link_twist_choice]
    return link_twist


# @jaxtyped(typechecker=beartype)
def _reject_link_twist(link_twist: Float[Tensor, "batch_size dofp1"], link_type: Int[Tensor, "batch_size dofp1"]) \
        -> Bool[Tensor, "batch_size"]:
    """
    Reject link twists that lead to degeneracy by having
    more than three consecutive parallel axes (i.e. α_i=0, α_{i+1}=0, α_{i+2}=0),
    collinear axes (i.e. a Type1 joint by another Type1 Joint with α_{i+1}=0),
    or that do describe unmanufacturable spherical wrist joints.

    Args:
        link_twist: Link twist angles (alpha) to check
        link_type: Link types of the robots

    Returns:
        Mask of rejected link twists
    """
    rejected = ((link_twist[:, :-2] == 0) & (link_twist[:, 1:-1] == 0) & (link_twist[:, 2:] == 0)).any(dim=1)
    rejected |= ((link_twist[:, :-2] == 0) & (link_type[:, 1:-1] == 3) & (link_twist[:, 2:] == 0)).any(dim=1)

    rejected |= ((link_type[:, :-1] == 1) & (link_type[:, 1:] == 1) & (link_twist[:, 1:] == 0)).any(dim=1)

    rejected |= ((link_type == 3) & (link_twist == 0)).any(dim=1)
    return rejected


# @jaxtyped(typechecker=beartype)
def _sample_analytically_solvable_link_types_and_twist(batch_size: int, dof: int, device: torch.device) \
        -> tuple[Int[Tensor, "batch_size dofp1"], Float[Tensor, "batch_size dofp1"]]:
    link_type = _sample_link_type(batch_size, dof, device)
    link_twist = _sample_link_twist(link_type)
    if dof == 5:
        types = torch.randint(0, 3, size=(batch_size,))
        r"""
        5 DOF Analytically solvable robot types:
            0 <=> (a_1=0) | (a_4=0)                 - The last or first two axes intersect
            1 <=> (a_2=0 & α_4=0) | (a_3=0 & α_3=0) - One pair of consecutive, intermediate axes intersects while the other is parallel
            2 <=> α_i=0 & α_{i+1}=0, i \in [1, 3]   - Any three consecutive axes are parallel
        """
        link_type[types == 0, torch.randint(0, 2, ((types == 0).sum().item(),), device=link_type.device) * 3 + 1] = 1

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),), device=link_type.device)
        link_type[types == 1, axes_choice * 2 + 2] = 1
        link_twist[types == 1, (~axes_choice.bool()) * 2 + 2] = 0

        axes_choice = torch.randint(1, 4, ((types == 2).sum().item(),), device=link_type.device)
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof == 6:
        types = torch.randint(0, 3, size=(batch_size,), device=link_type.device)
        """
        6 DOF Analytically solvable robot types:
            0 <=> (a_4=0 & a_5=0 & d_4=0) | (a_1=0 & a_2=0 & d_1=0) - Spherical wrist (3 intersecting axes) at the beginning or end
            1 <=> (α_1=0 & α_2=0 & a_5=0) | (α_4=0 & α_5=0 & a_1=0)                 - 3 Parallel & 2 intersecting axes on opposing ends
            2 <=> (α_2=0 & α_3=0) | (α_3=0 & α_4=0)                                 - 3 Parallel inner axes
        """
        axes_choice = torch.randint(0, 2, ((types == 0).sum().item(),), device=link_type.device)
        link_type[types == 0, 1 + 3 * axes_choice] = 3
        link_type[types == 0, 2 + 3 * axes_choice] = 1

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),), device=link_type.device)
        link_type[types == 1, 1 + 4 * axes_choice] = 1
        link_twist[types == 1, 1 + 3 * (~axes_choice.bool())] = 0
        link_twist[types == 1, 2 + 3 * (~axes_choice.bool())] = 0

        axes_choice = torch.randint(2, 4, ((types == 2).sum().item(),), device=link_type.device)
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof > 6:
        raise NotImplementedError("Analytically solvable sampling not implemented for DOF > 6")
    return link_type, link_twist


# @jaxtyped(typechecker=beartype)
def _sample_link_length(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1 2"]:
    """
    Sample link lengths uniformly on the simplex by normalising exponential samples and splitting the lengths
    according to the link types and a random angle for the links with both a and d non-zero.

    Args:
        link_type: Link types of the robots

    Returns:
        Link lengths (a, d)
    """
    dist = torch.distributions.exponential.Exponential(torch.tensor([1.0], device=link_type.device))
    link_lengths = dist.sample(torch.Size((*link_type.shape, 1)))[..., 0].repeat(1, 1, 2)
    link_lengths /= (link_lengths * (link_type.unsqueeze(-1) != 3)).sum(dim=1, keepdim=True)
    link_lengths *= torch.sign(torch.rand_like(link_lengths) - 0.5)

    gamma = torch.rand((link_type == 0).sum(), device=link_type.device) * 2 * torch.pi
    link_lengths[..., 0][link_type == 0] *= torch.sin(gamma).abs()
    link_lengths[..., 1][link_type == 0] *= torch.cos(gamma).abs()

    link_lengths[..., 0][(link_type == 1) | (link_type == 3)] = 0
    link_lengths[..., 1][(link_type == 2) | (link_type == 3)] = 0
    return link_lengths


# @jaxtyped(typechecker=beartype)
def _reject_link_length(link_length: Float[Tensor, "batch_size dofp1 2"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link lengths that are smaller than twice the link radius (0.025) but not zero.

    Args:
        link_length: Link lengths to check

    Returns:
        Mask of rejected link lengths
    """
    rejected = ((link_length.abs() < 2 * LINK_RADIUS) & (link_length != 0)).any(dim=(1, 2))
    if link_length.shape[1] > 2:
        rejected |= torch.sqrt((link_length[:, 0, :] ** 2).sum(dim=-1)) > 1 / link_length.shape[1]
        rejected |= torch.sqrt((link_length[:, -1, :] ** 2).sum(dim=-1)) > 1 / link_length.shape[1]
    return rejected


# @jaxtyped(typechecker=beartype)
def _sample_morph(batch_size: int, dof: int, analytically_solvable: bool, device=torch.device) \
        -> Float[Tensor, "batch_size {dof+1} 3"]:
    """
    Sample morphologies, encoded as MDH parameters (alpha, a, d), given their respective rejection criteria.

    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots
        analytically_solvable: whether to sample only analytically solvable robots
        device: Torch device.

    Returns:
        MDH parameters (alpha, a, d) describing the robot morphology
    """
    oversampling = 2 * batch_size
    if analytically_solvable:
        link_types, link_twists = _sample_analytically_solvable_link_types_and_twist(oversampling, dof, device)
        while (mask := _reject_link_type(link_types) | _reject_link_twist(link_twists, link_types)).any():
            link_types[mask], link_twists[mask] = _sample_analytically_solvable_link_types_and_twist(mask.sum().item(),
                                                                                                     dof, device)
    else:
        link_types = _sample_link_type(oversampling, dof, device)
        while (mask := _reject_link_type(link_types)).any():
            link_types[mask] = _sample_link_type(mask.sum().item(), dof, device)

        link_twists = _sample_link_twist(link_types)
        while (mask := _reject_link_twist(link_twists, link_types)).any():
            link_twists[mask] = _sample_link_twist(link_types[mask])

    link_lengths = _sample_link_length(link_types)
    while oversampling - (mask := _reject_link_length(link_lengths)).sum() < batch_size:
        link_lengths[mask] = _sample_link_length(link_types[mask])
    link_lengths = link_lengths[~mask][:batch_size]
    link_twists = link_twists[~mask][:batch_size]

    morph = torch.cat([link_twists.unsqueeze(-1), link_lengths], dim=2)

    return morph


@jaxtyped(typechecker=beartype)
def get_joint_limits(morph: Float[Tensor, "*batch dof 3"]) -> Float[Tensor, "*batch dof 2"]:
    """
    Compute joint limits based on the morphology to avoid self-collisions.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Joint limits
    """
    joint_limits = torch.zeros(*morph.shape[:-1], 2, device=morph.device)

    extended_morph = torch.cat([torch.zeros_like(morph[..., :1, :]), morph], dim=-2)
    alpha0, a0, d0 = extended_morph[..., :-2, :].split(1, dim=-1)
    alpha1, a1, d1 = extended_morph[..., 1:-1, :].split(1, dim=-1)

    coordinate_fix = torch.eye(4, device=morph.device, dtype=morph.dtype).repeat(*morph.shape[:-2], morph.shape[-2] - 1,
                                                                                 1, 1)
    wrist = (a1[..., 0] == 0) & (d1[..., 0] == 0)
    coordinate_fix[wrist] = transformation_matrix(alpha0, a0, d0, torch.zeros_like(d0))[wrist]

    plane_normal = torch.stack([
        torch.zeros_like(alpha1),
        -torch.sin(alpha1),
        torch.cos(alpha1),
        torch.zeros_like(alpha1)], dim=-1)
    plane_anchor = torch.stack([
        a1,
        -d1 * torch.sin(alpha1),
        d1 * torch.cos(alpha1),
        torch.ones_like(alpha1)], dim=-1)

    plane_normal = torch.sum(coordinate_fix * plane_normal, dim=-1)[..., :3]
    plane_anchor = torch.sum(coordinate_fix * plane_anchor, dim=-1)[..., :3]

    stacked_morph = torch.stack([extended_morph[..., :-2, :], extended_morph[..., 1:-1, :], extended_morph[..., 2:, :]],
                                dim=-2)
    stacked_morph[~wrist, 0, :] = 0.0
    stacked_poses = forward_kinematics(stacked_morph, torch.zeros(*stacked_morph.shape[:-1], 1, device=morph.device))
    start, end = get_capsules(stacked_morph, stacked_poses)
    capsules = end - start

    # Get closest non-zero capsule before joint
    pre_capsule = capsules[..., 3, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 2, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 1, :]

    # Get closest non-zero capsule after joint
    post_capsule = capsules[..., -2, :]
    post_capsule[mask] = capsules[mask := post_capsule.norm(dim=-1) < 1e-6, -1, :]

    in_plane = ((pre_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6
    in_plane &= ((post_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6

    limited = (pre_capsule.norm(dim=-1) > EPS) & (post_capsule.norm(dim=-1) > EPS) & in_plane

    mask = post_capsule.norm(dim=-1) > pre_capsule.norm(dim=-1)
    arc = torch.arcsin(2 * LINK_RADIUS / post_capsule.norm(dim=-1))
    arc[mask] = torch.arcsin(2 * LINK_RADIUS / pre_capsule.norm(dim=-1))[mask]

    joint_limits[..., :-1, 0] = torch.where(limited, 2 * torch.pi - 2 * arc, 2 * torch.pi)  # Range
    angle = torch.atan2(torch.sum(torch.cross(pre_capsule, post_capsule, dim=-1) * plane_normal, dim=-1),
                        torch.sum(pre_capsule * post_capsule, dim=-1))
    # if their angle becomes pi, they collide and are antiparallel
    angle = torch.atan2(torch.sin(torch.pi - angle), torch.cos(torch.pi - angle))
    joint_limits[..., :-1, 1] = torch.where(limited, angle + arc, -torch.pi)  # Offset

    return joint_limits


# @jaxtyped(typechecker=beartype)
def _reject_morph(morph: Float[Tensor, "batch_size dofp1 3"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject morphologies that have adjacent links that seem constantly in collision or that seem
    degenerate (Jacobian is rank-deficient for all configurations).

    Args:
        morph: Morphologies to check

    Returns:
        Mask of rejected morphologies
    """
    joint_limits = get_joint_limits(morph)

    joint_limits = joint_limits.unsqueeze(1).expand(-1, 1000, -1, -1)
    morph = morph.unsqueeze(1).expand(-1, 1000, -1, -1)
    joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
        ..., 1:2]

    poses = forward_kinematics(morph, joints)
    rejected = collision_check(morph, poses).all(dim=1)
    jacobian = geometric_jacobian(poses)
    rejected |= (yoshikawa_manipulability(jacobian, True) < 1e-4).all(dim=1)

    return rejected


# @jaxtyped(typechecker=beartype)
def sample_morph(num_robots: int, dof: int, analytically_solvable: bool, device: torch.device= torch.device("cpu")) \
        -> Float[Tensor, "num_robots {dof+1} 3"]:
    """
   Sample valid morphologies, encoded in modified Denavit-Hartenberg parameters (alpha, a, d).

   Args:
       num_robots: number of robots to sample
       dof: degrees of freedom of the robots
       analytically_solvable: whether to sample only analytically solvable robots
       device: Torch device to operate on.

   Returns:
       MDH parameters (alpha, a, d) describing the robot morphology
   """
    num_samples = int(num_robots * 1.5)
    morph = _sample_morph(num_samples, dof, analytically_solvable, device)
    while num_samples - (mask := _reject_morph(morph)).sum() < num_robots:
        morph[mask] = _sample_morph(mask.sum().item(), dof, analytically_solvable, device)

    return morph[~mask][:num_robots]


if __name__ == "__main__":
    sample_morph(num_robots=1, dof=6, analytically_solvable=False, device=torch.device("cuda"))
