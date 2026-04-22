import torch

from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool

LINK_RADIUS = 0.025
EPS = 1e-4

# @jaxtyped(typechecker=beartype)
def get_capsules(mdh: Float[Tensor, "*batch dofp1 3"],
                 poses: Float[Tensor, "*batch dofp1 4 4"]) -> tuple[
    Float[Tensor, "*batch 2*dofp1 3"],
    Float[Tensor, "*batch 2*dofp1 3"],
]:
    """
    Given a robot morphology compute the link-enclosing capsule start and endpoints.

    Args:
        mdh: Modified DH parameters [alpha, a, d, theta]
        poses: Pose of the robot

    Returns:
        Startpoints and endpoints
    """
    *batch_shape, dof, _ = mdh.shape
    device = mdh.device
    dtype = mdh.dtype

    # Prepend the Identity matrix (Base Frame 0) to poses (poses currently contains [T1, T2, ..., TN]. We need [T0, T1, ..., TN].)
    identity = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 1, 4, 4)
    poses = torch.cat([identity, poses], dim=-3)

    # Starting point of the first capsule (a) is the pose from base
    s_a = poses[..., :-1, :3, 3]
    # End point of the second capsule (d) is the pose after
    e_d = poses[..., 1:, :3, 3]
    # The middle point is deducted by reversing the translation d along the z-axis
    z_axis = poses[..., 1:, :3, 2]
    d = mdh[..., 2].unsqueeze(-1)
    e_a = s_d = e_d - d * z_axis

    # Assemble the chain (stack+flatten essentially zips such that s_a_1, s_d_1, s_a_2, s_d_2, ..)
    s_all = torch.stack([s_a, s_d], dim=-2).flatten(-3, -2)
    e_all = torch.stack([e_a, e_d], dim=-2).flatten(-3, -2)
    return s_all, e_all

# @jaxtyped(typechecker=beartype)
def signed_distance_capsule_capsule(s1: Float[Tensor, "*batch 3"], e1: Float[Tensor, "*batch 3"], r1: float,
                                    s2: Float[Tensor, "*batch 3"], e2: Float[Tensor, "*batch 3"], r2: float) \
        -> Float[Tensor, "*batch"]:
    """
    Compute the signed squared distance between two capsules.

    Args:
        s1: Startpoint of the first capsule.
        e1: Endpoint of the first capsule.
        r1: Radius of the first capsule.
        s2: Startpoint of the second capsule.
        e2: Endpoint of the second capsule.
        r2: Radius of the second capsule.

    Returns:
        Signed squared distance
    """
    l1 = e1 - s1
    l2 = e2 - s2
    ds = s1 - s2

    alpha = (l1 * l1).sum(dim=-1, keepdim=True)
    beta = (l2 * l2).sum(dim=-1, keepdim=True)
    gamma = (l1 * l2).sum(dim=-1, keepdim=True)
    delta = (l1 * ds).sum(dim=-1, keepdim=True)
    epsilon = (l2 * ds).sum(dim=-1, keepdim=True)

    det = alpha * beta - gamma ** 2

    t1 = torch.clamp((gamma * epsilon - beta * delta) / (det + 1e-10), 0.0, 1.0)
    t2 = torch.clamp((gamma * t1 + epsilon) / (beta + 1e-10), 0.0, 1.0)

    t1 = torch.where((t2 == 0.0) | (t2 == 1.0), torch.clamp((t2 * gamma - delta) / (alpha + 1e-10), 0.0, 1.0), t1)

    c1 = s1 + t1 * l1
    c2 = s2 + t2 * l2

    point_distance = ((c1 - c2) ** 2).sum(dim=-1)

    return point_distance - (r1 + r2) ** 2

def signed_distance_capsule_ball(s1: Float[Tensor, "*batch 3"], e1: Float[Tensor, "*batch 3"], r1: float,
                                 s2: Float[Tensor, "*batch 3"], r2: float) \
        -> Float[Tensor, "*batch"]:
    """
    Computes the signed squared distance between a capsule and a ball.

    Args:
        s1: Start point of the capsule segment.
        e1: End point of the capsule segment.
        r1: Radius of the capsule.
        s2: Center of the ball.
        r2: Radius of the ball.

    Returns:
        Signed squared distance
    """
    l1 = e1 - s1
    v = s2 - s1

    dot_v_l = (v * l1).sum(dim=-1, keepdim=True)
    len_sq_l = (l1 * l1).sum(dim=-1, keepdim=True)
    t = torch.clamp(dot_v_l / (len_sq_l + 1e-10), 0.0, 1.0)
    closest_point = s1 + t * l1

    point_distance = ((closest_point -s2) ** 2).sum(dim=-1)

    return point_distance - (r1 + r2) ** 2

PAIR_COMBINATIONS = [torch.triu_indices(2 * dof, 2 * dof, offset=2) for dof in range(1, 9)]


# #@jaxtyped(typechecker=beartype)
def collision_check(mdh: Float[Tensor, "*batch dofp1 3"],
                    poses: Float[Tensor, "*batch dofp1 4 4"],
                    radius: float = LINK_RADIUS,
                    debug=False) -> Bool[Tensor, "*batch"] | Float[Tensor, "*batch"]:
    """
    Compute whether the robot is in self-collision for each batch element.

    Args:
        mdh: Modified DH parameters [alpha, a, d, theta]
        poses: Homogeneous transforms for each joint (world frame)
        radius: Capsule radius
        debug: Whether to return the relevant signed distances directly
    Returns:
        A boolean indicating whether each configuration is in collision.
    """
    global PAIR_COMBINATIONS
    if mdh.device != PAIR_COMBINATIONS[0].device:
        PAIR_COMBINATIONS = [pair.to(mdh.device) for pair in PAIR_COMBINATIONS]

    *batch_shape, dof, _ = mdh.shape

    s_all, e_all = get_capsules(mdh, poses)

    # Capsule pair combinations
    i_idx, j_idx = PAIR_COMBINATIONS[dof - 1]
    num_pairs = i_idx.shape[0]

    # Gather capsule endpoints
    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)

    # Ignore distances with missing capsules
    collisions = (torch.norm(s1 - e1, dim=-1) > EPS) & (torch.norm(s2 - e2, dim=-1) > EPS)
    # Ignore adjacent capsules
    collisions &= torch.norm(e1 - s2, dim=-1) > EPS

    # Compute signed distances for inner parts of the kinematic chain
    distances = signed_distance_capsule_capsule(s1, e1, radius, s2, e2, radius)

    # Compute signed distances for the start and end of the kinematic chain (within 6 capsules one has to be nonzero)
    expansion_shape = s_all.shape
    s_end = torch.cat([s_all, s_all], dim=-2)
    e_end = torch.cat([e_all, e_all], dim=-2)
    c_end = torch.cat([ s_all[..., :1, :].expand(*expansion_shape), e_all[..., -1:, :].expand(*expansion_shape)], dim=-2)
    distance_end = signed_distance_capsule_ball(s_end, e_end, radius, c_end, radius)
    collisions_end = torch.norm(e_end - s_end, dim=-1) > EPS
    # We are only interested in the second and second to last capsule
    cum_sum = torch.cumsum(collisions_end, dim=-1)
    collisions_end &= (cum_sum == 2) | (cum_sum == (cum_sum[..., -1:] - 1))

    if not debug:
        collisions &= distances < 0.0
        collisions = collisions.any(dim=-1).reshape(batch_shape)
        collisions_end &= distance_end < 0.0
        collisions_end = collisions_end.any(dim=-1)
        return collisions | collisions_end
    else:
        critical_distance = torch.where(collisions, distances, torch.ones_like(distances)*torch.inf).min(dim=-1).values.reshape(batch_shape)
        critical_distance_end = torch.where(collisions_end, distance_end, torch.ones_like(distance_end)*torch.inf).min(dim=-1).values.reshape(batch_shape)
        return torch.stack([critical_distance, critical_distance_end], dim=-1).min(dim=-1).values

