import jax
import jax.numpy as jnp
import optimistix as optx
from colorlog import critical

from jax import Array
from jaxtyping import Float, Bool

import paper_archive.nrm_jax.se3 as se3
import paper_archive.nrm_jax.so3 as so3
import paper_archive.nrm_jax.r3 as r3
from nrm.dataset.self_collision import EPS, LINK_RADIUS


# @jaxtyped(typechecker=beartype)
def get_capsules(mdh: Float[Array, "*batch dofp1 3"],
                 poses: Float[Array, "*batch dofp1 4 4"]) -> tuple[
    Float[Array, "*batch 2*dofp1 3"],
    Float[Array, "*batch 2*dofp1 3"],
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
    dtype = mdh.dtype

    # Prepend the Identity matrix (Base Frame 0) to poses (poses currently contains [T1, T2, ..., TN]. We need [T0, T1, ..., TN].)
    identity = jnp.eye(4, dtype=dtype)
    identity_expanded = jnp.broadcast_to(identity, batch_shape + [1, 4, 4])
    poses = jnp.concatenate([identity_expanded, poses], axis=-3)

    # Starting point of the first capsule (a) is the pose from base
    s_a = poses[..., :-1, :3, 3]
    # End point of the second capsule (d) is the pose after
    e_d = poses[..., 1:, :3, 3]
    # The middle point is deducted by reversing the translation d along the z-axis
    z_axis = poses[..., 1:, :3, 2]
    d = mdh[..., 2][..., None]
    e_a = s_d = e_d - d * z_axis

    # Assemble the chain
    s_all = jnp.stack([s_a, s_d], axis=-2).reshape(batch_shape + [2 * dof, 3])
    e_all = jnp.stack([e_a, e_d], axis=-2).reshape(batch_shape + [2 * dof, 3])
    return s_all, e_all


# @jaxtyped(typechecker=beartype)
def signed_distance_capsule_capsule(s1: Float[Array, "*batch 3"], e1: Float[Array, "*batch 3"], r1: float,
                                    s2: Float[Array, "*batch 3"], e2: Float[Array, "*batch 3"], r2: float) \
        -> Float[Array, "*batch"]:
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

    alpha = jnp.sum(l1 * l1, axis=-1, keepdims=True)
    beta = jnp.sum(l2 * l2, axis=-1, keepdims=True)
    gamma = jnp.sum(l1 * l2, axis=-1, keepdims=True)
    delta = jnp.sum(l1 * ds, axis=-1, keepdims=True)
    epsilon = jnp.sum(l2 * ds, axis=-1, keepdims=True)

    det = alpha * beta - gamma ** 2

    t1 = jnp.clip((gamma * epsilon - beta * delta) / (det + 1e-10), 0.0, 1.0)
    t2 = jnp.clip((gamma * t1 + epsilon) / (beta + 1e-10), 0.0, 1.0)

    t1_fallback = jnp.clip((t2 * gamma - delta) / (alpha + 1e-10), 0.0, 1.0)
    t1 = jnp.where((t2 == 0.0) | (t2 == 1.0), t1_fallback, t1)

    c1 = s1 + t1 * l1
    c2 = s2 + t2 * l2

    point_distance = jnp.sum((c1 - c2) ** 2, axis=-1)

    return point_distance - (r1 + r2) ** 2


def signed_distance_capsule_ball(s1: Float[Array, "*batch 3"], e1: Float[Array, "*batch 3"], r1: float,
                                 s2: Float[Array, "*batch 3"], r2: float) \
        -> Float[Array, "*batch"]:
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

    dot_v_l = jnp.sum(v * l1, axis=-1, keepdims=True)
    len_sq_l = jnp.sum(l1 * l1, axis=-1, keepdims=True)
    t = jnp.clip(dot_v_l / (len_sq_l + 1e-10), 0.0, 1.0)
    closest_point = s1 + t * l1

    point_distance = jnp.sum((closest_point - s2) ** 2, axis=-1)

    return point_distance - (r1 + r2) ** 2


PAIR_COMBINATIONS = [jnp.triu_indices(2 * dof, k=2) for dof in range(1, 9)]


# @jaxtyped(typechecker=beartype)
def collision_check(mdh: Float[Array, "*batch dofp1 3"],
                    poses: Float[Array, "*batch dofp1 4 4"],
                    radius: float = LINK_RADIUS,
                    debug=False) -> Bool[Array, "*batch"] | Float[Array, "*batch"]:
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
    collisions = (jnp.linalg.norm(s1 - e1, axis=-1) > EPS) & (jnp.linalg.norm(s2 - e2, axis=-1) > EPS)
    # Ignore adjacent capsules
    collisions &= jnp.linalg.norm(e1 - s2, axis=-1) > EPS

    # Compute signed distances for inner parts of the kinematic chain
    distances = signed_distance_capsule_capsule(s1, e1, radius, s2, e2, radius)

    # Compute signed distances for the start and end of the kinematic chain
    expansion_shape = s_all.shape
    s_end = jnp.concatenate([s_all, s_all], axis=-2)
    e_end = jnp.concatenate([e_all, e_all], axis=-2)

    c_end_start = jnp.broadcast_to(s_all[..., :1, :], expansion_shape)
    c_end_last = jnp.broadcast_to(e_all[..., -1:, :], expansion_shape)
    c_end = jnp.concatenate([c_end_start, c_end_last], axis=-2)

    distance_end = signed_distance_capsule_ball(s_end, e_end, radius, c_end, radius)
    collisions_end = jnp.linalg.norm(e_end - s_end, axis=-1) > EPS

    cum_sum = jnp.cumsum(collisions_end, axis=-1)
    collisions_end &= (cum_sum == 2) | (cum_sum == (cum_sum[..., -1:] - 1))

    if not debug:
        collisions &= distances < 0.0
        collisions = collisions.any(axis=-1).reshape(batch_shape)
        collisions_end &= distance_end < 0.0
        collisions_end = collisions_end.any(axis=-1).reshape(batch_shape)
        return collisions | collisions_end
    else:
        critical_distance = jnp.where(collisions, distances, 1.0).min(axis=-1).reshape(batch_shape)
        critical_distance_end = jnp.where(collisions_end, distance_end, 1.0).min(axis=-1).reshape(batch_shape)

        return jnp.minimum(critical_distance, critical_distance_end)
