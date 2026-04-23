import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

import paper_archive.nrm_jax.r3 as r3
import paper_archive.nrm_jax.so3 as so3


# @jax.custom_vjp
# def safe_sqrt(x):
#     return jnp.sqrt(x)
#
#
# def safe_sqrt_fwd(x):
#     return safe_sqrt(x), x
#
#
# def safe_sqrt_bwd(x, g):
#     return (g / 2 * x + 1e-11,)
#
#
# safe_sqrt.defvjp(safe_sqrt_fwd, safe_sqrt_bwd)

@jax.custom_jvp
def safe_sqrt(x):
    return jnp.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents

    primal_out = safe_sqrt(x)

    # Derivative of sqrt(x) is 1 / (2 * sqrt(x)). We use primal_out (which is sqrt(x))
    # and add an epsilon to prevent division by zero.
    tangent_out = x_dot / (2.0 * jnp.maximum(primal_out, 1e-11))

    return primal_out, tangent_out


# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Array, "*batch 4 4"], x2: Float[Array, "*batch 4 4"]) -> Float[Array, "*batch 1"]:
    r"""
    Pose distance arising from the unique left-invariant riemannian metric for SE(3) that produces physically meaningful
    accelerations plus a weighting between translation and rotation.

    Args:
        x1: First homogeneous transformation.
        x2: Second homogeneous transformation.

    Returns:
        SE(3) distance between x1 and x2.

    Notes:
        Since the maximum rotational distance is \pi and the maximum translational distance in our setting is 2,
        we weigh the distances "equal" importance and also such that the maximum distance between two cells is 1.
    """
    t1 = x1[..., :3, 3]
    r1 = x1[..., :3, :3]
    t2 = x2[..., :3, 3]
    r2 = x2[..., :3, :3]

    return safe_sqrt(r3.distance(t1, t2) / 8 + so3.distance(r1, r2) ** 2 / (2 * jnp.pi ** 2))


def from_vector(vec: Float[Array, "*batch 9"]) -> Float[Array, "*batch 4 4"]:
    """
    Convert 9D vector representation to 4x4 homogeneous transformation matrix

    Args:
        vec: 9D vector representation

    Returns:
        Homogeneous transformation matrix
    """
    translation = vec[..., :3]
    rotation_cont = vec[..., 3:]
    batch_shape = vec.shape[:-1]

    rot_matrix = so3.from_vector(rotation_cont)
    top_block = jnp.concatenate([rot_matrix, translation[..., None]], axis=-1)

    bottom_row = jnp.array([0.0, 0.0, 0.0, 1.0])
    bottom_row = jnp.broadcast_to(bottom_row, (*batch_shape, 1, 4))

    homogeneous = jnp.concatenate([top_block, bottom_row], axis=-2)
    return homogeneous


# @jaxtyped(typechecker=beartype)
def exp(pose: Float[Array, "*batch 4 4"], tangent: Float[Array, "*batch 6"]) -> Float[Array, "*batch 4 4"]:
    """
    Differential geometry version of addition.

    Args:
        pose: Pose.
        tangent: Tangent vector.

    Returns:
        Moves from pose along the tangent vector.

    Notes:
        In Euclidean space, 𝑎𝑑𝑑𝑖𝑡𝑖𝑜𝑛 is a tool which takes two points 𝑝1,𝑝2, “adds” them, and generates a third, larger point
        𝑝3. Addition gives us a way to “move forward” in Euclidean space. On manifolds, the 𝑒𝑥𝑝𝑜𝑛𝑒𝑛𝑡𝑖𝑎𝑙 provides a tool,
        which “takes the exponential of the tangent vector at point 𝑝” to generate a third point on the manifold.
        The exponential does this by
        1) identifying the unique geodesic 𝛾 that goes through 𝑝 and 𝑣𝑝,
        2) identifying the “length” 𝑙 of the tangent vector 𝑣𝑝, and
        3) calculating another point 𝑝′ along 𝛾⁡(𝑡) that is a “distance” 𝑙 from the initial point 𝑝.
        Note again that the notion of “length” and “distance” is different on a manifold than it is in Euclidean space
        and that quantifying length is not something that we will be able to do without specifying a metric.
        [Source https://geomstats.github.io/notebooks/02_foundations__connection_riemannian_metric.html]
    """
    new_position = r3.exp(pose[..., :3, 3], tangent[..., :3])
    new_orientation = so3.exp(pose[..., :3, :3], tangent[..., 3:])

    new_pose = pose.at[..., :3, 3].set(new_position)
    new_pose = new_pose.at[..., :3, :3].set(new_orientation)
    return new_pose
