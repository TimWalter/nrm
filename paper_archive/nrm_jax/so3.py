import os
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from scipy.spatial.transform import Rotation

enabled = os.environ.get("SCIPY_ARRAY_API", "").lower() in ("1")
if not enabled:
    raise RuntimeError(
        "SCIPY_ARRAY_API must be enabled! "
        "Please run: export SCIPY_ARRAY_API=1 before importing this module"
    )


# @jax.custom_vjp
# def safe_arccos(x):
#     """arccos with exact forward pass, finite-gradient backward pass."""
#     return jnp.arccos(x)
#
#
# def safe_arccos_fwd(x):
#     return safe_arccos(x), x
#
#
# def safe_arccos_bwd(x, g):
#     denom = jnp.sqrt(jnp.maximum(1.0 - x ** 2, 1e-7))
#     return (-g / denom,)
#
#
# safe_arccos.defvjp(safe_arccos_fwd, safe_arccos_bwd)

@jax.custom_jvp
def safe_arccos(x):
    """arccos with exact forward pass, finite-gradient backward pass."""
    return jnp.arccos(x)

@safe_arccos.defjvp
def safe_arccos_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents

    primal_out = safe_arccos(x)

    # Safe denominator to prevent division by zero at the boundaries (-1.0, 1.0)
    denom = jnp.sqrt(jnp.maximum(1.0 - x ** 2, 1e-7))
    tangent_out = -x_dot / denom

    return primal_out, tangent_out


# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Array, "*batch 3 3"],
             x2: Float[Array, "*batch 3 3"]) -> Float[Array, "*batch 1"]:
    """
    Geodesic distance between rotation matrices.

    Args:
        x1: First rotation matrix.
        x2: Second rotation matrix.

    Returns:
        Geodesic distance between x1 and x2.
    """
    r_err = jnp.matmul(jnp.swapaxes(x1, -1, -2), x2)
    trace = r_err[..., 0, 0] + r_err[..., 1, 1] + r_err[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    rot_err = safe_arccos(cos_angle)
    return rot_err[..., None]


# @jaxtyped(typechecker=beartype)
def from_vector(vec: Float[Array, "*batch 6"]) -> Float[Array, "*batch 3 3"]:
    """
    Convert continuous 6D rotation representation to 3x3 rotation matrix.

    Args:
        vec: 6D rotation representation

    Returns:
        Rotation matrix
    """
    r1 = vec[..., :3]
    r2 = vec[..., 3:]
    r3 = jnp.cross(r1, r2, axis=-1)

    matrix = jnp.stack([r1, r2, r3], axis=-1)
    return matrix


@jax.jit
def to_index(orientation: Float[Array, "batch 3 3"]) -> Float[Array, "batch 3"]:
    """
    Convert 3x3 rotation matrix to rotation vector (axis-angle representation), which we use for indexing the lookup.

    Args:
        orientation: Rotation matrix

    Returns:
        Rotation vector
    """
    return Rotation.from_matrix(orientation, assume_valid=True).as_rotvec()


@jax.jit
def from_index(rot_vec: Float[Array, "batch 3"]) -> Float[Array, "batch 3 3"]:
    """
    Convert the rotation vector, which we use for indexing the lookup, to 3x3 rotation matrix.

    Args:
        rot_vec: Rotation vector

    Returns:
        Rotation matrix
    """
    return Rotation.from_rotvec(rot_vec).as_matrix()


# @jaxtyped(typechecker=beartype)
def log(orientation1: Float[Array, "*batch 3 3"], orientation2: Float[Array, "*batch 3 3"]) -> Float[
    Array, "*batch 3"]:
    """
    Differential geometry version of addition.

    Args:
        orientation1: First orientation.
        orientation2: Second orientation.

    Returns:
        The tangent vector at orientation1 pointing to orientation2.

    Notes:
        In Euclidean space, 𝑠𝑢𝑏𝑡𝑟𝑎𝑐𝑡𝑖𝑜𝑛 is an operation which allows us to take the third point 𝑝3 and one of the
        initial points 𝑝1 and extract the other initial point 𝑝2. Similarly, the 𝑙𝑜𝑔𝑎𝑟𝑖𝑡ℎ𝑚 allows us to take the
        final point 𝑝′ and the initial point 𝑝 to extract the tangent vector 𝑣𝑝 at the initial point.
        The logarithm is able to do this by
        1) identifying the unique geodesic 𝛾 that connects the two points
        2) calculating the “length” of that geodesic
        3) generating the unique tangent vector at 𝑝, with a “length” equal to that of the geodesic.
        Again, remember that “length” is not something that we can quantify without specifying a metric.
        A key point here is that if you know a point and a tangent vector at that point, you can calculate a unique
        geodesic that goes through that point. Similarly, if you know the point and geodesic, you should be able to
        extract the unique tangent vector that produced that geodesic.
        [Source https://geomstats.github.io/notebooks/02_foundations__connection_riemannian_metric.html]
    """
    return to_index_differentiable(jnp.swapaxes(orientation1, -1, -2) @ orientation2)


# @jaxtyped(typechecker=beartype)
def exp(orientation: Float[Array, "*batch 3 3"], tangent: Float[Array, "*batch 3"]) -> Float[Array, "*batch 3 3"]:
    """
    Differential geometry version of addition.

    Args:
        orientation: Orientation.
        tangent: Tangent vector.

    Returns:
        Moves from orientation along the tangent vector.

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
    return orientation @ from_index(tangent)

# @jaxtyped(typechecker=beartype)
def to_index_differentiable(orientation: Float[Array, "*batch 3 3"]) -> Float[Array, "*batch 3"]:
    """
    Differentiable conversion from SO(3) rotation matrix to rotation vector (axis-angle).
    Replaces the scipy.spatial.transform.Rotation dependency.
    """
    # 1. Compute trace and angle
    trace = jnp.trace(orientation, axis1=-2, axis2=-1)

    # Clip to avoid NaNs in arccos and to dodge the exact pi-singularity (where cos(pi) = -1)
    cos_angle = jnp.clip((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = safe_arccos(cos_angle)[..., None]

    # 2. Extract the skew-symmetric part (proportional to the axis of rotation)
    vec = jnp.stack([
        orientation[..., 2, 1] - orientation[..., 1, 2],
        orientation[..., 0, 2] - orientation[..., 2, 0],
        orientation[..., 1, 0] - orientation[..., 0, 1]
    ], axis=-1)

    # 3. Calculate scale factor: angle / (2 * sin(angle))
    # We use a Taylor expansion for small angles to prevent division by zero (0/0)
    scale = jnp.where(
        angle < 1e-3,
        0.5 + (angle ** 2) / 12.0,
        angle / (2.0 * jnp.sin(angle))
    )

    return vec * scale