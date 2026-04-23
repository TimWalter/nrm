import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Array, "*batch 3"], x2: Float[Array, "*batch 3"]) -> Float[Array, "*batch 1"]:
    """
    Euclidean distance between vectors.

    Args:
        x1: First vector.
        x2: Second vector.

    Returns:
        Euclidean distance between vector x1 and x2.
    """
    return jnp.sum(jnp.square(x1 - x2), axis=-1, keepdims=True)


# @jaxtyped(typechecker=beartype)
def log(position1: Float[Array, "*batch 3"], position2: Float[Array, "*batch 3"]) -> Float[Array, "*batch 3"]:
    """
    Differential geometry version of subtraction.

    Args:
        position1: First position.
        position2: Second position.

    Returns:
        The tangent vector at position1 pointing to position2.

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
    return position2 - position1

# @jaxtyped(typechecker=beartype)
def exp(position: Float[Array, "*batch 3"], tangent: Float[Array, "*batch 3"]) -> Float[Array, "*batch 3"]:
    """
    Differential geometry version of addition.

    Args:
        position: Position.
        tangent: Tangent vector.

    Returns:
        Moves from position along the tangent vector.

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
    return position + tangent