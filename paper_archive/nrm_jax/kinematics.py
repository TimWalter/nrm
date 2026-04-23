import equinox
import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from jax import Array
from jaxtyping import Float

import paper_archive.nrm_jax.se3 as se3
import paper_archive.nrm_jax.so3 as so3
import paper_archive.nrm_jax.r3 as r3

from paper_archive.nrm_jax.self_collision import collision_check

# @jaxtyped(typechecker=beartype)
def transformation_matrix(alpha: Float[Array, "*batch 1"], a: Float[Array, "*batch 1"], d: Float[Array, "*batch 1"],
                          theta: Float[Array, "*batch 1"]) -> Float[Array, "*batch 4 4"]:
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
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    zero = jnp.zeros_like(alpha)
    one = jnp.ones_like(alpha)

    row1 = jnp.concatenate([ct, -st, zero, a], axis=-1)
    row2 = jnp.concatenate([st * ca, ct * ca, -sa, -d * sa], axis=-1)
    row3 = jnp.concatenate([st * sa, ct * sa, ca, d * ca], axis=-1)
    row4 = jnp.concatenate([zero, zero, zero, one], axis=-1)
    return jnp.stack([row1, row2, row3, row4], axis=-2)


# @jaxtyped(typechecker=beartype)
def forward_kinematics(mdh: Float[Array, "*batch dofp1 3"],
                       theta: Float[Array, "*batch dofp1 1"]) -> Float[Array, "*batch dofp1 4 4"]:
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
    pose_shape = mdh.shape[:-2] + (4, 4)
    pose = jnp.broadcast_to(jnp.eye(4), pose_shape)
    for i in range(mdh.shape[-2]):
        pose = pose @ transforms[..., i, :, :]
        poses.append(pose)

    return jnp.stack(poses, axis=-3)



def numerical_inverse_kinematics(
        morph: Float[Array, "dofp1 3"],
        target_poses: Float[Array, "n_poses 4 4"],
        *,
        num_seeds: int = 10,
        key: Array | None = None,
) -> Float[Array, "n_poses dofp1 1"]:
    if key is None:
        key = jax.random.PRNGKey(42)

    dof = morph.shape[0] - 1
    n_poses = target_poses.shape[0]
    total = n_poses * num_seeds

    init_joints = jax.random.uniform(key, (total, dof, 1), minval=-jnp.pi, maxval=jnp.pi)
    tiled_poses = jnp.repeat(target_poses, num_seeds, axis=0)

    def solve_one(init, target_pose):
        def residual(joints, args):
            full_joints = jnp.concatenate([joints, jnp.zeros_like(joints[0:1])], axis=0)
            reached = forward_kinematics(morph, full_joints)[-1]
            t_err = r3.log(reached[:3, 3], target_pose[:3, 3])
            r1, r2 = reached[:3, :3], target_pose[:3, :3]
            r_err = r1 @ so3.log(r1, r2)
            return jnp.concatenate([t_err, r_err])

        solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
        sol = optx.least_squares(
            fn=residual,
            solver=solver,
            y0=init,
            args=target_pose,
            max_steps=100,
            adjoint=optx.ImplicitAdjoint(),
            throw=False,
        )
        return sol.value

    solved_joints = jax.vmap(solve_one)(init_joints, tiled_poses)

    solved_joints = solved_joints.reshape(n_poses, num_seeds, dof, 1)
    full_joints = jnp.concatenate([solved_joints, jnp.zeros_like(solved_joints[..., 0:1, :])], axis=-2)

    tiled_morph = jnp.broadcast_to(morph, (n_poses, num_seeds, dof+1, 3))
    reached_poses = forward_kinematics(tiled_morph, full_joints)

    errors = se3.distance(
        target_poses[:, None, :, :],
        reached_poses[..., -1, :, :],
    )[..., 0]
    self_collision = collision_check(tiled_morph, reached_poses)

    best_idx = jnp.argmin(jnp.where(self_collision, jnp.inf, errors), axis=1)
    return full_joints[jnp.arange(n_poses), best_idx]