import torch.utils.dlpack

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Bool, Int

import jax
import jax.numpy as jnp
import optax
import jax.dlpack

import paper_archive.nrm_jax.se3 as jax_se3
import paper_archive.nrm_jax.kinematics as jax_kinematics
from paper_archive.nrm_jax.self_collision import collision_check, EPS, LINK_RADIUS

@jax.custom_vjp
def squasher(param):
    mask = (jnp.abs(param) >= 2 * LINK_RADIUS).astype(param.dtype)
    return param * mask


def squasher_fwd(param):
    return squasher(param), None


def squasher_bwd(res, g):
    return (g,)


squasher.defvjp(squasher_fwd, squasher_bwd)


@jax.custom_vjp
def normaliser(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return param / safe_norm


def normaliser_fwd(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return (param / safe_norm), (param, l2_norm, safe_norm)


def normaliser_bwd(res, g):
    param, l2_norm, safe_norm = res
    safe_l2_norm = jnp.maximum(l2_norm, 1e-12)

    chain = jnp.where(
        jnp.any(jnp.abs(param) > EPS, axis=1, keepdims=True),
        param / safe_l2_norm,
        jnp.zeros_like(param)
    )
    grad_param = (g * safe_norm - chain * jnp.sum(g * param)) / (safe_norm ** 2)
    return (grad_param,)


normaliser.defvjp(normaliser_fwd, normaliser_bwd)


def preprocess(param):
    norm_param = normaliser(param)
    squashed = squasher(norm_param)
    return normaliser(squashed), norm_param


def loss_fn(lengths, alpha, task_poses):
    param, norm_lengths = preprocess(lengths)
    morph = jnp.concatenate([alpha, param], axis=1)
    bmorph = jnp.broadcast_to(morph, (task_poses.shape[0], *morph.shape))

    optimal_joints = jax_kinematics.numerical_inverse_kinematics(morph, task_poses)
    reached_poses = jax_kinematics.forward_kinematics(bmorph, optimal_joints)
    ee_poses = reached_poses[:, -1, :, :]
    dists = jax_se3.distance(ee_poses, task_poses)
    critical_distance = collision_check(bmorph, reached_poses, debug=True)

    pose_loss = jax.nn.relu(dists[:, 0] - EPS)
    self_collision_loss = 1e4 * jax.nn.relu(-critical_distance)
    reachability_score = pose_loss + self_collision_loss
    loss = jnp.mean(reachability_score)

    pose_error = jnp.mean(dists)
    self_collision = (critical_distance < 0.0).sum()

    return loss, (pose_loss.mean(), self_collision_loss.mean(), morph,reachability_score, pose_error, self_collision)


loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


def baseline(initial_morph: Float[Tensor, "dofp1 3"], task: Float[Tensor, "num_samples 4 4"], n_iter: int,
             logging: bool = True) \
        -> tuple[
            Float[Tensor, "n_iter"],
            Float[Tensor, "n_iter"],
            Float[Tensor, "n_iter"],
            Bool[Tensor, "n_iter num_samples"],
            Float[Tensor, "n_iter"],
            Int[Tensor, "n_iter"],
            Tensor
        ]:
    jax_task = jax.dlpack.from_dlpack(task.contiguous().clone())
    jax_alpha = jax.dlpack.from_dlpack(initial_morph[:, 0:1].contiguous().clone())
    jax_lengths = jax.dlpack.from_dlpack(initial_morph[:, 1:].contiguous().clone())

    optimizer = optax.adamw(learning_rate=0.01)
    opt_state = optimizer.init(jax_lengths)

    loss_list = []
    pose_loss_list = []
    self_collision_loss_list = []
    reachability = []
    pose_error_list = []
    self_collision_list = []
    morphs = []
    for i in tqdm(range(n_iter)):
        (loss, (pose_loss, self_collision_loss, morph,reachability_score, pose_error, self_collision)), grads = loss_and_grad_fn(jax_lengths, jax_alpha, jax_task)

        updates, opt_state = optimizer.update(grads, opt_state, jax_lengths)
        jax_lengths = optax.apply_updates(jax_lengths, updates)

        # Logging
        if logging:
            loss_list += [loss.item()]
            pose_loss_list += [pose_loss.item()]
            self_collision_loss_list += [self_collision_loss.item()]
            reachability += [torch.from_dlpack(jax.lax.stop_gradient(reachability_score)).cpu() == 0.0]
            pose_error_list += [pose_error.item()]
            self_collision_list += [self_collision.item()]
            morphs += [torch.from_dlpack(jax.lax.stop_gradient(morph)).cpu()]

    return (torch.tensor(loss_list),
            torch.tensor(pose_loss_list),
            torch.tensor(self_collision_loss_list),
            torch.stack(reachability, dim=0),
            torch.tensor(pose_error_list),
            torch.tensor(self_collision_list),
            torch.stack(morphs, dim=0))

if __name__ == "__main__":
    import pickle
    from pathlib import Path
    from plotly.subplots import make_subplots

    from nrm.visualisation import visualise_workspace
    from paper_archive.utils import bootstrap_mean_ci
    from nrm.dataset.morphology import sample_morph
    import nrm.dataset.se3 as se3

    save_dir = Path(__file__).parent / "data" / "base"
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 1000

    device = torch.device("cuda")

    loss_list = []
    pose_loss_list = []
    self_collision_loss_list = []
    reachability_list = []
    pose_error_list = []
    self_collision_list = []

    last_reachability = None
    last_morph = None

    for s in range(100):
        torch.manual_seed(s)
        task = se3.random_ball(1000, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])).to(device)
        initial_morph = sample_morph(1, 6, False, device)[0]

        loss, pose_loss, self_collision_loss, reachability, pose_error, self_collision, morph = baseline(initial_morph, task, 100)

        loss_list += [loss]
        pose_loss_list += [pose_loss]
        self_collision_loss_list += [self_collision_loss]
        reachability_list += [reachability.sum(dim=1)]
        pose_error_list += [pose_error]
        self_collision_list += [self_collision]

        last_reachability = reachability[-1]
        last_morph = morph[-1]

    loss = bootstrap_mean_ci(torch.stack(loss_list).numpy())
    pose_loss = bootstrap_mean_ci(torch.stack(pose_loss_list).numpy())
    self_collision_loss = bootstrap_mean_ci(torch.stack(self_collision_loss_list).numpy())
    reachability = bootstrap_mean_ci(torch.stack(reachability_list).numpy())
    pose_error = bootstrap_mean_ci(torch.stack(pose_error_list).numpy())
    self_collision = bootstrap_mean_ci(torch.stack(self_collision_list).numpy())

    pickle.dump(loss, open(save_dir / "loss.pkl", "wb"))
    pickle.dump(pose_loss, open(save_dir / "pose_loss.pkl", "wb"))
    pickle.dump(self_collision_loss, open(save_dir / "self_collision_loss.pkl", "wb"))
    pickle.dump(reachability, open(save_dir / "reachability.pkl", "wb"))
    pickle.dump(pose_error, open(save_dir / "pose_error.pkl", "wb"))
    pickle.dump(self_collision, open(save_dir / "self_collision.pkl", "wb"))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsmath}",

        "axes.labelsize": 34,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34,
        "legend.fontsize": 34,
        "axes.titlesize": 34,
        "lines.linewidth": 3,
    })

    fig, ax = plt.subplots(2, 1, figsize=(15, 20))


    colors = sns.color_palette("colorblind", 3)

    ln1 = ax[0].plot(reachability[0], color=colors[0], label="Reachability")
    ln2 = ax[0].plot(self_collision[0], color=colors[1], label=r"\# Self Collision")
    ax02 = ax[0].twinx()
    ln3 = ax02.plot(pose_error[0], color=colors[2], label="Pose Error")

    ax[0].set_ylim(0.0, num_samples)
    ax[0].set_ylabel("Counts")
    ax02.set_ylabel("Error Value")
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax[0].legend(lines, labels, loc='upper right')

    ax[1].plot(loss[0], label="Loss", alpha=0.7)
    ax[1].plot(pose_loss[0], label="Pose Loss", alpha=0.7)
    ax[1].plot(self_collision_loss[0], label="Self-collision Loss", alpha=0.7)
    ax[1].legend()

    for i in range(len(ax)):
        ax[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    visualise_workspace(last_morph.cpu(), task.cpu(), last_reachability)