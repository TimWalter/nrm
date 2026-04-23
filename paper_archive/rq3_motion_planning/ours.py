import torch

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Int, Bool

import nrm.dataset.se3 as se3
from nrm.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from nrm.dataset.self_collision import collision_check, EPS
from nrm.model import MLP


def from_index(rot_vec: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 3 3"]:
    angle_sq = torch.sum(torch.square(rot_vec), dim=-1, keepdim=True)
    is_small = angle_sq < 1e-12

    safe_angle_sq = torch.where(is_small, 1.0, angle_sq)
    angle = torch.sqrt(safe_angle_sq)

    sin_coeff = torch.where(is_small,
                            1.0 - angle_sq / 6.0,
                            torch.sin(angle) / angle)
    cos_coeff = torch.where(is_small,
                            0.5 - angle_sq / 24.0,
                            (1.0 - torch.cos(angle)) / safe_angle_sq)


    x, y, z = rot_vec[..., 0], rot_vec[..., 1], rot_vec[..., 2]
    zero = torch.zeros_like(x)

    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1)
    ], dim=-2)

    I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype).expand_as(K)

    return I + sin_coeff.unsqueeze(-1) * K + cos_coeff.unsqueeze(-1) * (K @ K)


# @jaxtyped(typechecker=beartype)
def differentiable_exp(pose: Float[Tensor, "*batch 4 4"], tangent: Float[Tensor, "*batch 6"]) -> Float[
    Tensor, "*batch 4 4"]:
    new_translation = pose[..., :3, 3] + tangent[..., :3]
    new_rotation = pose[..., :3, :3] @ from_index(tangent[..., 3:])
    top = torch.cat([new_rotation, new_translation.unsqueeze(-1)], dim=-1)
    bottom = pose[..., 3:, :]

    return torch.cat([top, bottom], dim=-2)


def ours(morph: Float[Tensor, "dofp1 3"], target_trajectory: Float[Tensor, "num_samples 4 4"], n_iter: int,
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
    bmorph = morph.unsqueeze(0).expand(target_trajectory.shape[0], -1, -1)

    offset = torch.zeros((target_trajectory.shape[0], 6), device=morph.device, requires_grad=True)
    optimizer = torch.optim.AdamW([offset], lr=0.002)

    loss_list = []
    prediction_loss_list = []
    deviation_loss_list = []
    reachability = []
    pose_error_list = []
    self_collision_list = []

    model = MLP.from_id(13).to(morph.device)

    prediction_loss_c = 10.0
    deviation_loss_c = 0.5

    for i in tqdm(range(n_iter)):
        optimizer.zero_grad()

        trajectory = differentiable_exp(target_trajectory, offset)

        trajectory_vector = se3.to_vector(trajectory)
        logit = model(bmorph, trajectory_vector)
        prediction_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, torch.ones_like(logit))
        deviation_loss = se3.distance(target_trajectory, trajectory).mean()

        loss = (prediction_loss_c * prediction_loss +
                deviation_loss_c * deviation_loss)
        loss.backward()

        optimizer.step()

        # Logging
        if logging:
            with torch.no_grad():
                loss_list += [loss.item()]
                prediction_loss_list += [prediction_loss_c * prediction_loss.item()]
                deviation_loss_list += [deviation_loss_c * deviation_loss.item()]

                joints = numerical_inverse_kinematics(morph, trajectory)[0]
                reached_pose = forward_kinematics(bmorph, joints)
                dists = se3.distance(reached_pose[:, -1, :, :], trajectory).squeeze(-1)
                critical_distance = collision_check(bmorph, reached_pose, debug=True)

                pose_loss = torch.relu(dists - EPS)
                self_collision_loss = 1e4 * torch.relu(-critical_distance)
                reachability_score = pose_loss + self_collision_loss

                reachability += [reachability_score.cpu() == 0.0]
                pose_error_list += [dists.mean().item()]
                self_collision_list += [(critical_distance < 0.0).sum().item()]
    return (torch.tensor(loss_list),
            torch.tensor(prediction_loss_list),
            torch.tensor(deviation_loss_list),
            torch.stack(reachability, dim=0),
            torch.tensor(pose_error_list),
            torch.tensor(self_collision_list),
            trajectory.cpu().detach())


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    from plotly.subplots import make_subplots

    from nrm.visualisation import visualise_trajectories
    from paper_archive.utils import bootstrap_mean_ci
    from nrm.dataset.morphology import sample_morph, get_joint_limits
    from nrm.dataset.reachability_manifold import sample_reachable_poses

    save_dir = Path(__file__).parent / "data" / "ours"
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 10

    device = torch.device("cuda")

    loss_list = []
    prediction_loss_list = []
    deviation_loss_list = []
    reachability_list = []
    pose_error_list = []
    self_collision_list = []

    last_reachability = None
    last_trajectory = None
    for s in range(100):
        torch.manual_seed(s)
        morph = sample_morph(1, 6, False, device)[0]
        joint_limits = get_joint_limits(morph)
        start = sample_reachable_poses(morph, joint_limits)[0]
        while start.shape[0] == 0:
            start = sample_reachable_poses(morph, joint_limits)[0]
        end = sample_reachable_poses(morph, joint_limits)[0]
        while end.shape[0] == 0:
            end = sample_reachable_poses(morph, joint_limits)[0]

        tangent = se3.log(start, end)
        t = torch.linspace(0, 1, num_samples, device=tangent.device).view(-1, 1)
        target_trajectory = se3.exp(start.repeat(num_samples, 1, 1), t * tangent)

        loss, prediction_loss, deviation_loss, reachability, pose_error, self_collision, trajectory = ours(morph, target_trajectory, 100)
        loss_list += [loss]
        prediction_loss_list += [prediction_loss]
        deviation_loss_list += [deviation_loss]
        reachability_list += [reachability.sum(dim=1)]
        pose_error_list += [pose_error]
        self_collision_list += [self_collision]

        last_reachability = reachability[-1]
        last_trajectory = trajectory

    loss = bootstrap_mean_ci(torch.stack(loss_list).numpy())
    prediction_loss = bootstrap_mean_ci(torch.stack(prediction_loss_list).numpy())
    deviation_loss = bootstrap_mean_ci(torch.stack(deviation_loss_list).numpy())
    reachability = bootstrap_mean_ci(torch.stack(reachability_list).numpy())
    pose_error = bootstrap_mean_ci(torch.stack(pose_error_list).numpy())
    self_collision = bootstrap_mean_ci(torch.stack(self_collision_list).numpy())

    pickle.dump(loss, open(save_dir / "loss.pkl", "wb"))
    pickle.dump(prediction_loss, open(save_dir / "prediction_loss.pkl", "wb"))
    pickle.dump(deviation_loss, open(save_dir / "deviation_loss.pkl", "wb"))
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
    ax[1].plot(prediction_loss[0], label="Prediction Loss", alpha=0.7)
    ax[1].plot(deviation_loss[0], label="Deviation Loss", alpha=0.7)
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

    visualise_trajectories(morph.cpu(), [target_trajectory.cpu(), trajectory.cpu()],
                           [numerical_inverse_kinematics(morph.cpu(), target_trajectory.cpu())[1] != -1,
                            last_reachability], ["Target", "Ours"])
