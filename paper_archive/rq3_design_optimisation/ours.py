import torch

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Int, Bool

import nrm.dataset.se3 as se3

from nrm.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from nrm.dataset.self_collision import collision_check
from nrm.dataset.self_collision import LINK_RADIUS, EPS
from nrm.model import MLP


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        # Keep the hard mask for the forward pass
        mask = (param.abs() >= 2 * LINK_RADIUS).float()
        return param * mask

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: Pass the gradient exactly as is
        # This prevents the gradient from vanishing when the parameter is 0
        return grad_output


class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output):
        param, l2_norm, norm = ctx.saved_tensors

        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param)
        )

        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def preprocess(param):
    norm_param = Normaliser.apply(param)
    squashed = SquasherSTE.apply(norm_param)
    return Normaliser.apply(squashed), norm_param


def ours(initial_morph: Float[Tensor, "dofp1 3"], task: Float[Tensor, "num_samples 4 4"], n_iter: int,
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
    task_vec = se3.to_vector(task)
    alpha = initial_morph[:, 0:1].clone()

    lengths = initial_morph[:, 1:]
    lengths.requires_grad = True

    loss_list = []
    pose_loss_list = []
    self_collision_loss_list = []
    reachability = []
    pose_error_list = []
    self_collision_list = []
    morphs = []

    optimizer = torch.optim.AdamW([lengths], lr=0.01)
    model = MLP.from_id(13).to(initial_morph.device)
    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()

        param, norm_lengths = preprocess(lengths)

        morph = torch.cat([alpha, param], dim=1)
        bmorph = morph.unsqueeze(0).expand(task.shape[0], -1, -1)
        logit = model(bmorph, task_vec)

        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, torch.ones_like(logit))

        loss.backward()
        optimizer.step()

        # Logging
        if logging:
            with torch.no_grad():
                loss_list += [loss.item()]
                joints = numerical_inverse_kinematics(morph, task)[0]
                reached_pose = forward_kinematics(bmorph, joints)
                dists = se3.distance(reached_pose[:, -1, :, :], task).squeeze(-1)
                critical_distance = collision_check(bmorph, reached_pose, debug=True)

                pose_loss = torch.relu(dists - EPS)
                self_collision_loss = 1e4 * torch.relu(-critical_distance)
                reachability_score = pose_loss + self_collision_loss

                pose_loss_list += [pose_loss.mean().item()]
                self_collision_loss_list += [self_collision_loss.sum().item()]
                reachability += [reachability_score.cpu() == 0.0]
                pose_error_list += [dists.mean().item()]
                self_collision_list += [(critical_distance < 0.0).sum().item()]
                morphs += [morph.detach().clone().cpu()]
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

    save_dir = Path(__file__).parent / "data" / "ours"
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

        loss, pose_loss, self_collision_loss, reachability, pose_error, self_collision, morph = ours(initial_morph, task, 100)

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
