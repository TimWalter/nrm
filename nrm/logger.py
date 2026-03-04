import json
from typing import Type
from pathlib import Path

import optuna
import wandb
import torch
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from jaxtyping import Float, Bool
from torch import Tensor

import nrm.dataset.se3 as se3
from nrm.dataset.reachability_manifold import estimate_reachable_ball
from nrm.visualisation import get_pose_traces
from nrm.model import Model
from nrm.dataset.loader import TrainingSet, ValidationSet
from nrm.dataset.kinematics import inverse_kinematics, transformation_matrix
from nrm.visualisation import display_geodesic, display_slice, display_sphere


class Logger:
    def __init__(self,
                 device: torch.device,
                 model_class: Type[Model],
                 hyperparameter: dict,
                 epochs: int,
                 batch_size: int,
                 early_stopping: int,
                 lr: float,
                 trial: optuna.Trial | None,
                 training_set: TrainingSet,
                 validation_set: ValidationSet,
                 model: Model,
                 loss_function: torch.nn.modules.loss._Loss
                 ):
        self.device = device
        self.batch_size = batch_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        metadata = {"num_training_samples": len(training_set) * batch_size,
                    "num_validation_samples": len(validation_set) * batch_size,
                    "model_class": model_class.__name__,
                    "hyperparameter": hyperparameter,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "early_stopping": early_stopping,
                    "lr": lr}

        self.run = self.setup_wandb(metadata, trial)
        self.folder = self.save_metadata(metadata)

        # For graphics
        self.geodesic_morph, self.geodesic_poses, self.geodesic_labels = self.generate_geodesic()
        self.geodesic_filter = self._get_ood_filter_ood(self.geodesic_morph[0], self.geodesic_poses)
        self.slice_morph, self.slice_poses, self.slice_labels = self.generate_slice()
        self.slice_filter = self._get_ood_filter_ood(self.slice_morph[0], self.slice_poses)
        self.sphere_morph, self.sphere_poses, self.sphere_labels, self.sphere_radius = self.generate_sphere()
        self.sphere_filter = self._get_ood_filter_ood(self.sphere_morph[0], self.sphere_poses)
        # For metric
        self.boundary_morph = []
        self.boundary_pose = []
        self.boundary_label = []
        self.boundary_filter = []
        for i in range(100):
            morph, poses, labels = self.generate_geodesic(100, True)
            self.boundary_morph += [morph]
            self.boundary_pose += [poses]
            self.boundary_label += [labels]
            self.boundary_filter += [self._get_ood_filter_ood(morph[0], poses).float()]
        self.boundary_morph = torch.cat(self.boundary_morph, dim=0)
        self.boundary_pose = torch.cat(self.boundary_pose, dim=0)
        self.boundary_label = torch.cat(self.boundary_label, dim=0)
        self.boundary_filter = torch.cat(self.boundary_filter, dim=0)

        self.buffer = {}

        self.step = 0

    def setup_wandb(self, metadata: dict, trial: optuna.Trial) -> wandb.Run:
        # wandb.login(key="")
        run = wandb.init(project="Neural Reachability Manifolds",
                         config=metadata,
                         group="MLP",
                         tags=[metadata["model_class"]],
                         dir=Path(__file__).parent.parent / "wandb")

        if trial is not None:
            run.name = f"trial/{trial.number}/{run.name}"
        # wandb.watch(self.model, criterion=self.loss_function, log="all")

        return run

    def save_metadata(self, metadata: dict) -> Path:
        parts = self.run.name.split("-")
        folder = Path(__file__).parent.parent / "trained_models" / f"{parts[-1]}-{'-'.join(parts[:-1])}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        json.dump(metadata, open(folder / 'metadata.json', 'w'), indent=4)
        return folder

    def _get_one_robot(self, random: bool = False) -> tuple[
        Float[Tensor, "dofp1 3"],
        Float[Tensor, "batch 4 4"],
        Bool[Tensor, "batch"]]:
        if random:
            batch_idx = torch.randint(0, self.validation_set.num_batches, (1,)).item()
            morphs, poses, labels = self.validation_set[batch_idx]
            morph_ids = self.validation_set._get_batch(batch_idx)[:, 0].long()
        else:
            morphs, poses, labels = self.validation_set[0]
            morph_ids = self.validation_set._get_batch(0)[:, 0].long()

        for i in morph_ids.unique():
            mask = i == morph_ids
            morph = morphs[mask][0]
            pose = poses[mask]
            label = labels[mask]
            if label.sum() != label.shape[0] and label.sum() != 0:
                return morph, se3.from_vector(pose), label
        return self._get_one_robot(True)

    @staticmethod
    def _get_ood_filter_ood(morph: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> Bool[
        Tensor, "batch"]:
        poses = se3.from_vector(poses)

        centre, radius_s = estimate_reachable_ball(morph[:-1])
        mat = torch.linalg.inv(
            transformation_matrix(morph[-1, 0:1], morph[-1, 1:2], morph[-1, 2:3], torch.zeros_like(morph[-1, 0:1])))
        mask = ((poses @ mat)[:, :3, 3] - centre).norm(dim=-1) < radius_s

        return mask

    def generate_geodesic(self, num_samples:int = 1000, random: bool = False) -> tuple[
        Float[Tensor, "num_samples dofp1 3"],
        Float[Tensor, "num_samples 9"],
        Bool[Tensor, "num_samples"]]:
        geodesic_morph, pose, label = self._get_one_robot(random)

        start = pose[label][0]
        end = pose[~label][0]

        tangent = se3.log(start, end)
        t = torch.linspace(0, 1, num_samples).view(-1, 1)
        geodesic_poses = se3.exp(start.unsqueeze(0).repeat(num_samples, 1, 1), t * tangent)
        geodesic_labels = inverse_kinematics(geodesic_morph.double(), geodesic_poses.double())[1] != -1

        return (geodesic_morph.unsqueeze(0).expand(num_samples, -1, -1).clone().pin_memory(),
                se3.to_vector(geodesic_poses).pin_memory(), geodesic_labels)

    def generate_slice(self) -> tuple[Float[Tensor, "10000 dofp1 3"], Float[Tensor, "10000 9"], Bool[Tensor, "10000"]]:
        slice_morph, pose, label = self._get_one_robot()

        mat = transformation_matrix(slice_morph[0, 0:1],
                                    slice_morph[0, 1:2],
                                    slice_morph[0, 2:3],
                                    torch.zeros_like(slice_morph[0, 0:1]))
        torus_axis = torch.nn.functional.normalize(mat[:3, 2], dim=0)
        centre, radius = estimate_reachable_ball(slice_morph)
        fixed_axes = torch.argmax(torus_axis.abs())
        axes_mask = torch.ones(3, dtype=torch.bool)
        axes_mask[fixed_axes] = False
        axes_range = torch.linspace(-radius, radius, 100)

        anchor = pose[label][torch.median(pose[label][:, :3, 3].norm(dim=1), dim=0).indices]
        slice_poses = anchor.unsqueeze(0).expand(100 * 100, -1, -1).clone()
        slice_poses[:, :3, 3][:, axes_mask] = centre[axes_mask]

        slice_poses[:, :3, 3][:, axes_mask] += torch.stack(torch.meshgrid(axes_range, axes_range, indexing='ij'),
                                                           dim=-1).reshape(-1, 2)
        slice_labels = inverse_kinematics(slice_morph.double(), slice_poses.double())[1] != -1

        return slice_morph.unsqueeze(0).expand(10000, -1, -1).clone().pin_memory(), se3.to_vector(
            slice_poses).pin_memory(), slice_labels

    def generate_sphere(self) -> tuple[
        Float[Tensor, "10000 dofp1 3"],
        Float[Tensor, "10000 9"],
        Bool[Tensor, "10000"],
        float]:
        sphere_morph, pose, label = self._get_one_robot()

        centre, radius = estimate_reachable_ball(sphere_morph)

        sphere_anchor = pose[label][torch.median(pose[label][:, :3, 3].norm(dim=1), dim=0).indices]
        sphere_poses = sphere_anchor.unsqueeze(0).expand(100 * 100, -1, -1).clone()
        sphere_poses[:, :3, 3] = centre

        theta = torch.linspace(0, torch.pi, 100)
        phi = torch.linspace(0, 2 * torch.pi, 100)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

        x = sphere_anchor[:3, 3].norm() * torch.sin(theta_grid) * torch.cos(phi_grid)
        y = sphere_anchor[:3, 3].norm() * torch.sin(theta_grid) * torch.sin(phi_grid)
        z = sphere_anchor[:3, 3].norm() * torch.cos(theta_grid)

        sphere_poses[:, :3, 3] = centre + torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        sphere_labels = inverse_kinematics(sphere_morph.double(), sphere_poses.double())[1] != -1

        return sphere_morph.unsqueeze(0).expand(10000, -1, -1).clone().pin_memory(), se3.to_vector(
            sphere_poses).pin_memory(), sphere_labels, sphere_anchor[:3, 3].norm()

    def save_model(self):
        torch.save(self.model.state_dict(), self.folder / "checkpoint.pth")

    def __del__(self):
        self.run.finish()

    @staticmethod
    def compute_metrics(logit: Float[Tensor, "batch"], label: Bool[Tensor, "batch"]) -> dict:
        (true_positives, false_negatives), (false_positives, true_negatives) = binary_confusion_matrix(logit, label)
        accuracy = ((torch.nn.Sigmoid()(logit) > 0.5) == label).cpu().sum() / label.shape[0] * 100

        hist, bin_edges = np.histogram(torch.nn.Sigmoid()(logit).cpu().numpy(), bins=64, range=(0.0, 1.0))

        metrics = {'True Positives': true_positives,
                   'True Negatives': true_negatives,
                   'False Positives': false_positives,
                   'False Negatives': false_negatives,
                   'Accuracy': accuracy,
                   'F1 Score': 2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100 if true_positives + false_positives + false_negatives != 0 else 100,
                   'Predictions': wandb.Histogram(np_histogram=(hist, bin_edges)),
                   }

        return metrics

    def compute_boundary_metrics(self) -> dict:
        boundary_logits = self.model.predict(self.boundary_morph.to(self.device, non_blocking=True),
                                          self.boundary_pose.to(self.device, non_blocking=True)).cpu()
        metrics = self.compute_metrics(boundary_logits * self.boundary_filter, self.boundary_label)

        geodesic_logits = self.model.predict(self.geodesic_morph.to(self.device, non_blocking=True),
                                             self.geodesic_poses.to(self.device, non_blocking=True)).cpu()
        geodesic_pred = (torch.nn.Sigmoid()(geodesic_logits) > 0.5) & self.geodesic_filter
        metrics["Geodesic"] = display_geodesic([geodesic_pred, self.geodesic_labels],
                                               ["Prediction", "Truth"], True)

        slice_logits = self.model.predict(self.slice_morph.to(self.device, non_blocking=True),
                                          self.slice_poses.to(self.device, non_blocking=True)).cpu()
        slice_pred = (torch.nn.Sigmoid()(slice_logits) > 0.5) & self.slice_filter
        metrics["Slice"] = display_slice([slice_pred, self.slice_labels],
                                         ["Prediction", "Truth"], self.slice_morph[0], True)

        sphere_logits = self.model.predict(self.sphere_morph.to(self.device, non_blocking=True),
                                           self.sphere_poses.to(self.device,
                                                                non_blocking=True)).cpu()
        sphere_pred = (torch.nn.Sigmoid()(sphere_logits) > 0.5) & self.sphere_filter
        metrics["Sphere"] = display_sphere([sphere_pred, self.sphere_labels],
                                           ["Prediction", "Truth"], self.sphere_radius, True)

        return metrics

    @staticmethod
    def compute_input_metrics(morph: Float[Tensor, "batch dof 3"],
                              pose: Float[Tensor, "batch 9"],
                              label: Bool[Tensor, "batch"]) -> dict:
        metrics = {"DOF": wandb.Histogram((morph != 0).any(dim=-1).sum(dim=-1).cpu().numpy() - 1)}

        centre, radius = estimate_reachable_ball(morph)
        metrics["Length (Excluding Base)"] = torch.mean(1 - centre.norm(dim=1).cpu())

        rel_radius = (pose[:, :3] - centre).norm(dim=1) / radius
        metrics["Relative Radius"] = wandb.Histogram(rel_radius.cpu().numpy())

        fig = go.Figure()
        colors = sns.color_palette("colorblind", n_colors=2).as_hex()
        for t in get_pose_traces(morph[label][:1000].cpu(), se3.from_vector(pose[label][:1000]).cpu(), colors[0],
                                 "Reachable",
                                 True):
            fig.add_trace(t)
        for t in get_pose_traces(morph[~label][:1000].cpu(), se3.from_vector(pose[~label][:1000]).cpu(), colors[1],
                                 "Unreachable", True):
            fig.add_trace(t)
        metrics["Poses"] = fig

        metrics["Reachable [%]"] = label.sum().item() / label.shape[0] * 100

        return metrics

    @staticmethod
    def assign_space(data: dict, space: str) -> dict:
        for key in list(data.keys()):
            data[f"{space}/{key}"] = data.pop(key)
        return data

    @torch.no_grad()
    def log_training(self,
                     epoch: int,
                     batch_idx: int,
                     morph: Float[Tensor, "batch dof 3"],
                     pose: Float[Tensor, "batch 9"],
                     label: Bool[Tensor, "batch"],
                     logit: Float[Tensor, "batch"],
                     loss: float):
        data = {"Loss": loss / self.batch_size, "Batch ID": self.training_set.current_batch_idx}
        data |= self.compute_metrics(logit, label)
        data = self.assign_space(data, "Training")

        if batch_idx % 10000 == 0:
            v_morph, v_pose, v_label = self.validation_set.get_semi_random_batch()
            v_morph = v_morph.to(self.device, non_blocking=True)
            v_pose = v_pose.to(self.device, non_blocking=True)
            v_label = v_label.to(self.device, non_blocking=True)

            v_logit = self.model.predict(v_morph, v_pose)
            v_loss = self.loss_function(v_logit, v_label.float())

            intermediate_val_data = {"Loss": v_loss}
            intermediate_val_data |= self.compute_metrics(v_logit, v_label)
            intermediate_val_data = self.assign_space(intermediate_val_data, "Intermediate Validation")
            data |= intermediate_val_data

            data |= self.assign_space(self.compute_boundary_metrics(), "Boundaries")

            data |= self.assign_space(self.compute_input_metrics(morph, pose, label), "TrainingsInput")
            data |= self.assign_space(self.compute_input_metrics(v_morph, v_pose, v_label), "ValidationInput")

        self.step = epoch * len(self.training_set) + batch_idx
        self.run.log(data=data, step=self.step, commit=True)

    @torch.no_grad()
    def log_validation(self,
                       epoch: int,
                       batch_idx: int,
                       morph: Float[Tensor, "batch dof 3"],
                       pose: Float[Tensor, "batch 9"],
                       label: Bool[Tensor, "batch"],
                       logit: Float[Tensor, "batch"],
                       loss: float):
        if "loss" not in self.buffer:
            self.buffer["loss"] = 0.0
        if "logit" not in self.buffer:
            self.buffer["logit"] = []
        if "label" not in self.buffer:
            self.buffer["label"] = []

        self.buffer["label"] += [label.cpu()]
        self.buffer["logit"] += [logit.cpu()]
        self.buffer["loss"] += loss

        if batch_idx + 1 == len(self.validation_set):
            label = torch.cat(self.buffer["label"])
            data = {"Loss": self.buffer["loss"] / len(self.validation_set) / self.batch_size}
            data |= self.compute_metrics(torch.cat(self.buffer["logit"]), label)
            data = self.assign_space(data, "Validation")

            self.run.log(data=data, step=self.step + 1, commit=False)
            self.buffer = {}


def binary_confusion_matrix(logit: Float[Tensor, "batch"], label: Bool[Tensor, "batch"]) \
        -> Float[Tensor, "2 2"]:
    predicted_label = torch.nn.Sigmoid()(logit) > 0.5
    confusion_matrix = torch.zeros(2, 2)
    confusion_matrix[0, 0] = (predicted_label & label).sum().item() / label.sum() * 100  # TP
    confusion_matrix[0, 1] = (~predicted_label & label).sum().item() / label.sum() * 100  # FN
    confusion_matrix[1, 0] = (predicted_label & ~label).sum().item() / (~label).sum() * 100  # FP
    confusion_matrix[1, 1] = (~predicted_label & ~label).sum().item() / (~label).sum() * 100  # TN
    return confusion_matrix
