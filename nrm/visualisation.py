import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor
from jaxtyping import jaxtyped, Float, Bool
from beartype import beartype
from scipy.spatial.transform import Rotation
import seaborn as sns

from nrm.dataset.reachability_manifold import estimate_reachable_ball
from nrm.dataset.self_collision import get_capsules, LINK_RADIUS
from nrm.dataset.kinematics import forward_kinematics, transformation_matrix

def to_ploty_color(color):
    return f"rgb{tuple((255*np.array(color)).astype(int).tolist())}"


@jaxtyped(typechecker=beartype)
def get_cylinder_mesh(start: Float[Tensor, "3"], end: Float[Tensor, "3"], radius: float, resolution: int = 12):
    v = end - start
    height = torch.norm(v)
    v_unit = v / (height + 1e-6)
    theta = torch.linspace(0, 2 * torch.pi, resolution, dtype=torch.float)
    z_coords = torch.tensor([0, height])
    theta_grid, z_grid = torch.meshgrid(theta, z_coords, indexing='ij')
    x_grid = radius * torch.cos(theta_grid)
    y_grid = radius * torch.sin(theta_grid)
    points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    rot, _ = Rotation.align_vectors(v_unit.unsqueeze(0), z_axis.unsqueeze(0))
    points = rot.apply(points)
    points += start
    return (points[:, 0].view(resolution, 2),
            points[:, 1].view(resolution, 2),
            points[:, 2].view(resolution, 2))


@jaxtyped(typechecker=beartype)
def get_sphere_mesh(center: Float[Tensor, "3"], radius: float, resolution: int = 12):
    u = torch.linspace(0, 2 * torch.pi, resolution)
    v = torch.linspace(0, torch.pi, resolution)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    x = center[0] + radius * torch.cos(u_grid) * torch.sin(v_grid)
    y = center[1] + radius * torch.sin(u_grid) * torch.sin(v_grid)
    z = center[2] + radius * torch.cos(v_grid)
    return x, y, z


@jaxtyped(typechecker=beartype)
def get_robot_traces(mdh, color, show_legend: bool = False, poses=None):
    """
    Generates robot meshes.
    show_legend: If True, adds one entry to the legend. If False, hides all from legend.
    """
    if poses is None:
        joints = torch.zeros((*mdh.shape[:-1], 1), device=mdh.device, dtype=mdh.dtype)
        poses = forward_kinematics(mdh, joints)

    s_all, e_all = get_capsules(mdh, poses)
    traces = []

    # We only want one single legend entry for the whole robot to avoid clutter
    legend_added = False

    for i in range(len(s_all)):
        if torch.norm(s_all[i] - e_all[i]) < 1e-6:
            continue

        cx, cy, cz = get_cylinder_mesh(s_all[i].cpu(), e_all[i].cpu(), radius=LINK_RADIUS, resolution=15)

        current_show_legend = show_legend and not legend_added
        if current_show_legend:
            legend_added = True

        traces.append(go.Surface(
            x=cx, y=cy, z=cz,
            showscale=False,
            opacity=0.5,
            surfacecolor=torch.zeros_like(cx),
            colorscale=[[0, color], [1, color]],
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
            name="Robot",  # Unified name
            legendgroup="Robot_Group",  # Unified group
            showlegend=current_show_legend,
            hoverinfo='skip'
        ))

    joints = torch.cat([s_all, e_all[-1:]]).cpu()
    colors = ["black", "red", color]
    latest_color = None
    latest_pos = None
    for i, j_pos in enumerate(joints):
        sx, sy, sz = get_sphere_mesh(j_pos, radius=LINK_RADIUS, resolution=15)
        if i == 0:
            c = 0
        elif i % 2 == 0 and i != len(joints) - 1:
            c = 1
        else:
            c = 2
        if latest_pos is not None and torch.allclose(latest_pos, j_pos) and c > latest_color:
            c = latest_color

        traces.append(go.Surface(
            x=sx, y=sy, z=sz,
            showscale=False,
            surfacecolor=torch.zeros_like(sx),
            colorscale=[[0, colors[c]],
                        [1, colors[c]]],
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
            legendgroup="Robot_Group",  # Link to same group
            showlegend=False,  # Spheres never need their own legend
            hoverinfo='skip'
        ))
        if c == 1 and i % 2 == 0:
            # The 3rd column of the rotation matrix is the Z-axis (rotation axis)
            z_axis = poses[i // 2 - 1, :3, 2].cpu()
            axis_start = j_pos - (z_axis * LINK_RADIUS * 2.5)
            axis_end = j_pos + (z_axis * LINK_RADIUS * 2.5)

            traces.append(go.Scatter3d(
                x=[axis_start[0], axis_end[0]],
                y=[axis_start[1], axis_end[1]],
                z=[axis_start[2], axis_end[2]],
                mode='lines',
                line=dict(color='red', width=4),
                legendgroup="Robot_Group",
                showlegend=False,
                hoverinfo='skip'
            ))

        latest_color = c
        latest_pos = j_pos
    return traces


@jaxtyped(typechecker=beartype)
def get_pose_traces(mdh, poses, color, name, show_legend: bool = False):
    """
    Generates pose L-frames.
    show_legend: Only True for the first subplot to prevent duplicate legend entries.
    """
    traces = []

    origins = poses[:, :3, 3]
    z_axes = poses[:, :3, 2]
    x_axes = poses[:, :3, 0]

    a = mdh[..., -1, 1:2]
    d = mdh[..., -1, 2:3]

    idx = -1
    while torch.any(mask := (a[..., 0] == 0) & (d[..., 0] == 0)):
        idx -= 1
        if len(mdh.shape) > 2:
            a[mask] = mdh[mask, idx, 1:2]
            d[mask] = mdh[mask, idx, 2:3]
        else:
            a = mdh[idx, 1:2]
            d = mdh[idx, 2:3]
    a = a.abs()
    d = d.abs()
    # Calculate start points
    z_ends = origins - (z_axes * 0.025 * (d / (a + d)))
    x_ends = z_ends - (x_axes * 0.025 * (a / (a + d)))

    # Build line segments: [start, origin, NaN]
    l_shapes = torch.stack([z_ends, origins, z_ends, x_ends], dim=1)
    nans = torch.full((l_shapes.shape[0], 1, 3), float('nan'))
    with_nans = torch.cat([l_shapes, nans], dim=1)

    plot_data = with_nans.reshape(-1, 3).numpy()

    # Use the name itself as the group, so toggling "Reachable" in legend toggles it everywhere
    legend_group = f"group_{name}"

    opacity = 0.2 if name == "Unreachable" else 1.0

    traces.append(go.Scatter3d(
        x=plot_data[:, 0],
        y=plot_data[:, 1],
        z=plot_data[:, 2],
        mode='lines',
        opacity=opacity,
        line=dict(color=color, width=4),
        name=name,
        legendgroup=legend_group,
        showlegend=show_legend  # Controlled by parent function
    ))

    traces.append(go.Scatter3d(
        x=origins[:, 0],
        y=origins[:, 1],
        z=origins[:, 2],
        mode='markers',
        marker=dict(size=5, color=color, opacity=opacity),
        legendgroup=legend_group,
        showlegend=False,
        hoverinfo='skip'
    ))
    return traces


# @jaxtyped(typechecker=beartype)
def visualise(
        mdh: list[Float[Tensor, "dofp1 3"]] | Float[Tensor, "dofp1 3"],
        poses: list[list[Float[Tensor, "batch 4 4"]]] | list[Float[Tensor, "batch 4 4"]] | Float[Tensor, "batch 4 4"],
        labels: list[list[Float[Tensor, "batch 4 4"]]] | list[Bool[Tensor, "batch"]] | Bool[Tensor, "batch"],
        names: list[list[str]] | list[str] = None):
    # --- Input Normalization ---
    if isinstance(mdh, Tensor): mdh = [mdh]
    if isinstance(poses, Tensor): poses = [poses]
    if isinstance(poses[0], Tensor): poses = [poses]
    if isinstance(labels, Tensor): labels = [labels]
    if isinstance(labels[0], Tensor): labels = [labels]

    if names is None:
        names = [f"Set {i}" for i in range(len(poses))]
    if isinstance(names[0], str):
        base_names = names
        names = [base_names for _ in range(len(poses))]

    # --- Layout Calculation ---
    num_plots = len(poses)
    cols = 3 if num_plots >= 3 else num_plots
    rows = math.ceil(num_plots / cols)

    unique_categories = len(names[0])
    colors = sns.color_palette("colorblind", n_colors=unique_categories + 1).as_hex()

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    # --- Track Global Legend State ---
    # We use this set to ensure each category appears in the legend exactly once,
    # regardless of which subplot it first appears in.
    added_legend_groups = set()

    for subplot_idx in range(num_plots):
        row = (subplot_idx // cols) + 1
        col = (subplot_idx % cols) + 1

        current_poses = poses[subplot_idx]
        current_labels = labels[subplot_idx]
        current_names = names[subplot_idx]

        # 1. Plot Poses
        for i, (pose_batch, label_batch, name_str) in enumerate(zip(current_poses, current_labels, current_names)):
            subset_poses = pose_batch[label_batch]

            # If this batch is empty (e.g. no False Positives in this specific plot), skip it completely.
            if len(subset_poses) == 0:
                continue

            # Check if we have already added a legend entry for this category name globally
            should_show_legend = name_str not in added_legend_groups
            if should_show_legend:
                added_legend_groups.add(name_str)

            pose_traces = get_pose_traces(
                mdh[subplot_idx],
                subset_poses,
                colors[i],
                name_str,
                show_legend=should_show_legend
            )
            for t in pose_traces:
                fig.add_trace(t, row=row, col=col)

        # 2. Plot Robot
        # Same logic for the robot: only show legend if we haven't shown it yet
        robot_name = "Robot Structure"
        should_show_robot = robot_name not in added_legend_groups
        if should_show_robot:
            added_legend_groups.add(robot_name)

        robot_traces = get_robot_traces(
            mdh[subplot_idx],
            colors[-1],
            show_legend=should_show_robot
        )
        for t in robot_traces:
            fig.add_trace(t, row=row, col=col)

    # Clean up Layout
    fig.update_layout(
        margin=dict(l=10, r=10, b=10, t=40),  # Minimize outer margins
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            itemsizing='constant',
            groupclick='togglegroup',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        paper_bgcolor='white',
        height=500 * rows if cols == 3 else 1000,  # Dynamic height
        width=1500,  # Fixed comfortable width
    )
    axis_style = dict(
        showgrid=True,
        gridcolor='lightgray',  # Subtle grid lines
        gridwidth=1,
        range=[-1, 1],
        showbackground=False,  # Hides the gray walls (cleaner look)
        zeroline=True,  # distinct line at 0
        zerolinecolor='gray',
        showticklabels=True,
        title_font=dict(size=10),
        tickfont=dict(size=8)
    )

    fig.update_scenes(
        aspectmode='cube',
        xaxis=dict(title='X', **axis_style),
        yaxis=dict(title='Y', **axis_style),
        zaxis=dict(title='Z', **axis_style),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    fig.show()


@jaxtyped(typechecker=beartype)
def visualise_predictions(mdh, poses, pred, gt):
    if isinstance(mdh, Tensor):
        true_positives = pred & gt
        true_negatives = ~pred & ~gt
        false_positives = pred & ~gt
        false_negatives = ~pred & gt

        visualise(mdh,
                  [poses, poses, poses, poses],
                  [true_positives, true_negatives, false_positives, false_negatives],
                  names=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
    else:
        visualise(mdh,
                  [[p, p, p, p] for p in poses],
                  [[p & g, (~p) & (~g), p & (~g), (~p) & g] for p, g in zip(pred, gt)],
                  names=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'])


@jaxtyped(typechecker=beartype)
def visualise_workspace(mdh, poses, labels):
    if isinstance(mdh, Tensor):
        visualise(mdh,
                  [poses, poses],
                  [labels, ~labels],
                  names=['Reachable', 'Unreachable'])
    else:
        visualise(mdh,
                  [[p, p] for p in poses],
                  [[l, ~l] for l in labels],
                  names=['Reachable', 'Unreachable'])


def display_geodesic(preds, names, return_fig: bool = False):
    fig = go.Figure()
    spacing = 1.3
    for i, (pred, name) in enumerate(zip(preds, names)):
        fig.add_trace(go.Scatter(
            x=torch.linspace(0, 1, pred.shape[0]),
            y=pred.float() + i * spacing,
            name=name,
            line=dict(width=2.5,color=to_ploty_color(sns.color_palette("colorblind", len(preds))[i])),
            mode='lines',
        ))

        # Optional: Add a subtle baseline for each lane to make it readable
        fig.add_shape(
            type="line", x0=0, y0=i * spacing, x1=1, y1=i * spacing,
            line=dict(color="rgba(0,0,0,0.1)", width=1, dash="dot")
        )

    fig.update_layout(
        xaxis_title=r"Geodesic Progress",
        yaxis_title=r"Reachability",
        font=dict(
            family="Times New Roman, TeX Gyre Termes, serif", # Match your LaTeX font
            size=24
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[item for i in range(len(preds)) for item in (i * spacing, i * spacing + 0.5, i * spacing + 1.0)],
            ticktext=[item for name in names for item in ('0', '', '1')],
            # Use this property to move R and UR away from the axis
            ticklabelstandoff=20,
            gridcolor='rgba(0,0,0,0.05)',
            fixedrange=True,
            automargin=True
        ),
        template="plotly_white",
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=22)
        )
    )
    if return_fig:
        return fig
    else:
        fig.show()


def display_slice(preds, names, morph, name:str=None):
    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsmath}",

    })

    n_plots = len(preds)
    n_rows = max(1, math.ceil(len(preds) / 2))
    n_cols = min(2, n_plots)

    steps = int(math.sqrt(preds[0].shape[0]))
    mat = transformation_matrix(morph[0, 0:1],
                                morph[0, 1:2],
                                morph[0, 2:3],
                                torch.zeros_like(morph[0, 0:1]))

    torus_axis = torch.nn.functional.normalize(mat[:3, 2], dim=0)
    centre, radius = estimate_reachable_ball(morph)
    fixed_axes = torch.argmax(torus_axis.abs())
    axes_mask = torch.ones(3, dtype=torch.bool)
    axes_mask[fixed_axes] = False

    axes_range = torch.linspace(-radius, radius, steps)
    limit = radius.item() if torch.is_tensor(radius) else radius

    # Determine axis labels based on your mask
    axis_names = ['x', 'y', 'z']
    x_label, y_label = [axis_names[i] for i, m in enumerate(axes_mask) if m]

    # 1. Setup Figure and Axes
    # 400px ~ 4 inches (assuming default 100 dpi)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(4 * n_cols, 4 * n_rows),
                            squeeze=False)
    axs = axs.flatten()

    # Create the custom colormap matching Plotly's [[0, '#ffffff'], [1, '#555555']]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_grey", ["#ffffff00", "#555555ff"])

    # Font dictionary to mimic Plotly layout updates
    font_dict = {
        'family': 'serif',
        'fontname': 'Times New Roman', # Falls back to serif if not installed
        'size': 24
    }

    for i, pred in enumerate(preds):
        ax = axs[i]
        Z = pred.reshape(steps, steps).detach().cpu().float().numpy()

        # 1. DRAW CENTER CROSSHAIR (Bottom Layer)
        ax.axhline(0,  color='black', linewidth=1, alpha=0.1, zorder=1)
        ax.axvline(0, color='black', linewidth=1, alpha=0.1, zorder=1)

        # 2. DRAW GRAPHIC (Top Layer)
        ax.imshow(Z, extent=[-limit, limit, -limit, limit],
                  origin='lower', cmap=cmap, aspect='equal', zorder=2)

        # 4. PLACE LABELS INSIDE THE CORNERS
        # Using transAxes (0 to 1) makes these independent of the data 'limit'
        text_opts = {'fontsize': 18, 'color': 'gray', 'zorder': 4}

        ax.text(0.07, limit-0.12, '$1$', va='bottom', ha='center', **text_opts)
        ax.text(limit-0.07, +0.07, '$1$', va='center', ha='left', **text_opts)
        ax.text(-0.07, -limit, '$-1$', va='bottom', ha='center', **text_opts)
        ax.text(-limit, -0.07, '$-1$', va='center', ha='left', **text_opts)

        # 5. AXIS LABELS (x, y, z) - Just outside the box
        ax.text(limit-0.07, -0.07, f'${x_label}$', va='center', ha='left', **text_opts)
        ax.text(-0.07, limit-0.12, f'${y_label}$', va='bottom', ha='center', **text_opts)

        # 6. REMOVE BORDER AND DEFAULT TICKS
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Remove any empty subplots if n_plots is odd
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    if name:
        # 1. Clear margins and UI
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # 2. Transparent backgrounds
        fig.patch.set_alpha(0.0)
        for ax in axs:
            ax.patch.set_alpha(0.0)

            # 3. Tighten the axes boundaries natively
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(name, bbox_inches="tight", pad_inches=0,dpi=600,transparent=True)
    else:
        # Prevent labels from overlapping
        plt.tight_layout()
        plt.show()


def display_sphere(preds, names, radius, return_fig: bool = False):
    steps = int(math.sqrt(preds[0].shape[0]))
    theta = torch.linspace(0, torch.pi, steps)
    phi = torch.linspace(0, 2 * torch.pi, steps)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    x = radius * torch.sin(theta_grid) * torch.cos(phi_grid)
    y = radius * torch.sin(theta_grid) * torch.sin(phi_grid)
    z = radius * torch.cos(theta_grid)

    n_plots = len(preds)
    n_rows = max(1, math.ceil(len(preds) / 2))
    n_cols = min(2, n_plots)
    dynamic_specs = [[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=names,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
        specs=dynamic_specs
    )

    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu().numpy()
    for i, res in enumerate(preds):
        row, col = (i // 2) + 1, (i % 2) + 1
        # We filter for only 'reachable' points to make the sphere shell clear
        mask = res.numpy().astype(bool)

        fig.add_trace(
            go.Scatter3d(
                x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                mode='markers',
                marker=dict(size=2, color=res[mask], colorscale='Viridis', opacity=0.8),
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=400 * max(1, len(preds) // 2),
        width=400 * min(2, len(preds))
    )

    fig.update_scenes(
        aspectmode='cube',
        xaxis=dict(title='X', showticklabels=False),
        yaxis=dict(title='Y', showticklabels=False),
        zaxis=dict(title='Z', showticklabels=False),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    if return_fig:
        return fig
    else:
        fig.show()

def visualise_trajectories(morph, trajectories:list[Tensor], labels:list[Tensor], names:list[str]):
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )
    colors = sns.color_palette("colorblind", n_colors=2*len(trajectories)).as_hex()
    i = -1
    for t, l, n in zip(trajectories, labels, names):
        for trace in get_pose_traces(morph, t[l], colors[i := i+1], f"Reachable ({n})", True):
            fig.add_trace(trace, 1, 1)
        for trace in get_pose_traces(morph, t[~l], colors[i := i+1], f"Unreachable ({n})", True):
            fig.add_trace(trace, 1, 1)


    # Clean up Layout
    fig.update_layout(
        margin=dict(l=10, r=10, b=10, t=40),  # Minimize outer margins
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            itemsizing='constant',
            groupclick='togglegroup',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        paper_bgcolor='white',
        height=1000,  # Dynamic height
        width=1500,  # Fixed comfortable width
    )
    axis_style = dict(
        showgrid=True,
        gridcolor='lightgray',  # Subtle grid lines
        gridwidth=1,
        range=[-1, 1],
        showbackground=False,  # Hides the gray walls (cleaner look)
        zeroline=True,  # distinct line at 0
        zerolinecolor='gray',
        showticklabels=True,
        title_font=dict(size=10),
        tickfont=dict(size=8)
    )

    fig.update_scenes(
        aspectmode='cube',
        xaxis=dict(title='X', **axis_style),
        yaxis=dict(title='Y', **axis_style),
        zaxis=dict(title='Z', **axis_style),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    fig.show()