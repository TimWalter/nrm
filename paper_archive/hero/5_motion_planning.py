import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from nrm.dataset.kinematics import inverse_kinematics
from matplotlib.ticker import MaxNLocator

import os
from matplotlib.lines import Line2D

save_dir = Path(os.getcwd()) / "rq3_motion_planning" / "data"  /"base"
#%%
morph = pickle.load(open(save_dir / "morph.pkl", "rb"))
target_trajectory = pickle.load(open(save_dir / "target_trajectory.pkl", "rb"))
trajectory = pickle.load(open(save_dir / "trajectory.pkl", "rb"))
last_reachability = pickle.load(open(save_dir / "last_reachability.pkl", "rb"))
#%%
true_reachability = inverse_kinematics(morph.cpu(), target_trajectory.cpu())[1] != -1
#%%

trajectories = [target_trajectory.cpu(), trajectory.cpu()]
labels = [true_reachability, last_reachability]
names = ["Nominal", "Feasible"]

sns.set_style("ticks")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pgf.rcfonts": False,
    "text.latex.preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage[T1]{fontenc}
    \usepackage{helvet}
    \renewcommand{\familydefault}{\sfdefault}
    \usepackage[italic]{mathastext} % Forces math to match the Helvetica text font
""",
    "font.size": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "axes.unicode_minus": False,
})

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Color Palette ---
palette_id = sns.color_palette("colorblind", n_colors=4)
color_reach = palette_id[2]   # Clean Green
color_unreach = palette_id[3] # Clean Red

def get_robust_l_frame(poses, axis_len=0.04):
    origins = poses[:, :3, 3]
    z_axes = poses[:, :3, 2]
    x_axes = poses[:, :3, 0]

    z_ends = origins + (z_axes * axis_len)
    x_ends = origins + (x_axes * axis_len)/8

    segments = torch.stack([z_ends, origins, x_ends], dim=1)
    nans = torch.full((segments.shape[0], 1, 3), float('nan'), device=poses.device)
    line_data = torch.cat([segments, nans], dim=1).reshape(-1, 3).cpu().numpy()

    return line_data, origins.cpu().numpy()

all_origins = []

for idx, (t, l, n) in enumerate(zip(trajectories, labels, names)):
    id_color = palette_id[idx]

    # PATH (Identity)
    _, all_pts = get_robust_l_frame(t)
    all_origins.append(all_pts)

    ax.plot(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
            color=id_color, linewidth=2.5, alpha=0.7, zorder=2)

    # REACHABLE
    if torch.any(l):
        r_lines, r_pts = get_robust_l_frame(t[l])
        ax.scatter(r_pts[:, 0], r_pts[:, 1], r_pts[:, 2],
                   color=color_reach, s=80, edgecolors='white', linewidth=0.5, zorder=11)
        ax.plot(r_lines[:, 0], r_lines[:, 1], r_lines[:, 2],
                color=color_reach, linewidth=2, zorder=10)

    # UNREACHABLE
    if torch.any(~l):
        u_lines, u_pts = get_robust_l_frame(t[~l])
        ax.scatter(u_pts[:, 0], u_pts[:, 1], u_pts[:, 2],
                   color=color_unreach, s=80, edgecolors='white', linewidth=0.5, zorder=11)
        ax.plot(u_lines[:, 0], u_lines[:, 1], u_lines[:, 2],
                color=color_unreach, linewidth=1.5, linestyle=':', alpha=0.5, zorder=10)

# --- DYNAMIC TIGHT ZOOM ---
stacked_origins = np.vstack(all_origins)

# 1. Find the min and max for each axis independently
mins = stacked_origins.min(axis=0)
maxs = stacked_origins.max(axis=0)

# 2. Calculate the spread (range) for each axis
spreads = maxs - mins

# 3. Add 20% padding to each axis independently.
# If an axis has 0 spread (e.g. 2D motion), give it a minimum 5cm padding
padding = np.maximum(spreads * 0.2, 0.05) *0

x_lims = [mins[0] - padding[0], 0.05]
y_lims = [mins[1] - padding[1], -0.05]
z_lims = [mins[2] - padding[2], -0.15]

ax.set_xlim(x_lims)
ax.set_ylim(y_lims)
ax.set_zlim(z_lims)

# 4. Set the visual aspect ratio to match the physical bounds
# This prevents Matplotlib from stretching your L-frames into a cube shape!
ax.set_box_aspect((
    x_lims[1] - x_lims[0],
    y_lims[1] - y_lims[0],
    z_lims[1] - z_lims[0]
))

# --- AESTHETICS ---
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(True, color='#F0F0F0', alpha=0.8)

ax.set_xlabel(r"$X$", labelpad=0)
ax.set_ylabel(r"$Y$", labelpad=-10)
ax.set_zlabel(r"$Z$", labelpad=-10)

# --- CUSTOM COMPOSITE LEGEND ---
custom_lines = []
for idx, n in enumerate(names):
    custom_lines.append(Line2D([0], [0], color=palette_id[idx], lw=3, label=n))

custom_lines.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_reach, markersize=10, label='Reachable'))
custom_lines.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_unreach, markersize=10, label='Unreachable'))

legend = ax.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.87),
                   ncol=2, frameon=True, handletextpad=0.5, columnspacing=1.5)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 0, 0.0))

ax.view_init(elev=25, azim=80)
x_locator = MaxNLocator(nbins=4)
y_locator = MaxNLocator(nbins=3)
z_locator = MaxNLocator(nbins=2)
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)
ax.zaxis.set_major_locator(z_locator)
# 2. Force the limits to SNAP to the grid lines
# We use the current data to find the "pretty" ticks, then crop the axis there.
def snap_limits_to_grid(axis, locator, data_min, data_max):
    ticks = locator.tick_values(data_min, data_max)
    # Filter ticks to only those within a reasonable range of our data
    actual_ticks = [t for t in ticks if t >= ticks[0] and t <= ticks[-1]]
    return actual_ticks[0], actual_ticks[-1]

new_x_min, new_x_max = snap_limits_to_grid(ax.xaxis, x_locator, mins[0], 0.05)
new_y_min, new_y_max = snap_limits_to_grid(ax.yaxis, y_locator, mins[1], -0.05)
new_z_min, new_z_max = snap_limits_to_grid(ax.zaxis, z_locator, mins[2], -0.15)

ax.set_xlim(new_x_min, new_x_max)
ax.set_ylim(new_y_min, new_y_max)
ax.set_zlim(new_z_min, new_z_max)

# 3. Apply the Box Aspect after the limits are snapped
ax.set_box_aspect((new_x_max - new_x_min, new_y_max - new_y_min, new_z_max - new_z_min))
ax.xaxis.set_tick_params(which='both', length=0, colors=(0,0,0,0))
ax.yaxis.set_tick_params(which='both', length=0, colors=(0,0,0,0))
ax.zaxis.set_tick_params(which='both', length=0, colors=(0,0,0,0))
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.savefig("motion_planning.png", transparent=True, bbox_inches='tight', dpi=300)
plt.show()