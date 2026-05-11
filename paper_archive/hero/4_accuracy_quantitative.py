import seaborn as sns
import matplotlib.pyplot as plt

# --- Matching TikZ Colors ---
palette = sns.color_palette("colorblind", 4)
color_true = palette[2]
color_false = palette[3]

# Data
data = [
    {'label_left': 'Reachable poses', 'val_left': 97.10, 'label_right': 'False Negatives', 'val_right': 2.90, 'color_l': color_true, 'color_r': color_false},
    {'label_left': 'Unreachable poses', 'val_left': 74.98, 'label_right': 'False Positives', 'val_right': 25.02, 'color_l': color_true, 'color_r': color_false}
]

# --- Unified Styling ---
sns.set_style("white")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 24, # Slightly reduced to fit side-by-side
    "axes.unicode_minus": False,
})

# Create 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(17, 3))

for i, entry in enumerate(data):
    ax = axes[i]
    y = 0 # Each subplot only needs one bar at the same height

    # Left part of the bar
    ax.barh(y, entry['val_left'], color=entry['color_l'], height=0.8, edgecolor='none')
    # Right part (stacked)
    ax.barh(y, entry['val_right'], left=entry['val_left'], color=entry['color_r'], height=0.8, edgecolor='none')

    # Text inside left bar (Centered label)
    ax.text(
        50, y,
        f"{entry['label_left']}",
        va='center', ha='center', color='white', fontweight='bold', fontsize=40
    )

    # Percentage on the far left
    ax.text(
        0.5, y,
        f"{entry['val_left']:.1f}%",
        va='center', ha='left', color='white', rotation=90, fontweight='bold', fontsize=31
    )

    # Percentage on the far right (Fixed r_color to entry['color_r'])
    ax.text(
        100, y,
        f"{entry['val_right']:.1f}%",
        va='center', ha='right', color="white", rotation=90, fontweight='bold', fontsize=31
    )

    ax.vlines(x=entry["val_left"], ymin=-0.5, ymax=0.4, colors="#FAF0F0")
    ax.text(
        entry["val_left"]+1, -0.55,
        "FP" if i == 0 else "FN",
        va='center', ha='left', color=color_false, fontweight='bold', fontsize=34
    )
    ax.text(
        entry["val_left"]-1, -0.55,
        "TP" if i == 0 else "TN",
        va='center', ha='right', color=color_true, fontweight='bold', fontsize=34
    )

    # Cleanup individual axis
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)

plt.tight_layout()
fig.patch.set_alpha(0.0)
plt.savefig("metrics.png", transparent=True, bbox_inches='tight', dpi=300)
plt.show()