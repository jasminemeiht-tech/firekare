import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# =========================================================
# 1. Load data
# =========================================================
data_path = "prediction_probability_distribution_data.csv"
df = pd.read_csv(data_path)

# =========================================================
# 2. Construct class-aligned decision margin
#    Negative sample: threshold - prob
#    Positive sample: prob - threshold
# =========================================================
df["aligned_margin"] = np.where(
    df["true_class"].str.lower() == "negative",
    df["threshold"] - df["prob"],
    df["prob"] - df["threshold"]
)

# =========================================================
# 3. Define order
# =========================================================
model_order = [
    "Linear SVM",
    "Biomarker MLP q65",
    "RBF-SVM",
    "Random Forest",
    "XGBoost",
    "MLP-full features"
]

class_order = ["Negative", "Positive"]

df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df["true_class"] = pd.Categorical(df["true_class"], categories=class_order, ordered=True)

# =========================================================
# 4. Summary statistics: Q1 / median / Q3
# =========================================================
summary = (
    df.groupby(["model", "true_class"], observed=True)["aligned_margin"]
      .agg(
          q1=lambda x: np.percentile(x, 25),
          median="median",
          q3=lambda x: np.percentile(x, 75)
      )
      .reset_index()
)

mis_df = df[df["is_misclassified"] == True].copy()

# =========================================================
# 5. Nature-like plotting parameters
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42
})

# restrained colors
neg_color = "#4C78A8"   # muted blue
pos_color = "#F28E2B"   # muted orange
err_color = "#C00000"   # strong but not overly saturated red
shade_color = "#F3DADA" # pale error-side background
grid_color = "#D9D9D9"
spine_color = "#333333"
text_color = "#222222"

panel_specs = {
    "Negative": {"title": "True negative class", "color": neg_color},
    "Positive": {"title": "True positive class", "color": pos_color}
}

# top-to-bottom row positions
y_positions = np.arange(len(model_order))[::-1]
y_map = {m: y for m, y in zip(model_order, y_positions)}

# =========================================================
# 6. Create figure
# =========================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 4.6),   # compact journal-style layout
    sharey=True,
    dpi=300,
    gridspec_kw={"wspace": 0.12}
)

xlim = (-0.75, 0.75)
xticks = np.arange(-0.75, 0.76, 0.25)

for ax, cls in zip(axes, class_order):
    spec = panel_specs[cls]
    main_color = spec["color"]

    # error-side shading
    ax.axvspan(xlim[0], 0, color=shade_color, alpha=0.45, zorder=0)

    # subtle grid
    ax.grid(axis="x", color=grid_color, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.grid(axis="y", color=grid_color, linewidth=0.7, linestyle=(0, (2, 2)))

    # decision boundary
    ax.axvline(0, color="black", linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)

    sub_sum = summary[summary["true_class"] == cls]
    sub_mis = mis_df[mis_df["true_class"] == cls]

    # IQR band + median
    for _, row in sub_sum.iterrows():
        model = row["model"]
        y = y_map[model]

        # IQR band
        ax.hlines(
            y=y,
            xmin=row["q1"],
            xmax=row["q3"],
            color=main_color,
            linewidth=7.5,
            alpha=0.28,
            zorder=2
        )

        # median
        ax.vlines(
            x=row["median"],
            ymin=y - 0.18,
            ymax=y + 0.18,
            color=main_color,
            linewidth=2.8,
            zorder=3
        )

    # misclassified samples
    for model in model_order:
        tmp = sub_mis[sub_mis["model"] == model].copy()
        if len(tmp) == 0:
            continue

        y0 = y_map[model]

        # deterministic small vertical spread
        if len(tmp) == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.14, 0.14, len(tmp))

        ax.scatter(
            tmp["aligned_margin"].values,
            y0 + offsets,
            s=62,
            facecolors="white",
            edgecolors=err_color,
            linewidths=1.5,
            zorder=4
        )

    # axis settings
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_title(spec["title"], color=main_color, pad=8, weight="bold")

    # cleaner spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", colors=text_color)

# y-axis
axes[0].set_yticks(y_positions)
axes[0].set_yticklabels(model_order, color=text_color)
axes[1].tick_params(labelleft=False)

# shared x label
fig.supxlabel("Class-aligned decision margin", y=0.09, fontsize=12, color=text_color)

# =========================================================
# 7. Minimal legend
# =========================================================
legend_handles = [
    Line2D([0], [0], color=neg_color, lw=7.5, alpha=0.28, label="IQR"),
    Line2D([0], [0], color=neg_color, marker='|', markersize=16,
           linestyle='None', markeredgewidth=2.8, label="Median"),
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='white', markeredgecolor=err_color,
           markeredgewidth=1.5, markersize=7.5,
           label="Misclassified sample"),
    Line2D([0], [0], color='black', lw=1.2, linestyle=(0, (4, 4)),
           label="Decision boundary")
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.015),
    handlelength=1.8,
    columnspacing=1.8
)

# =========================================================
# 8. Layout and save
# =========================================================
plt.tight_layout(rect=[0, 0.12, 1, 1])

plt.savefig("nature_style_aligned_margin_no_title.png", dpi=600, bbox_inches="tight")
plt.savefig("nature_style_aligned_margin_no_title.pdf", bbox_inches="tight")

plt.show()