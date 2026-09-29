"""Figure style shared by the figure notebooks: fonts and font embedding, mm units, method colours
and labels, the figure order of the datasets, and helpers for previews, saving and heatmaps.
A panel adds only its own sizes and positions to panel_style() or use_style()."""

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib import colors

from .paths import DATASET_ORDER, paper_name

MM = 1.0 / 25.4  # inches per mm
DISPLAY_DPI = 110  # inline previews only; the PDFs are vector
TEXT_COLOR = "#202124"
SANS = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]  # font fallback list of most panels

# The 12 screens in figure order, by paper name; the first seven are non-pluripotent.
PAPER_ORDER = [paper_name(dataset) for dataset in DATASET_ORDER]
N_NON_PLURIPOTENT = 7

# Legend label of each method key of the DES tables and the prediction metrics.
METHOD_LABELS = {
    "rejfreq": "DE frequency",
    "presage": "PRESAGE",
    "weighted": "Weighted",
    "linear_otherds": "Linear (cross-dataset)",
    "linear_embedding_from_otherds_10": "Linear (cross-dataset)",
    "linear_train": "Linear (training)",
    "paper_linear_embedding_from_training_10": "Linear (training)",
    "gears": "GEARS",
    "scGPT-ft": "scGPT-ft",
    "random": "Random",
    "train_mean": "Train mean",
}
# Colour of each method, by label (Fig. 2b, S3c, S5b; Fig. 1d and S2 draw Weighted darker).
METHOD_COLORS = {
    "DE frequency": "#000000",
    "PRESAGE": "#66B2A3",
    "Weighted": "#6F98C9",
    "Linear (cross-dataset)": "#A6CBE2",
    "Linear (training)": "#C69BB7",
    "GEARS": "#E3A06F",
    "scGPT-ft": "#DEC36D",
}


def method_palette(keys):
    """(key, legend label, colour) of each method key, in the given order."""
    return [(key, METHOD_LABELS[key], METHOD_COLORS[METHOD_LABELS[key]]) for key in keys]


def panel_style(rc=None, fonts=SANS, open_axes=False):
    """rcParams of a panel: sans-serif text kept editable in PDF and SVG, then the panel's own rc.
    open_axes hides the top and right spines and the legend frame."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    if open_axes:
        style.update({"axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    style.update(rc or {})
    return style


def use_style(rc=None, fonts=SANS, open_axes=False):
    """Reset matplotlib to its defaults, as each paper script started in its own process,
    then apply panel_style()."""
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(panel_style(rc, fonts, open_axes))


def save_panel(figure, path, bbox_inches=None, **savefig_kwargs):
    """Show a PNG preview in the notebook, save the figure to path on a white background and close it."""
    preview = io.BytesIO()
    figure.savefig(preview, format="png", dpi=DISPLAY_DPI, bbox_inches=bbox_inches, facecolor="white")
    display(Image(data=preview.getvalue()))
    figure.savefig(path, bbox_inches=bbox_inches, facecolor="white", **savefig_kwargs)
    plt.close(figure)


def masked_heatmap(axis, values, cmap, norm, masked_color):
    """A square matrix with its diagonal masked; the masked cells show masked_color."""
    cmap.set_bad(masked_color)
    axis.set_facecolor(masked_color)
    masked = np.ma.array(values, mask=np.eye(len(values), dtype=bool))
    return axis.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal", origin="upper")


def heatmap_labels(axis, labels, rotation, fontsize, pad, **x_label):
    """Row and column names of a square heatmap, the column names rotated; no ticks, grid or spines."""
    positions = np.arange(len(labels))
    axis.set_xticks(positions, labels)
    axis.set_yticks(positions, labels)
    text = {"rotation_mode": "anchor", "ha": "right", "fontsize": fontsize}
    plt.setp(axis.get_xticklabels(), rotation=rotation, **text, **x_label)
    plt.setp(axis.get_yticklabels(), rotation=0, **text)
    axis.tick_params(axis="both", which="both", length=0, pad=pad)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.grid(False)
    axis.spines[:].set_visible(False)


def pluripotent_split(axis, columns=True, **line):
    """Line between the non-pluripotent and the pluripotent rows (and columns) of a dataset axis."""
    if columns:
        axis.axvline(N_NON_PLURIPOTENT - 0.5, **line)
    axis.axhline(N_NON_PLURIPOTENT - 0.5, **line)


def diverging_heatmap(axis, values, labels):
    """Fig. S1c, S1d: a dataset-by-dataset matrix from -1 (blue) to 1 (red), diagonal masked."""
    cmap = colors.LinearSegmentedColormap.from_list("diverging", ("#3F72B5", "#FBFAF7", "#B51F2E"), N=256)
    norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = masked_heatmap(axis, values, cmap, norm, "#F2F3F4")
    heatmap_labels(axis, labels, rotation=90, fontsize=5.2, pad=2.2, va="center")
    pluripotent_split(axis, color="#737A80", linewidth=0.75, alpha=0.78, zorder=5)
    return image
