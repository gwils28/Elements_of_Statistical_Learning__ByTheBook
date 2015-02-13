"""Style graphique commun à tous les notebooks.

Usage typique en tête de notebook ::

    from esl.plotting import set_style, COLORS
    set_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#: Palette catégorielle (ordre fixe : on ne la fait jamais « tourner »).
COLORS = [
    "#2a78d6",  # bleu
    "#eb6834",  # orange
    "#1baf7a",  # turquoise
    "#eda100",  # jaune
    "#e87ba4",  # magenta
    "#008300",  # vert
    "#6250d6",  # violet
    "#e34948",  # rouge
]
BLUE, ORANGE = COLORS[0], COLORS[1]
GRAY = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"

#: Carte séquentielle (une seule teinte, du clair au foncé).
SEQ = LinearSegmentedColormap.from_list(
    "esl_seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
#: Carte divergente (bleu <-> rouge, milieu gris neutre).
DIV = LinearSegmentedColormap.from_list(
    "esl_div", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1d"])
#: Deux classes (bleu / orange), comme dans les figures du livre.
TWO_CLASS = ListedColormap([BLUE, ORANGE])


def set_style():
    """Applique le style ESL aux figures matplotlib."""
    mpl.rcParams.update({
        "figure.figsize": (7, 4.5),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "image.cmap": "esl_seq",
    })
    for cmap in (SEQ, DIV):
        try:
            mpl.colormaps.register(cmap)
        except ValueError:  # déjà enregistrée
            pass
