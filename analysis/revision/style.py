import matplotlib as mpl
LABELS = {"baseline": "Baseline", "hard_ttl": "Hard TTL", "fixed_decay": "Fixed decay", "adaptive_freshvote": "FreshVote"}
COLORS = {"baseline": "#3d3d3a", "hard_ttl": "#eb6834", "fixed_decay": "#2a78d6", "adaptive_freshvote": "#1baf7a"}
MARKERS = {"baseline": "o", "hard_ttl": "s", "fixed_decay": "^", "adaptive_freshvote": "D"}
LINES = {"baseline": "-", "hard_ttl": "--", "fixed_decay": "-.", "adaptive_freshvote": "-"}
POLICIES = ["baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"]
def apply():
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e0",
        "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
