"""
viz.py
------
Visualisation utilities for the binary voter swarm:
  - LiveVisualizer  — real-time window (Voronoi + comms graph + agents)
  - plot_terrain_2d — static top-down snapshot
  - plot_metrics    — 4-panel time-series
  - plot_opinion_heatmap
"""

import os
from typing import List, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt, gaussian_filter

from agent import Agent
from terrain import Terrain
from metrics import MetricsRecorder


# Opinion colours — keep consistent across all plots.
OPINION_COLORS = {0: "#1f77b4", 1: "#d62728"}   # blue, red
DEAD_COLOR = "#7f7f7f"                          # grey


def plot_terrain_2d(terrain: Terrain, agents: List[Agent],
                    title: str = "Swarm on Terrain",
                    save_path: Optional[str] = None):
    """
    Top-down 2D view: obstacle grid as a grey image with agents overlaid.
    Opinion 0 = blue, opinion 1 = red, dead = grey X.
    This is the primary visualisation for the 2D obstacle terrain.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Obstacle grid: True cells shown dark, clear cells white.
    # imshow uses (row, col) = (y, x) so no transpose needed.
    ax.imshow(
        terrain.grid, cmap="Greys", origin="lower",
        extent=[0,
                terrain.cfg.width  * terrain.cfg.cell_size,
                0,
                terrain.cfg.depth  * terrain.cfg.cell_size],
        alpha=0.6,
    )

    alive_0 = [a for a in agents if a.alive and a.opinion == 0 and a.mobile]
    alive_1 = [a for a in agents if a.alive and a.opinion == 1 and a.mobile]
    ghosts  = [a for a in agents if a.alive and not a.mobile]
    dead    = [a for a in agents if not a.alive]

    def _sc(group, color, marker, label, size=30):
        if not group:
            return
        ax.scatter([a.position[0] for a in group],
                   [a.position[1] for a in group],
                   c=color, s=size, marker=marker,
                   edgecolors="black", linewidths=0.4,
                   label=label, zorder=3)

    _sc(alive_0, OPINION_COLORS[0], "o", f"Op-0 mobile ({len(alive_0)})")
    _sc(alive_1, OPINION_COLORS[1], "o", f"Op-1 mobile ({len(alive_1)})")
    _sc(ghosts, "orange", "s", f"Ghost ({len(ghosts)})", size=40)
    _sc(dead,   DEAD_COLOR, "x", f"Dead ({len(dead)})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved 2D terrain plot → {save_path}")
    plt.show()


def plot_terrain_with_agents(terrain: Terrain, agents: List[Agent],
                             title: str = "Swarm on Terrain",
                             save_path: Optional[str] = None):
    """
    3D surface of the terrain with agents coloured by opinion.
    Dead agents are shown in grey with an 'x' marker.
    """
    cfg = terrain.cfg
    xs = np.arange(0, cfg.width * cfg.cell_size, cfg.cell_size)
    ys = np.arange(0, cfg.depth * cfg.cell_size, cfg.cell_size)
    X, Y = np.meshgrid(xs, ys)
    Z = terrain.heightmap

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.6, edgecolor="none")

    # Bucket agents by (alive, opinion) so we can scatter each group with
    # its own colour and label.
    alive_0 = [a for a in agents if a.alive and a.opinion == 0]
    alive_1 = [a for a in agents if a.alive and a.opinion == 1]
    dead = [a for a in agents if not a.alive]

    def _scatter(group, color, marker, label):
        if not group:
            return
        ax.scatter(
            [a.position[0] for a in group],
            [a.position[1] for a in group],
            [a.position[2] + 1.0 for a in group],   # +1m so markers float visibly
            c=color, s=40, marker=marker,
            edgecolors="black", linewidths=0.5,
            label=label, depthshade=True,
        )

    _scatter(alive_0, OPINION_COLORS[0], "o", f"Opinion 0 ({len(alive_0)})")
    _scatter(alive_1, OPINION_COLORS[1], "o", f"Opinion 1 ({len(alive_1)})")
    _scatter(dead, DEAD_COLOR, "x", f"Failed ({len(dead)})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved terrain plot to {save_path}")
    plt.show()


def plot_metrics(metrics: MetricsRecorder,
                 title_prefix: str = "",
                 save_path: Optional[str] = None):
    """
    4-panel time-series for the binary voter swarm.
    Collapse timestep is marked with a vertical red line if detected.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    t = np.array(metrics.time)
    collapse_t = metrics.collapse_timestep

    def _mark_collapse(ax):
        if collapse_t is not None:
            ax.axvline(collapse_t, color="red", linestyle="--", alpha=0.7,
                       label=f"Collapse @ {collapse_t}")

    # --- Fraction holding opinion 1 (the readable signal for binary) ---
    ax = axes[0, 0]
    ax.plot(t, metrics.fraction_one, color="steelblue", linewidth=1)
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.4)  # split reference
    ax.set_ylabel("Fraction with opinion 1")
    ax.set_title(f"{title_prefix} Opinion Balance")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _mark_collapse(ax)
    if collapse_t is not None:
        ax.legend(fontsize=8)

    # --- Order parameter |2p - 1| in [0, 1] ---
    ax = axes[0, 1]
    ax.plot(t, metrics.order_param, color="darkorange", linewidth=1)
    ax.set_ylabel("Order Parameter |2p - 1|")
    ax.set_title(f"{title_prefix} Consensus Strength")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _mark_collapse(ax)

    # --- Bernoulli variance p(1-p) in [0, 0.25] ---
    ax = axes[1, 0]
    ax.plot(t, metrics.variance, color="seagreen", linewidth=1)
    ax.set_ylabel("Bernoulli variance p(1-p)")
    ax.set_xlabel("Timestep")
    ax.set_title(f"{title_prefix} Opinion Spread")
    ax.set_ylim(-0.005, 0.26)
    ax.grid(True, alpha=0.3)
    _mark_collapse(ax)

    # --- Alive count over time ---
    ax = axes[1, 1]
    ax.plot(t, metrics.alive_count, color="purple", linewidth=1)
    ax.set_ylabel("Alive agents")
    ax.set_xlabel("Timestep")
    ax.set_title(f"{title_prefix} Agent Survival")
    ax.grid(True, alpha=0.3)
    _mark_collapse(ax)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved metrics plot to {save_path}")
    plt.show()


def plot_opinion_heatmap(opinion_history: list,
                         save_path: Optional[str] = None):
    """
    Per-agent opinion over time as a heatmap.
    `opinion_history` is a list (per recorded timestep) of length-N int arrays.
    """
    if not opinion_history:
        print("[viz] No opinion history provided for heatmap.")
        return

    data = np.array(opinion_history)   # shape: (timesteps, num_agents)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        data.T, aspect="auto", cmap="coolwarm", origin="lower",
        vmin=0, vmax=1, interpolation="nearest",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Agent ID")
    ax.set_title("Per-agent opinion over time (0=blue, 1=red)")
    plt.colorbar(im, ax=ax, ticks=[0, 1], label="Opinion")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved heatmap to {save_path}")
    plt.show()


# ===========================================================================
# Live Visualizer — real-time window updated every N simulation steps
# ===========================================================================

class LiveVisualizer:
    """
    Opens a live matplotlib window that redraws every `update_interval` steps.

    Layout: two side-by-side axes.
      Left  — terrain map: obstacles, Voronoi coverage cells, comms edges,
               agent dots coloured by opinion, ghost markers.
      Right — live time-series: order parameter and alive count.

    Usage:
        vis = LiveVisualizer(terrain, total_steps, scenario_label)
        # inside sim loop:
        vis.update(agents, neighbours, t, metrics)
        # after loop:
        vis.freeze()   # keeps window open, blocks until closed
    """

    UPDATE_INTERVAL = 20   # redraw every N simulation steps

    def __init__(self, terrain: Terrain, total_steps: int,
                 scenario_label: str = ""):
        self.terrain = terrain
        self.total_steps = total_steps
        self.label = scenario_label

        # World bounds
        cfg = terrain.cfg
        self.wx = cfg.width  * cfg.cell_size
        self.wy = cfg.depth  * cfg.cell_size

        # Precompute topo elevation: distance from obstacles, inverted + smoothed
        dist = distance_transform_edt(~terrain.grid.astype(bool))
        elev = 1.0 - dist / (dist.max() + 1e-9)
        elev = gaussian_filter(elev, sigma=4.0)
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-9)
        self._terrain_elev = elev

        # Dirty-yellow geography colormap: dark open terrain → tan → dirty yellow → brown peak
        self._topo_cmap = LinearSegmentedColormap.from_list("topo", [
            "#0f0f1a",   # open / valley
            "#2b2010",   # dark earth
            "#5c4018",   # brown
            "#9a7828",   # muddy tan
            "#c8a03a",   # dirty yellow (mid-slope)
            "#e0bc50",   # light yellow (near peak)
            "#6e4c18",   # dark brown cap (obstacle cell)
        ], N=256)

        plt.ion()
        self.fig, (self.ax_map, self.ax_ts) = plt.subplots(
            1, 2, figsize=(16, 8),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        self.fig.patch.set_facecolor("#1a1a2e")   # dark background

        # --- Static terrain background (drawn once) ---
        self.ax_map.imshow(
            self._terrain_elev, cmap=self._topo_cmap, origin="lower",
            extent=[0, self.wx, 0, self.wy],
            vmin=0, vmax=1, zorder=0,
        )
        self.ax_map.set_facecolor("#0f0f1a")
        self.ax_map.set_xlim(0, self.wx)
        self.ax_map.set_ylim(0, self.wy)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_xlabel("X (m)", color="white")
        self.ax_map.set_ylabel("Y (m)", color="white")
        self.ax_map.tick_params(colors="white")

        # --- Time-series axis ---
        self.ax_ts.set_facecolor("#0f0f1a")
        self.ax_ts.tick_params(colors="white")
        self.ax_ts.set_xlabel("Timestep", color="white")
        self.ax_ts.set_xlim(0, total_steps)
        self.ax_ts.set_ylim(0, 1.05)

        # History buffers for the time-series lines
        self._ts_t: List[int]   = []
        self._ts_order: List[float] = []
        self._ts_alive: List[float] = []   # normalised to [0,1]
        self._n_agents: int = 0            # filled on first update

        # Dynamic artist handles — removed and re-added each frame
        self._dynamic: List = []

        plt.tight_layout(pad=2.0)
        self.fig.canvas.draw()
        plt.pause(0.001)

    # ------------------------------------------------------------------

    def update(self, agents: List[Agent],
               neighbours: Dict[int, List[int]],
               t: int,
               metrics: MetricsRecorder):
        """Call this inside the simulation loop."""
        if t % self.UPDATE_INTERVAL != 0:
            return

        if self._n_agents == 0:
            self._n_agents = len(agents)

        dead    = [a for a in agents if not a.alive]
        alive   = [a for a in agents if a.alive]
        mobile  = [a for a in agents if a.alive and a.mobile]
        ghosts  = [a for a in agents if a.alive and not a.mobile]
        op0     = [a for a in mobile if a.opinion == 0]
        op1     = [a for a in mobile if a.opinion == 1]

        # Clear and redraw the map axis each frame
        self.ax_map.cla()
        self.ax_map.set_facecolor("#0f0f1a")
        self.ax_map.set_xlim(0, self.wx)
        self.ax_map.set_ylim(0, self.wy)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_xlabel("X (m)", color="white")
        self.ax_map.set_ylabel("Y (m)", color="white")
        self.ax_map.tick_params(colors="white")

        # Terrain background
        self.ax_map.imshow(
            self._terrain_elev, cmap=self._topo_cmap, origin="lower",
            extent=[0, self.wx, 0, self.wy],
            vmin=0, vmax=1, zorder=0,
        )

        # --- Voronoi coverage cells (only when >= 4 mobile agents) ---
        if len(mobile) >= 4:
            pts = np.array([[a.position[0], a.position[1]] for a in mobile])
            far = max(self.wx, self.wy) * 4
            mirror = np.array([[-far, -far], [2*far, -far],
                                [-far, 2*far], [2*far, 2*far]])
            try:
                vor = Voronoi(np.vstack([pts, mirror]))
                for ridge in vor.ridge_vertices:
                    if -1 in ridge:
                        continue
                    p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
                    self.ax_map.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                     color="white", lw=0.4, alpha=0.25, zorder=1)
            except Exception:
                pass   # skip Voronoi if degenerate (collinear points)

        # --- Comms graph edges ---
        pos_map = {a.id: a.position for a in alive}
        segs = []
        for aid, nbrs in neighbours.items():
            if aid not in pos_map:
                continue
            for nid in nbrs:
                if nid > aid and nid in pos_map:
                    segs.append([(pos_map[aid][0], pos_map[aid][1]),
                                 (pos_map[nid][0], pos_map[nid][1])])
        if segs:
            self.ax_map.add_collection(
                LineCollection(segs, colors="#aaaaaa",
                               linewidths=0.4, alpha=0.35, zorder=2)
            )

        # --- Agent comm-range circles (faint) ---
        for a in mobile:
            self.ax_map.add_patch(mpatches.Circle(
                (a.position[0], a.position[1]), radius=25.0,
                color=OPINION_COLORS[a.opinion],
                fill=True, alpha=0.04, zorder=2,
            ))

        # --- Agent dots ---
        def _sc(group, color, marker, size, zorder):
            if not group:
                return
            self.ax_map.scatter(
                [a.position[0] for a in group],
                [a.position[1] for a in group],
                c=color, s=size, marker=marker,
                edgecolors="white", linewidths=0.5,
                zorder=zorder,
            )

        _sc(dead,   "#111111",          "o", 25, 3)
        _sc(op0,    OPINION_COLORS[0], "o", 50, 4)
        _sc(op1,    OPINION_COLORS[1], "o", 50, 4)
        _sc(ghosts, "orange",          "s", 70, 5)

        order = metrics.order_param[-1] if metrics.order_param else 0.0
        p1    = metrics.fraction_one[-1] if metrics.fraction_one else 0.0
        n_mob = len(mobile)
        n_gho = len(ghosts)

        self.ax_map.set_title(
            f"{self.label}   t={t}/{self.total_steps}   "
            f"mobile={n_mob}  ghost={n_gho}   "
            f"order={order:.3f}  p1={p1:.2f}",
            color="white", fontsize=10,
        )
        self.ax_map.legend(
            handles=[
                mpatches.Patch(color=OPINION_COLORS[0], label=f"Op-0 ({len(op0)})"),
                mpatches.Patch(color=OPINION_COLORS[1], label=f"Op-1 ({len(op1)})"),
                mpatches.Patch(color="orange",           label=f"Ghost ({n_gho})"),
                mpatches.Patch(color="#111111",          label=f"Dead ({len(dead)})"),
            ],
            loc="upper right", fontsize=8,
            facecolor="#1a1a2e", labelcolor="white", framealpha=0.8,
        )

        # --- Time-series update ---
        self._ts_t.append(t)
        self._ts_order.append(order)
        self._ts_alive.append(len(alive) / max(self._n_agents, 1))

        self.ax_ts.clear()
        self.ax_ts.set_facecolor("#0f0f1a")
        self.ax_ts.set_xlim(0, self.total_steps)
        self.ax_ts.set_ylim(0, 1.05)
        self.ax_ts.set_xlabel("Timestep", color="white")
        self.ax_ts.tick_params(colors="white")
        self.ax_ts.plot(self._ts_t, self._ts_order,
                        color="darkorange", lw=1.5, label="Order param")
        self.ax_ts.plot(self._ts_t, self._ts_alive,
                        color="#7b68ee", lw=1.5, label="Alive (norm)")
        self.ax_ts.axhline(0.5, color="white", lw=0.5, ls=":", alpha=0.4)
        self.ax_ts.legend(fontsize=8, facecolor="#1a1a2e",
                          labelcolor="white", framealpha=0.8)
        self.ax_ts.set_title(f"{self.label} — live metrics",
                             color="white", fontsize=9)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    # ------------------------------------------------------------------

    def freeze(self):
        """Keep the window open after the sim ends until the user closes it."""
        plt.ioff()
        plt.show(block=True)
