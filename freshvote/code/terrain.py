"""
terrain.py
----------
2D obstacle-grid terrain for the ground swarm simulation.

The terrain is a flat plane divided into a boolean grid:
  True  = obstacle (impassable — rock, debris, mud trap, etc.)
  False = clear

Obstacle density is heterogeneous: the grid is split into
zone_count x zone_count zones, each with its own randomly drawn density.
This produces the "heterogeneous obstacle densities" required by the paper.

No elevation, no PyBullet. Line-of-sight is a 2D grid ray march.
"""

import numpy as np
from config import TerrainConfig


def generate_obstacle_grid(cfg: TerrainConfig) -> np.ndarray:
    """
    Build a (depth x width) boolean grid where True = obstacle.

    The grid is divided into cfg.zone_count x cfg.zone_count rectangular
    zones. Each zone independently draws an obstacle probability uniformly
    from [min_density, max_density], then independently Bernoulli-samples
    each cell. This gives spatial heterogeneity — some corridors are nearly
    clear, others are almost impassable — matching real contested terrain.
    """
    rng = np.random.default_rng(cfg.seed)
    grid = np.zeros((cfg.depth, cfg.width), dtype=bool)

    zone_h = cfg.depth  // cfg.zone_count   # rows per zone
    zone_w = cfg.width  // cfg.zone_count   # cols per zone

    for zy in range(cfg.zone_count):
        for zx in range(cfg.zone_count):
            # Random obstacle density for this zone
            density = rng.uniform(cfg.min_density, cfg.max_density)
            row_start = zy * zone_h
            row_end   = row_start + zone_h
            col_start = zx * zone_w
            col_end   = col_start + zone_w
            # Bernoulli sample each cell in the zone
            grid[row_start:row_end, col_start:col_end] = (
                rng.random((zone_h, zone_w)) < density
            )

    return grid


class Terrain:
    def __init__(self, cfg: TerrainConfig):
        self.cfg = cfg
        self.grid = generate_obstacle_grid(cfg)   # (depth x width) bool array

    # ------------------------------------------------------------------
    # Public interface — same names as before so comms/dynamics still work
    # ------------------------------------------------------------------

    def is_obstacle(self, x: float, y: float) -> bool:
        """Return True if world position (x, y) falls on an obstacle cell."""
        row, col = self._grid_idx(x, y)
        return bool(self.grid[row, col])

    def height(self, x: float, y: float) -> float:
        """Flat 2D terrain — always returns 0.0."""
        return 0.0

    def slope(self, x: float, y: float) -> float:
        """Flat 2D terrain — always returns 0.0."""
        return 0.0

    def line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        2D grid ray march (DDA) between two world positions.
        Returns True if no obstacle cell intersects the line p1→p2.
        z-coordinates are ignored (terrain is flat).
        """
        x0, y0 = p1[0], p1[1]
        x1, y1 = p2[0], p2[1]

        c = self.cfg.cell_size
        # Starting and ending grid indices
        col0, row0 = int(x0 / c), int(y0 / c)
        col1, row1 = int(x1 / c), int(y1 / c)

        dx, dy = abs(col1 - col0), abs(row1 - row0)
        sx = 1 if col1 > col0 else -1
        sy = 1 if row1 > row0 else -1
        err = dx - dy

        col, row = col0, row0
        max_dim = self.cfg.width + self.cfg.depth  # upper bound on steps

        for _ in range(max_dim):
            # Out-of-bounds → treat as clear (not our terrain)
            if not (0 <= row < self.cfg.depth and 0 <= col < self.cfg.width):
                break
            if self.grid[row, col]:
                return False   # hit an obstacle
            if col == col1 and row == row1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                col += sx
            if e2 < dx:
                err += dx
                row += sy

        return True   # ray reached destination without hitting anything

    def world_bounds(self):
        """Return (x_min, x_max, y_min, y_max) in world coordinates."""
        return (
            0.0,
            self.cfg.width * self.cfg.cell_size,
            0.0,
            self.cfg.depth * self.cfg.cell_size,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _grid_idx(self, x: float, y: float):
        """Convert world (x, y) to clamped (row, col) grid indices."""
        col = int(np.clip(x / self.cfg.cell_size, 0, self.cfg.width - 1))
        row = int(np.clip(y / self.cfg.cell_size, 0, self.cfg.depth - 1))
        return row, col
