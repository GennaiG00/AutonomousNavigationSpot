"""
Global sampler for Spot autonomous mission.

Generates a global point cloud by sampling the entire explored region
with a configurable density, instead of re-sampling at each step.
This allows for smoother navigation without recalculating at each iteration.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class GlobalSampler:
    """
    Generates and maintains a global set of sampled waypoints across the entire
    explored environment with configurable point density.
    """

    def __init__(self, env, point_density: float = 2):
        """
        Initialize the global sampler.

        Args:
            env: EnvironmentMap instance
            point_density: Number of points per square meter (default 0.2 = 1 point per 5m²)
        """
        self.env = env
        self.point_density = point_density
        self.global_points = []  # List of (x, y) tuples
        self.point_cell_map = {}  # Maps point index to cell (row, col)
        self.sampled = False

    def sample_global_grid(self) -> List[Tuple[float, float]]:
        """
        Sample the entire accessible grid once.

        Iterates through all cells and generates points based on density.
        Only samples cells that are accessible (not marked as obstacles).

        Returns:
            List of (x, y) sampled points in world coordinates
        """
        print("[SAMPLER] Starting global grid sampling...")

        points = []
        point_idx = 0

        # Calculate points per cell based on density and cell size
        cell_area = self.env.cell_size ** 2
        points_per_cell = max(1, int(cell_area * self.point_density))

        print(f"[SAMPLER] Cell area: {cell_area:.2f}m², points per cell: {points_per_cell}")

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                # Get cell center in world coordinates
                world_pos = self.env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue

                cell_x, cell_y = world_pos
                half_size = self.env.cell_size / 2.0

                # Generate random points within this cell
                for _ in range(points_per_cell):
                    # Random offset from center in grid frame
                    #FIXME: I delete the offset to make the points more uniformly distributed in the global grid and cells
                    offset_x = np.random.uniform(-half_size, half_size)
                    offset_y = np.random.uniform(-half_size, half_size)

                    # Rotate offset to world frame
                    cos_yaw = np.cos(self.env.origin_yaw)
                    sin_yaw = np.sin(self.env.origin_yaw)

                    world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
                    world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

                    # Final world position
                    sample_x = cell_x + world_offset_x
                    sample_y = cell_y + world_offset_y

                    points.append((sample_x, sample_y))
                    self.point_cell_map[point_idx] = (row, col)
                    point_idx += 1

        self.global_points = points
        self.sampled = True

        print(f"[SAMPLER] Generated {len(points)} global waypoints")
        return points

    def get_points_in_cell(self, row: int, col: int) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Get all sampled points that belong to a specific cell.

        Args:
            row, col: Cell coordinates

        Returns:
            List of (point_idx, (x, y)) tuples for points in the cell
        """
        points_in_cell = []
        for idx, (px, py) in enumerate(self.global_points):
            cell_row, cell_col = self.point_cell_map[idx]
            if cell_row == row and cell_col == col:
                points_in_cell.append((idx, (px, py)))
        return points_in_cell

    def get_nearest_points(self, x: float, y: float, k: int = 5) -> List[Tuple[int, float, Tuple[float, float]]]:
        """
        Get k nearest points to a given position.

        Args:
            x, y: Query position in world coordinates
            k: Number of nearest neighbors to return

        Returns:
            List of (point_idx, distance, (x, y)) tuples, sorted by distance
        """
        if not self.sampled or len(self.global_points) == 0:
            return []

        distances = []
        for idx, (px, py) in enumerate(self.global_points):
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            distances.append((idx, dist, (px, py)))

        # Sort by distance and return first k
        distances.sort(key=lambda x: x[1])
        return distances[:min(k, len(distances))]

    def get_all_points(self) -> List[Tuple[float, float]]:
        """Get all sampled points."""
        return self.global_points

    def update_density(self, point_density: float):
        """Update the point density and reset sampling."""
        self.point_density = point_density
        self.global_points = []
        self.point_cell_map = {}
        self.sampled = False

