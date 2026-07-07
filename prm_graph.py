"""
Probabilistic Road Map (PRM) for Spot autonomous mission.

Builds a graph connecting sampled waypoints with collision-free edges.
Edges are weighted with costs (currently uniform, but extensible for terrain-based weights).
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import heapq


class PRM:
    """
    Probabilistic Road Map for local path planning.

    Connects pre-sampled points (from global sampler) using collision-free edges.
    Uses Dijkstra's algorithm for pathfinding.
    """

    def __init__(self, max_edge_length: float = 2.0, connection_radius: float = 3.0):
        """
        Initialize the PRM.

        Args:
            max_edge_length: Maximum length of an edge in the graph (m)
            connection_radius: Only consider points within this radius for connection (m)
        """
        self.max_edge_length = max_edge_length
        self.connection_radius = connection_radius

        self.nodes = {}  # {point_idx: (x, y)}
        self.edges = {}  # {point_idx: [(neighbor_idx, weight), ...]}
        self.edge_validity = {}  # {(idx1, idx2): bool} - caches edge validity

        self.built = False

    def add_node(self, point_idx: int, x: float, y: float):
        """Add a node (waypoint) to the PRM."""
        self.nodes[point_idx] = (x, y)
        self.edges[point_idx] = []

    def add_nodes_from_sampler(self, global_sampler):
        """
        Populate PRM with nodes from global sampler.

        Args:
            global_sampler: GlobalSampler instance with sampled points
        """
        for idx, (x, y) in enumerate(global_sampler.get_all_points()):
            self.add_node(idx, x, y)

        print(f"[PRM] Added {len(self.nodes)} nodes from global sampler")

    # --- [NEW METHOD] Trova il nodo del grafo più vicino a coordinate (x,y) ---
    def get_nearest_node(self, x: float, y: float) -> Optional[int]:
        """Find the ID of the nearest node in the PRM to the given coordinates."""
        if not self.nodes:
            return None
        min_dist = float('inf')
        nearest_idx = None
        for idx, (nx, ny) in self.nodes.items():
            dist = np.sqrt((nx - x)**2 + (ny - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

    def build_graph(self):
        """
        Build the PRM graph by connecting nearby nodes.

        Currently connects all pairs within connection_radius.
        Assumes all edges are collision-free (validation happens during motion).
        """
        print(f"[PRM] Building graph with {len(self.nodes)} nodes...")

        node_ids = list(self.nodes.keys())

        for i, idx1 in enumerate(node_ids):
            x1, y1 = self.nodes[idx1]

            for idx2 in node_ids[i+1:]:
                x2, y2 = self.nodes[idx2]

                # Calculate distance
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Only connect if within radius and shorter than max length
                if dist <= self.connection_radius and dist <= self.max_edge_length:
                    weight = 1.0  # Uniform weight (can be modified for terrain costs)

                    # Add bidirectional edges
                    self.edges[idx1].append((idx2, weight))
                    self.edges[idx2].append((idx1, weight))

                    # Cache edge as valid (will be updated during motion)
                    self.edge_validity[(idx1, idx2)] = True
                    self.edge_validity[(idx2, idx1)] = True

        self.built = True

        total_edges = sum(len(neighbors) for neighbors in self.edges.values()) // 2
        print(f"[PRM] Built graph with {total_edges} edges")

    def mark_edge_invalid(self, idx1: int, idx2: int):
        """
        Mark an edge as blocked (collision detected).

        Args:
            idx1, idx2: Point indices
        """
        key = (min(idx1, idx2), max(idx1, idx2))
        self.edge_validity[key] = False

        # Remove from adjacency lists
        if idx2 in [n[0] for n in self.edges.get(idx1, [])]:
            self.edges[idx1] = [(n, w) for n, w in self.edges[idx1] if n != idx2]
        if idx1 in [n[0] for n in self.edges.get(idx2, [])]:
            self.edges[idx2] = [(n, w) for n, w in self.edges[idx2] if n != idx1]

    def is_edge_valid(self, idx1: int, idx2: int) -> bool:
        """Check if an edge is assumed valid (not yet marked as blocked)."""
        key = (min(idx1, idx2), max(idx1, idx2))
        return self.edge_validity.get(key, False)

    def find_path_dijkstra(self, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Args:
            start_idx: Starting node index
            goal_idx: Goal node index

        Returns:
            List of node indices representing the path, or None if no path exists
        """
        if start_idx not in self.nodes or goal_idx not in self.nodes:
            return None

        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.nodes.keys()}
        distances[start_idx] = 0
        parents = {node: None for node in self.nodes.keys()}

        pq = [(0, start_idx)]  # (distance, node)
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal_idx:
                # Reconstruct path
                path = []
                node = goal_idx
                while node is not None:
                    path.append(node)
                    node = parents[node]
                return path[::-1]

            # Check neighbors
            for neighbor, weight in self.edges.get(current, []):
                if neighbor not in visited:
                    new_dist = current_dist + weight

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parents[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))

        return None  # No path found

    def get_node_position(self, node_idx: int) -> Optional[Tuple[float, float]]:
        """Get the (x, y) position of a node."""
        return self.nodes.get(node_idx)

    def get_node_neighbors(self, node_idx: int) -> List[int]:
        """Get indices of neighboring nodes."""
        return [idx for idx, _ in self.edges.get(node_idx, [])]

    def add_temporary_node(self, x: float, y: float) -> int:
        """
        Add a temporary node (e.g., current robot position) for pathfinding.

        Returns:
            Temporary node index (negative value)
        """
        temp_idx = -(len(self.nodes) + 1)
        self.nodes[temp_idx] = (x, y)
        self.edges[temp_idx] = []

        # Connect to nearby permanent nodes
        for perm_idx, (px, py) in self.nodes.items():
            if perm_idx >= 0:  # Only permanent nodes
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                if dist <= self.connection_radius:
                    weight = 1.0
                    self.edges[temp_idx].append((perm_idx, weight))
                    self.edges[perm_idx].append((temp_idx, weight))

        return temp_idx

    def remove_temporary_node(self, temp_idx: int):
        """Remove a temporary node and its edges."""
        if temp_idx in self.nodes:
            # Remove edges from neighbors
            for neighbor in list(self.edges.get(temp_idx, [])):
                if neighbor in self.edges:
                    self.edges[neighbor] = [(n, w) for n, w in self.edges[neighbor] if n != temp_idx]

            del self.nodes[temp_idx]
            del self.edges[temp_idx]