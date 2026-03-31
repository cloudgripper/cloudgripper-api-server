from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import networkx as nx
import numpy as np
import skimage.morphology
from scipy.interpolate import splprep, splev

LOWER_KEEP_THRESHOLD = 345.0
UPPER_KEEP_THRESHOLD = 425.0

robot_rope_hsv_range = {
    "07": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "08": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "11": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "14": {
        "lower": np.array([102, 24, 65]),
        "upper": np.array([132, 129, 255]),
    },
    "15": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "17": {
        "lower": np.array([95, 29, 67]),
        "upper": np.array([146, 126, 255]),
    },
    "19": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "20": {
        "lower": np.array([95, 21, 74]),
        "upper": np.array([146, 126, 255]),
    },
    "22": {
        "lower": np.array([102, 33, 57]),
        "upper": np.array([179, 109, 255]),
    },
    "23": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "25": {
        "lower": np.array([95, 21, 40]),
        "upper": np.array([146, 126, 255]),
    },
    "27": {
        "lower": np.array([109, 21, 40]),
        "upper": np.array([130, 126, 255]),
    }
}
class RopeSegmentationUtil:
    def __init__(
        self,
        robot_id: str = "cr07",
        number_of_points: int = 20,
        morph_kernel_size: int = 5,
        min_component_area: int = 250,
        bridge_merge_threshold: float = 10.0,
        spline_smoothing_scale: float = 3.0,
    ) -> None:
        self.robot_id = robot_id
        self.number_of_points = number_of_points
        self.morph_kernel_size = morph_kernel_size
        self.min_component_area = min_component_area
        self.bridge_merge_threshold = bridge_merge_threshold
        self.spline_smoothing_scale = spline_smoothing_scale

    def segment_rope_unfiltered(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = robot_rope_hsv_range[self.robot_id]["lower"]
        upper_bound = robot_rope_hsv_range[self.robot_id]["upper"]
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

        return mask

    def apply_kernel(self, mask, morph_kernel_size=5):
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_rope_points(
        self,
        image: np.ndarray,
        number_of_points: Optional[int] = None,
        limit_rope_length: bool = True,
    ) -> List[Tuple[int, int]]:
        if image is None or image.size == 0:
            return []

        n_points = number_of_points or self.number_of_points

        raw_mask = self.segment_rope_unfiltered(image)

        largest_contour_mask = self._retain_largest_contour_safe(raw_mask)
        if largest_contour_mask is None:
            return []

        masked_raw = cv2.bitwise_and(raw_mask, largest_contour_mask)
        kernel_mask_raw = self.apply_kernel(masked_raw, morph_kernel_size=self.morph_kernel_size)
        kernel_mask = self._filter_disjoint_components_by_area(kernel_mask_raw, min_area=self.min_component_area)

        if np.count_nonzero(kernel_mask) > 0:
            kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_ellipse_3 = cv2.morphologyEx(kernel_mask, cv2.MORPH_CLOSE, kernel_ellipse_3)
        else:
            mask_ellipse_3 = np.zeros_like(kernel_mask)

        if np.count_nonzero(mask_ellipse_3) > 0:
            kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            mask_ellipse_5 = cv2.morphologyEx(mask_ellipse_3, cv2.MORPH_CLOSE, kernel_ellipse_5)
        else:
            mask_ellipse_5 = np.zeros_like(mask_ellipse_3)

        kernel_mask = mask_ellipse_5

        if np.count_nonzero(kernel_mask) == 0:
            return []

        skeleton_mask = (skimage.morphology.skeletonize(kernel_mask > 0).astype(np.uint8) * 255)

        graph = self._skeleton_to_graph(skeleton_mask)
        if graph.number_of_nodes() == 0:
            return []

        self._connect_or_remove_disjoint_components(graph)

        graph = self._clean_graph_topology(graph, bridge_merge_threshold=self.bridge_merge_threshold)

        rope_path = self._trace_longest_endpoint_path_no_overlap(graph)
        if not rope_path:
            return []

        spline_points = self._fit_spline(rope_path, n_points)

        rope_points = spline_points if spline_points else self._sample_path_points(rope_path, n_points)
        if not rope_points:
            return []

        length_px = self._polyline_length(rope_points)
        if not np.isfinite(length_px):
            return []
        if limit_rope_length and (length_px < LOWER_KEEP_THRESHOLD or length_px > UPPER_KEEP_THRESHOLD):
            return []

        # Ensure points are ordered so the point with highest px is at the end
        if rope_points:
            first_point = rope_points[0]
            last_point = rope_points[-1]
            if first_point[0] > last_point[0]:
                rope_points = list(reversed(rope_points))

        return rope_points

    def _retain_largest_contour_safe(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_pixel_count = 1000
        height, width = mask.shape[:2]
        max_distance = 0.5 * width

        def get_contour_info(contour: np.ndarray) -> Tuple[float, float, int]:
            """Return (center_x, center_y, pixel_count_in_mask) for the contour."""
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2
                center_y = y + h / 2

            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            pixel_count = int(cv2.countNonZero(cv2.bitwise_and(mask, contour_mask)))

            return (center_x, center_y, pixel_count)

        contour_info = []
        for contour in contours:
            cx, cy, pixel_count = get_contour_info(contour)
            contour_info.append((contour, cx, cy, pixel_count))

        sorted_contours = sorted(contour_info, key=lambda x: x[3], reverse=True)
        keep_contours = [sorted_contours[0][0]]

        if len(sorted_contours) > 1:
            largest_cx, largest_cy, largest_pixels = sorted_contours[0][1], sorted_contours[0][2], sorted_contours[0][3]
            second_contour, second_cx, second_cy, second_pixels = sorted_contours[1]
            distance = np.hypot(largest_cx - second_cx, largest_cy - second_cy)
            if second_pixels >= min_pixel_count and distance <= max_distance:
                keep_contours.append(second_contour)

        contour_mask = np.zeros_like(mask)
        for contour in keep_contours:
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        return contour_mask

    def _fit_spline(self, rope_path: List[Tuple[int, int]], number_of_points: int) -> List[Tuple[int, int]]:
        if len(rope_path) < 4 or number_of_points < 2:
            return []
        try:
            x_coords = np.array([p[0] for p in rope_path], dtype=float)
            y_coords = np.array([p[1] for p in rope_path], dtype=float)
            k = min(3, len(x_coords) - 1)
            tck, _ = splprep([x_coords, y_coords], s=len(x_coords) * self.spline_smoothing_scale, k=k)
            u_new = np.linspace(0, 1, number_of_points)
            x_spline, y_spline = splev(u_new, tck)
            return [(int(x), int(y)) for x, y in zip(x_spline, y_spline)]
        except Exception:
            return []

    def _sample_path_points(self, rope_path: List[Tuple[int, int]], number_of_points: int) -> List[Tuple[int, int]]:
        if not rope_path:
            return []
        if number_of_points <= 1:
            return [rope_path[0]]
        if len(rope_path) <= number_of_points:
            return rope_path
        indices = np.linspace(0, len(rope_path) - 1, number_of_points).astype(int)
        return [rope_path[i] for i in indices]

    def _polyline_length(self, points: List[Tuple[int, int]]) -> float:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return float("nan")
        if not np.isfinite(arr).all():
            return float("nan")
        diffs = np.diff(arr[:, :2], axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _bresenham_line_pixels(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def _refresh_all_node_types(self, graph: nx.Graph) -> None:
        for node_id in list(graph.nodes()):
            degree = graph.degree(node_id)
            if degree == 0:
                graph.remove_node(node_id)
                continue
            graph.nodes[node_id]["type"] = "endpoint" if degree == 1 else "junction"
            

    def _get_edge_pixels_between_nodes(self, graph: nx.Graph, start_node_id: int, end_node_id: int) -> List[Tuple[int, int]]:
        pixels = list(graph.edges[start_node_id, end_node_id].get("pixels", []))
        if not pixels:
            return []
        start_pos = graph.nodes[start_node_id]["pos"]
        end_pos = graph.nodes[end_node_id]["pos"]
        if tuple(pixels[0]) == start_pos and tuple(pixels[-1]) == end_pos:
            return pixels
        if tuple(pixels[0]) == end_pos and tuple(pixels[-1]) == start_pos:
            return list(reversed(pixels))
        d_first = abs(pixels[0][0] - start_pos[0]) + abs(pixels[0][1] - start_pos[1])
        d_last = abs(pixels[-1][0] - start_pos[0]) + abs(pixels[-1][1] - start_pos[1])
        return pixels if d_first <= d_last else list(reversed(pixels))

    def _stitch_pixel_paths(self, path_a: List[Tuple[int, int]], path_b: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not path_a:
            return list(path_b)
        if not path_b:
            return list(path_a)
        end_a = path_a[-1]
        start_b = path_b[0]
        if end_a == start_b:
            return path_a + path_b[1:]
        if max(abs(end_a[0] - start_b[0]), abs(end_a[1] - start_b[1])) <= 1:
            return path_a + path_b
        bridge = self._bresenham_line_pixels(end_a[0], end_a[1], start_b[0], start_b[1])
        return path_a + bridge[1:] + path_b[1:]

    def _merge_junction_nodes(self, graph: nx.Graph, node_a_id: int, node_b_id: int, bridge_edge: Tuple[int, int]) -> None:
        if node_a_id not in graph or node_b_id not in graph:
            return
        keep_id = min(node_a_id, node_b_id)
        remove_id = max(node_a_id, node_b_id)
        if keep_id not in graph or remove_id not in graph or not graph.has_edge(keep_id, remove_id):
            return

        bridge_pixels = self._get_edge_pixels_between_nodes(graph, keep_id, remove_id)
        if graph.has_edge(*bridge_edge):
            graph.remove_edge(*bridge_edge)

        for neighbor_id in list(graph.neighbors(remove_id)):
            if neighbor_id == keep_id or not graph.has_edge(remove_id, neighbor_id):
                continue
            remove_to_neighbor_pixels = self._get_edge_pixels_between_nodes(graph, remove_id, neighbor_id)
            pixels = self._stitch_pixel_paths(bridge_pixels, remove_to_neighbor_pixels)
            graph.remove_edge(remove_id, neighbor_id)
            if not graph.has_edge(keep_id, neighbor_id):
                graph.add_edge(keep_id, neighbor_id, pixels=pixels, length=float(len(pixels)))
            else:
                existing_length = graph.edges[keep_id, neighbor_id].get("length", 0.0)
                new_length = float(len(pixels))
                if new_length > existing_length:
                    graph.edges[keep_id, neighbor_id]["pixels"] = pixels
                    graph.edges[keep_id, neighbor_id]["length"] = new_length

        if remove_id in graph:
            graph.remove_node(remove_id)
        self._refresh_all_node_types(graph)

    def _collapse_short_junction_edge(self, graph: nx.Graph, max_bridge_length: float = 10.0) -> None:
        changed = True
        while changed:
            changed = False
            self._refresh_all_node_types(graph)
            for u, v, data in list(graph.edges(data=True)):
                if data.get("length", 0.0) >= max_bridge_length:
                    continue
                if u in graph and v in graph and graph.nodes[u].get("type") == "junction" and graph.nodes[v].get("type") == "junction":
                    self._merge_junction_nodes(graph, u, v, (u, v))
                    changed = True
                    break

    def _collapse_pass_through_junctions(self, graph: nx.Graph) -> None:
        changed = True
        while changed:
            changed = False
            for node_id in list(graph.nodes()):
                if node_id not in graph:
                    continue
                if graph.nodes[node_id].get("type") != "junction" or graph.degree(node_id) != 2:
                    continue
                neighbors = list(graph.neighbors(node_id))
                if len(neighbors) != 2:
                    continue
                a, b = neighbors
                if a == b:
                    continue
                pixels_a = self._get_edge_pixels_between_nodes(graph, a, node_id)
                pixels_b = self._get_edge_pixels_between_nodes(graph, node_id, b)
                merged_pixels = pixels_a + pixels_b[1:]
                graph.remove_node(node_id)
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, pixels=merged_pixels, length=float(len(merged_pixels)))
                else:
                    existing_length = graph.edges[a, b].get("length", 0.0)
                    new_length = float(len(merged_pixels))
                    if new_length > existing_length:
                        graph.edges[a, b]["pixels"] = merged_pixels
                        graph.edges[a, b]["length"] = new_length
                changed = True
                break

    def _keep_only_farthest_endpoint_pair(self, graph: nx.Graph) -> None:
        endpoints = [n for n, d in graph.nodes(data=True) if d.get("type") == "endpoint" and graph.degree(n) == 1]
        if len(endpoints) < 2:
            endpoints = [n for n, d in graph.nodes(data=True) if d.get("type") == "endpoint"]
        if len(endpoints) <= 2:
            return

        best_pair = None
        best_distance = -1.0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                u, v = endpoints[i], endpoints[j]
                pair_best_path = -1.0
                try:
                    for path in nx.all_simple_paths(graph, source=u, target=v):
                        path_len = 0.0
                        for n1, n2 in zip(path[:-1], path[1:]):
                            path_len += graph.edges[n1, n2].get("length", 0.0)
                        if path_len > pair_best_path:
                            pair_best_path = path_len
                except nx.NetworkXNoPath:
                    continue

                if pair_best_path > best_distance:
                    best_distance = pair_best_path
                    best_pair = (u, v)

        if best_pair is None:
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    u, v = endpoints[i], endpoints[j]
                    try:
                        dist = nx.shortest_path_length(graph, source=u, target=v, weight="length")
                        if dist > best_distance:
                            best_distance = dist
                            best_pair = (u, v)
                    except nx.NetworkXNoPath:
                        continue

        if not best_pair:
            return
        keep_ids = set(best_pair)
        remove_ids = [n for n in endpoints if n not in keep_ids]
        touched_nodes: Set[int] = set()
        for endpoint_id in remove_ids:
            if endpoint_id in graph:
                touched_nodes.update(graph.neighbors(endpoint_id))
                graph.remove_node(endpoint_id)
        for node_id in touched_nodes:
            if node_id in graph:
                if graph.degree(node_id) == 0:
                    graph.remove_node(node_id)
                else:
                    graph.nodes[node_id]["type"] = "junction"
        self._refresh_all_node_types(graph)

    def _clean_graph_topology(self, graph: nx.Graph, bridge_merge_threshold: float = 10.0) -> nx.Graph:
        graph = graph.copy()
        self._collapse_short_junction_edge(graph, max_bridge_length=bridge_merge_threshold)
        self._refresh_all_node_types(graph)
        self._collapse_pass_through_junctions(graph)
        self._refresh_all_node_types(graph)
        self._keep_only_farthest_endpoint_pair(graph)
        self._collapse_pass_through_junctions(graph)
        return graph

    def _trace_longest_endpoint_path_no_overlap(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        endpoints = [n for n, d in graph.nodes(data=True) if d.get("type") == "endpoint"]
        if len(endpoints) < 2:
            if graph.number_of_edges() > 0:
                longest = max(graph.edges(data=True), key=lambda e: e[2].get("length", 0))
                return longest[2].get("pixels", [])
            return []

        best_path_pixels: List[Tuple[int, int]] = []
        max_len = -1.0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                u, v = endpoints[i], endpoints[j]
                for path in nx.all_simple_paths(graph, source=u, target=v):
                    path_len = 0.0
                    path_pixels: List[Tuple[int, int]] = []
                    for k in range(len(path) - 1):
                        n1, n2 = path[k], path[k + 1]
                        edge_data = graph.get_edge_data(n1, n2)
                        path_len += edge_data.get("length", 0)
                        pixels = edge_data.get("pixels", [])
                        if not pixels:
                            continue
                        if tuple(pixels[0]) != graph.nodes[n1]["pos"]:
                            pixels = pixels[::-1]
                        if k == 0:
                            path_pixels.extend(pixels)
                        else:
                            path_pixels.extend(pixels[1:])
                    if path_len > max_len:
                        max_len = path_len
                        best_path_pixels = path_pixels

        if not best_path_pixels and graph.number_of_edges() > 0:
            longest = max(graph.edges(data=True), key=lambda e: e[2].get("length", 0))
            return longest[2].get("pixels", [])
        return best_path_pixels

    def _filter_disjoint_components_by_area(self, mask: np.ndarray, min_area: int = 200) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if (num_labels - 1) <= 1:
            return (binary * 255).astype(np.uint8)

        filtered = np.zeros_like(binary)
        kept = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                filtered[labels == label] = 1
                kept += 1

        if kept == 0:
            largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
            filtered[labels == largest_label] = 1
        return (filtered * 255).astype(np.uint8)

    def _get_neighbors(self, x: int, y: int, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        neighbors: List[Tuple[int, int]] = []
        h, w = skeleton.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_pos, ny_pos = x + dx, y + dy
                if 0 <= nx_pos < w and 0 <= ny_pos < h and skeleton[ny_pos, nx_pos] > 0:
                    neighbors.append((nx_pos, ny_pos))
        return neighbors

    def _count_skeleton_neighbors(self, x: int, y: int, skeleton: np.ndarray) -> int:
        return len(self._get_neighbors(x, y, skeleton))

    def _skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        graph = nx.Graph()
        node_counter = 0
        node_positions = {}

        y_coords, x_coords = np.nonzero(skeleton)
        for y, x in zip(y_coords, x_coords):
            y = int(y)
            x = int(x)
            neighbor_count = self._count_skeleton_neighbors(x, y, skeleton)
            if neighbor_count != 2:
                node_type = "endpoint" if neighbor_count == 1 else "junction"
                graph.add_node(node_counter, pos=(x, y), type=node_type)
                node_positions[(x, y)] = node_counter
                node_counter += 1

        visited_pixels = set()
        for start_node_id in graph.nodes():
            start_pos = graph.nodes[start_node_id]["pos"]
            for neighbor_pos in self._get_neighbors(start_pos[0], start_pos[1], skeleton):
                edge_key = frozenset((start_pos, neighbor_pos))
                if edge_key in visited_pixels:
                    continue

                path_pixels = [start_pos]
                current_pos = neighbor_pos
                prev_pos = start_pos

                while current_pos not in node_positions:
                    path_pixels.append(current_pos)
                    visited_pixels.add(frozenset((prev_pos, current_pos)))
                    next_neighbors = [n for n in self._get_neighbors(current_pos[0], current_pos[1], skeleton) if n != prev_pos]
                    if not next_neighbors:
                        break
                    prev_pos = current_pos
                    current_pos = next_neighbors[0]

                if current_pos in node_positions:
                    end_node_id = node_positions[current_pos]
                    path_pixels.append(current_pos)
                    if start_node_id != end_node_id and not graph.has_edge(start_node_id, end_node_id):
                        graph.add_edge(start_node_id, end_node_id, pixels=path_pixels, length=float(len(path_pixels)))

        return graph

    def _get_endpoint_orientation(self, graph: nx.Graph, endpoint_id: int, tangent_pixels: int = 5) -> np.ndarray:
        """Get local tangent orientation at an endpoint (pointing outward)."""
        neighbors = list(graph.neighbors(endpoint_id))
        if not neighbors:
            return np.array([0, 0])

        neighbor_id = neighbors[0]

        edge_pixels = list(graph.edges[endpoint_id, neighbor_id].get("pixels", []))
        if len(edge_pixels) >= 2:
            endpoint_pos = tuple(graph.nodes[endpoint_id]["pos"])

            # Orient edge pixels so the endpoint is last.
            if tuple(edge_pixels[-1]) != endpoint_pos:
                edge_pixels = edge_pixels[::-1]

            # Use local tangent from k pixels behind endpoint.
            idx_prev = max(0, len(edge_pixels) - 1 - int(tangent_pixels))
            p_prev = np.array(edge_pixels[idx_prev], dtype=float)
            p_end = np.array(edge_pixels[-1], dtype=float)
            direction = p_end - p_prev
        else:
            pos_endpoint = np.array(graph.nodes[endpoint_id]["pos"], dtype=float)
            pos_neighbor = np.array(graph.nodes[neighbor_id]["pos"], dtype=float)
            direction = pos_endpoint - pos_neighbor

        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return direction

    def _compute_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in degrees (0-180)."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 180.0
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def _compute_component_length(self, graph: nx.Graph, component: set) -> float:
        """Compute the length of a component as the longest path between endpoints."""
        # Create a subgraph for this component
        subgraph = graph.subgraph(component)
        
        # Get endpoints
        endpoints = [nid for nid in component if graph.nodes[nid].get("type") == "endpoint"]
        if len(endpoints) < 2:
            # If less than 2 endpoints, sum all edge lengths
            total_length = 0.0
            for u, v, data in subgraph.edges(data=True):
                total_length += data.get("length", 1.0)
            return total_length
        
        # Find longest path between any two endpoints
        max_length = 0.0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                try:
                    path_length = nx.shortest_path_length(
                        subgraph, source=endpoints[i], target=endpoints[j], weight="length"
                    )
                    max_length = max(max_length, path_length)
                except nx.NetworkXNoPath:
                    pass
        
        return max_length

    def _connect_or_remove_disjoint_components(self, graph: nx.Graph, max_connect_distance: float = 60.0, max_angle_diff: float = 65.0) -> None:
        while True:
            components = list(nx.connected_components(graph))
            if len(components) <= 1:
                return
            
            # Sort components by rope length (edge-to-edge path), longest first
            components_with_length = [
                (comp, self._compute_component_length(graph, comp)) 
                for comp in components
            ]
            components_with_length.sort(key=lambda x: x[1], reverse=True)
            
            main_component = components_with_length[0][0]  # Longest graph - always keep
            other_components = [comp for comp, _ in components_with_length[1:]]
            
            # Get endpoints of main component
            main_endpoints = [nid for nid in main_component if graph.nodes[nid].get("type") == "endpoint"]
            if not main_endpoints:
                return
            
            # Find the closest component to the main component using nearest endpoints
            closest_comp = None
            closest_dist = float("inf")
            closest_pair = None
            
            for comp in other_components:
                # Get endpoints of this component
                comp_endpoints = [nid for nid in comp if graph.nodes[nid].get("type") == "endpoint"]
                if not comp_endpoints:
                    continue
                
                # Find min distance between endpoints of main_component and this comp
                min_dist = float("inf")
                min_pair = None
                for nid_a in main_endpoints:
                    for nid_b in comp_endpoints:
                        pos_a = graph.nodes[nid_a]["pos"]
                        pos_b = graph.nodes[nid_b]["pos"]
                        dist = np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = (nid_a, nid_b)
                
                # Check if this component is closer than current closest
                if min_dist < closest_dist:
                    closest_dist = min_dist
                    closest_comp = comp
                    closest_pair = min_pair
            
            if closest_comp is None or closest_pair is None:
                return
            
            # Test against distance threshold
            if closest_dist > max_connect_distance:
                # Distance too far - delete the shorter (closest) component
                for node_id in list(closest_comp):
                    if node_id in graph:
                        graph.remove_node(node_id)
                continue
            
            # Test against orientation threshold
            endpoint_main, endpoint_other = closest_pair
            pos_main = np.array(graph.nodes[endpoint_main]["pos"])
            pos_other = np.array(graph.nodes[endpoint_other]["pos"])
            
            # Get orientation vectors at both endpoints
            orient_main = self._get_endpoint_orientation(graph, endpoint_main)
            orient_other = self._get_endpoint_orientation(graph, endpoint_other)
            
            # Direction of the connection (from main to other)
            connection_dir = pos_other - pos_main
            connection_norm = np.linalg.norm(connection_dir)
            if connection_norm > 0:
                connection_dir = connection_dir / connection_norm
            
            angle_main = self._compute_angle_between_vectors(orient_main, connection_dir)
            angle_other = self._compute_angle_between_vectors(orient_other, -connection_dir)
            
            if angle_main > max_angle_diff or angle_other > max_angle_diff:
                for node_id in list(closest_comp):
                    if node_id in graph:
                        graph.remove_node(node_id)
                continue 
            
            # Check if combined length would exceed maximum
            main_length = self._compute_component_length(graph, main_component)
            closest_length = self._compute_component_length(graph, closest_comp)
            bridge_length = closest_dist  # Distance between the two endpoints
            combined_length = main_length + closest_length + bridge_length
            max_rope_length = 420.0
            
            if combined_length > max_rope_length:
                for node_id in list(closest_comp):
                    if node_id in graph:
                        graph.remove_node(node_id)
                continue
            
            # Distance, orientation, and length all OK - connect them
            u, v = closest_pair
            pos_u = graph.nodes[u]["pos"]
            pos_v = graph.nodes[v]["pos"]
            bridge_pixels = self._bresenham_line_pixels(pos_u[0], pos_u[1], pos_v[0], pos_v[1])
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, pixels=bridge_pixels, length=float(len(bridge_pixels)))
            # After connecting, the components merge, so continue to check for more


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rope segmentation utility on one image.")
    parser.add_argument("image_path", type=Path, help="Path to input image")
    parser.add_argument("--num-points", type=int, default=20, help="Number of rope points to output")
    args = parser.parse_args()

    image = cv2.imread(str(args.image_path))
    if image is None:
        print(f"Failed to read image: {args.image_path}")
        return

    util = RopeSegmentationUtil(number_of_points=args.num_points)
    rope_points = util.get_rope_points(image)

    print(f"Image: {args.image_path}")
    print(f"Rope points ({len(rope_points)}): {rope_points}")


if __name__ == "__main__":
    main()
