#!/usr/bin/env python3
"""
vlmap_nav_node.py  —  Proof-of-concept VLMap → Nav2 navigation node.

Pipeline
--------
1. Load a pre-built VLMap (CLIP embeddings + grid coordinates).
2. Subscribe to /vlmap/query  (std_msgs/String).
3. On each query  →  compute cosine similarity over the map grid.
4. Find the cell with max similarity, convert its grid index to a
   world coordinate, and send a NavigateToPose goal to Nav2.

─────────────────────────────────────────────────────────────────────
COORDINATE FRAME NOTES (read this before testing)
─────────────────────────────────────────────────────────────────────
VLMap origin
  The map is built relative to the robot's starting pose during
  recording.  Cell (row=0, col=0) == the robot's position at t=0.
  Positive X  → forward,  Positive Y → left  (ROS REP-105).

Nav2 / map frame origin
  Nav2 expects goals in the `map` frame whose origin is defined by
  your map YAML file (usually the bottom-left corner of the PGM
  image, or wherever your SLAM session started).

How to reconcile them
  Option A – Same SLAM session  (most common for VLMap):
    If VLMap was built during the same SLAM run, set
      MAP_ORIGIN_X = 0.0
      MAP_ORIGIN_Y = 0.0
    The robot start == SLAM start, so frames already coincide.

  Option B – VLMap built offline / different session:
    Manually drive the robot to the VLMap starting location, read
    its (x, y) in the `map` frame via `ros2 topic echo /amcl_pose`,
    and put those values in MAP_ORIGIN_X / MAP_ORIGIN_Y below.

  Option C – Use TF (production approach):
    Replace the simple offset math with a tf2 lookup from
    `vlmap_origin` → `map`, publishing a static transform that
    anchors the VLMap origin in the map frame.

─────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import torch
import clip
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

# ── optional: pretty logging ─────────────────────────────────────────
try:
    import colorlog, logging

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s[%(levelname)s]%(reset)s %(message)s")
    )
    logger = colorlog.getLogger("vlmap_nav")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
except ImportError:
    import logging

    logger = logging.getLogger("vlmap_nav")
    logging.basicConfig(level=logging.DEBUG)


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION  
# ═══════════════════════════════════════════════════════════════════════

# Path to  saved VLMap data.
# Expected layout (numpy .npz):
#   embeddings : (N, D)  float32  –  CLIP feature per grid cell
#   positions  : (N, 2)  float32  –  (x, y) in VLMap frame [metres]
#   (optional) cell_size: scalar  –  metres per cell (default 0.05 m)
MAP_SAVE_PATH = os.path.expanduser("/tmp/vlmap_recording/test7/map/")

# Coordinate offset: VLMap origin in the Nav2 `map` frame (metres).
# See the COORDINATE FRAME NOTES above.
# MAP_ORIGIN_X = -3.241
# MAP_ORIGIN_Y = -1.939

MAP_ORIGIN_X = -3.241
MAP_ORIGIN_Y = -1.939
# Navigation frame that Nav2 listens to.
NAV_FRAME = "map"

# Similarity threshold: cells below this score are ignored.
# Range 0–1.  Lower  → more permissive;  higher → stricter.
SIM_THRESHOLD = 0.20

# How many top-K cells to average when computing the goal position.
# 1  → just the single best cell (noisy).
# 5  → centroid of the top-5 cells (smoother).
TOP_K = 5

# Fixed heading for the navigation goal [radians].
# 0.0  → robot faces +X (East on a standard ROS map).
GOAL_YAW = 0.0
GS = 150
CS = 0.05


def pos2grid_id(gs, cs, xx, yy):
    x = int(gs / 2 + int(xx / cs))
    y = int(gs / 2 - int(yy / cs))
    return [x, y]


def grid_id2pos(gs, cs, x, y):
    xx = (x - gs / 2) * cs
    zz = (gs / 2 - y) * cs

    return xx, zz

# ═══════════════════════════════════════════════════════════════════════
#  VLMap loader
# ═══════════════════════════════════════════════════════════════════════


class VLMap:
    """Load a pre-built VLMap and run text-similarity queries on it."""

    def __init__(self, map_save_dir: str):
        color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down.npy")
        grid_save_path = os.path.join(map_save_dir, f"grid_lseg.npy")
        obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

        def load_map(load_path):
            with open(load_path, "rb") as f:
                map = np.load(f)
            return map
        logger.info(
            f"Map loaded "
        )

        self.grid = load_map(grid_save_path)
        self.obstacles = np.load(obstacles_save_path)
        self.model, self.preprocess = self._load_clip_model()

    def _load_clip_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_version = "ViT-B/32"
        clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[clip_version]
        clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
        clip_model.to(device).eval()

        return clip_model, preprocess

    def query(
        self, query: np.ndarray, top_k: int = TOP_K, threshold: float = SIM_THRESHOLD
    ):
        text_tokens = clip.tokenize([query])  # (1, token_len)
        text_tokens = text_tokens.to(next(self.model.parameters()).device)
        with torch.no_grad():
            text_feats = self.model.encode_text(text_tokens)
        # text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-8)

        text_feats = text_feats.float().detach().cpu().numpy()  # (D,)

        # compute cosine similarity
        similarity = self.grid @ text_feats.T  # (N,)
        from matplotlib import pyplot as plt
        # plt.imshow(similarity)
        # plt.show()

        top_k_indices = np.argsort(similarity, axis=None)[-top_k:]
        top_k_coords = np.unravel_index(top_k_indices, similarity.shape)


        x = top_k_coords[1][0]
        y = top_k_coords[0][0]
        logger.info(f'Vlmap coordinate of the best cell: ({x}, {y})')
        return x, y


# ═══════════════════════════════════════════════════════════════════════
#  ROS2 Node
# ═══════════════════════════════════════════════════════════════════════


def yaw_to_quaternion(yaw: float):
    """Convert a heading angle (radians) to a geometry_msgs Quaternion."""
    from geometry_msgs.msg import Quaternion
    import math

    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q




class VLMapNavNode(Node):

    def __init__(self):
        super().__init__("vlmap_nav_node")

        # ── parameters (overridable via ros2 param set) ────────────────
        self.declare_parameter("vlmap_path", MAP_SAVE_PATH)
        self.declare_parameter("map_origin_x", MAP_ORIGIN_X)
        self.declare_parameter("map_origin_y", MAP_ORIGIN_Y)
        self.declare_parameter("nav_frame", NAV_FRAME)
        self.declare_parameter("sim_threshold", SIM_THRESHOLD)
        self.declare_parameter("top_k", TOP_K)
        self.declare_parameter("goal_yaw", GOAL_YAW)

        vlmap_path = self.get_parameter("vlmap_path").value
        self._origin_x = self.get_parameter("map_origin_x").value
        self._origin_y = self.get_parameter("map_origin_y").value
        self._frame = self.get_parameter("nav_frame").value
        self._thresh = self.get_parameter("sim_threshold").value
        self._top_k = self.get_parameter("top_k").value
        self._yaw = self.get_parameter("goal_yaw").value

        # ── load map and encoder ───────────────────────────────────────
        self.get_logger().info(f"Loading VLMap from {vlmap_path} …")
        self._vlmap = VLMap(vlmap_path)

        # ── Nav2 action client ─────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.get_logger().info("Waiting for navigate_to_pose action server …")
        self._nav_client.wait_for_server()
        self.get_logger().info("Action server ready.")

        # ── debug publisher: the goal pose for visualisation in RViz ───
        self._goal_pub = self.create_publisher(PoseStamped, "/vlmap/goal_pose", 10)

        # ── query subscriber ───────────────────────────────────────────
        self._query_sub = self.create_subscription(
            String, "/vlmap/query", self._query_cb, 10
        )

        self.get_logger().info(
            "vlmap_nav_node ready.  " "Publish a query to /vlmap/query to navigate."
        )

    # ─────────────────────────────────────────────────────────────────

    def _query_cb(self, msg: String):
        query = msg.data.strip()
        if not query:
            return
        self.get_logger().info(f'Query received: "{query}"')


        # 1. Similarity over map
        result = self._vlmap.query(query=query, threshold=self._thresh, top_k=self._top_k)
        if result is None:
            return

        vlmap_x, vlmap_y= result
        logger.info(f"VLMap query result: ({vlmap_x}, {vlmap_y})  ")

        nav_x , nav_y = grid_id2pos(GS, CS, vlmap_x, vlmap_y)
        self.get_logger().info(
            f"VLMap coords ({vlmap_x:.2f}, {vlmap_y:.2f}) m  →  "
            f"map frame ({nav_x:.2f}, {nav_y:.2f}) m  "
        )

        # 4. Check if goal is reachable and obstacle-free
        goal_grid_x, goal_grid_y = pos2grid_id(GS, CS, nav_x, nav_y)
        
        # Check if the original goal is free
        if not self._is_cell_free(goal_grid_x, goal_grid_y):
            self.get_logger().warn(
                f"Original goal ({goal_grid_x}, {goal_grid_y}) is in an obstacle. "
                "Finding nearest free cell…"
            )
            # Get 8 surrounding points and find the closest free one
            safe_grid_x, safe_grid_y = self._find_nearest_free_cell(
                goal_grid_x, goal_grid_y
            )
            if safe_grid_x is None:
                self.get_logger().error("No free cells found near goal. Aborting.")
                return
            nav_x, nav_y = grid_id2pos(GS, CS, safe_grid_x, safe_grid_y)
            self.get_logger().info(
                f"Using safe cell ({safe_grid_x}, {safe_grid_y}) → "
                f"({nav_x:.2f}, {nav_y:.2f}) m"
            )
        
        # 5. Build & send goal
        goal_pose = self._build_pose(nav_x, nav_y)
        self._goal_pub.publish(goal_pose)  # visible in RViz immediately
        self._send_nav_goal(goal_pose)

    # ─────────────────────────────────────────────────────────────────

    def _is_cell_free(self, grid_x: int, grid_y: int) -> bool:
        """Check if a grid cell is free (not an obstacle)."""
        # convert grid_x , grid_y to grid_id
        if grid_x < 0 or grid_x >= GS or grid_y < 0 or grid_y >= GS:
            return False
        return self._vlmap.obstacles[grid_y, grid_x] == 0

    def _find_nearest_free_cell(self, grid_x: int, grid_y: int):
        """
        Find the nearest free cell among the 8 surrounding cells.
        Returns (grid_x, grid_y) of the nearest free cell, or (None, None) if none found.
        """
        # 8 surrounding offsets: (dx, dy)
        offsets = [
            (-4, -4), (0, -4), (4, -4),  # top row
            (-4, 0),          (4, 0),    # middle row (exclude center)
            (-4, 4),  (0, 4),  (4, 4),   # bottom row
        ]
        
        best_cell = None
        best_dist = float("inf")
        
        for dx, dy in offsets:
            nx, ny = grid_x + dx, grid_y + dy
            if self._is_cell_free(nx, ny):
                # Compute distance (Euclidean)
                dist = np.sqrt(dx**2 + dy**2)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = (nx, ny)
        
        if best_cell is None:
            return None, None
        return best_cell

    # ─────────────────────────────────────────────────────────────────
 
    def _build_pose(self, x: float, y: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self._frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quaternion(self._yaw)
        return ps

    def _send_nav_goal(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f"Sending NavigateToPose goal → "
            f"({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})"
        )

        send_future = self._nav_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb
        )
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal REJECTED by Nav2.")
            return
        self.get_logger().info("Goal ACCEPTED by Nav2.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = fb.distance_remaining
        self.get_logger().debug(
            f"Distance remaining: {dist:.2f} m", throttle_duration_sec=2.0
        )

    def _result_cb(self, future):
        result = future.result().result
        self.get_logger().info(f"Navigation finished. Result: {result}")


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════


def main(args=None):
    rclpy.init(args=args)
    node = VLMapNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
