#!/usr/bin/env python3
"""
VLMap Data Recorder Node  (robust version)
==========================================
Replaces ApproximateTimeSynchronizer with a manual rolling-cache sync so
QoS mismatches and bad header stamps never silently block recording.

Subscribes to:
  - /camera/image_raw            (sensor_msgs/Image)
  - /depth_camera/depth          (sensor_msgs/Image)
  - /pose                        (geometry_msgs/PoseStamped)

Writes to  <output_dir>/
  color/   000000.png …
  depth/   000000.npy …          (float32, metres)
  poses.txt                      (x y z qx qy qz qw, one line per frame)

Usage
-----
  ros2 run <pkg> data_recorder_node \
      --ros-args \
      -p output_dir:=/tmp/vlmap_recording \
      -p pose_topic:=/pose \
      -p min_translation_m:=0.15 \
      -p min_rotation_deg:=5.0 \
      -p sync_slop_sec:=0.1
"""

import os
import math
import threading
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import  qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


# ── helpers ──────────────────────────────────────────────────────────────────

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def angle_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return 2.0 * math.acos(abs(dot))


def depth_to_float32(img_msg: Image, bridge: CvBridge) -> np.ndarray:
    enc = img_msg.encoding.lower()
    if enc == "32fc1":
        depth = np.array(bridge.imgmsg_to_cv2(img_msg, "32FC1"), dtype=np.float32)
    elif enc in ("16uc1", "mono16"):
        depth = bridge.imgmsg_to_cv2(img_msg, "16UC1").astype(np.float32) / 1000.0
    else:
        depth = np.array(bridge.imgmsg_to_cv2(img_msg), dtype=np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


# ── node ─────────────────────────────────────────────────────────────────────

class DataRecorderNode(Node):

    def __init__(self):
        super().__init__("vlmap_data_recorder")

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter("output_dir",        "/tmp/vlmap_recording")
        self.declare_parameter("pose_topic",        "/camera_pose")
        self.declare_parameter("min_translation_m", 0.10)
        self.declare_parameter("min_rotation_deg",  5.0)
        self.declare_parameter("sync_slop_sec",     0.05)
        self.declare_parameter("session_name",      "first")

        self.output_dir  = self.get_parameter("output_dir").value
        self.pose_topic  = self.get_parameter("pose_topic").value
        self.min_trans   = self.get_parameter("min_translation_m").value
        self.min_rot_rad = math.radians(self.get_parameter("min_rotation_deg").value)
        self.slop        = self.get_parameter("sync_slop_sec").value
        self.session_name = self.get_parameter("session_name").value
        # ── output dirs ──────────────────────────────────────────────────────
        self.color_dir = os.path.join(self.output_dir,self.session_name, "rgb")
        self.depth_dir = os.path.join(self.output_dir,self.session_name, "depth")
        self.pose_dir = os.path.join(self.output_dir, self.session_name, "pose")
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)
        # self.poses_file = open(os.path.join(self.output_dir, "poses.txt"), "w")
        # self.poses_file.write("# x y z qx qy qz qw\n")

        # ── state ────────────────────────────────────────────────────────────
        self.bridge     = CvBridge()
        self.frame_idx  = 0
        self.last_pos   = None
        self.last_quat  = None
        self._lock      = threading.Lock()

        # per-topic receive counters for diagnostics
        self._rx = {"rgb": 0, "depth": 0, "pose": 0}

        # rolling message caches: list of (stamp_sec, msg)
        self._rgb_cache   = []
        self._depth_cache = []
        self._pose_cache  = []
        self._cache_max   = 60

        # ── subscribers (BEST_EFFORT — typical for Gazebo sensor streams) ────
        # If [DIAG] shows 0 messages on any topic after a few seconds, that
        # topic is publishing with RELIABLE QoS.  Change sensor_qos below to
        # reliable_qos and restart.
        sensor_qos = qos_profile_sensor_data   # BEST_EFFORT, KEEP_LAST(10)


        self.create_subscription(Image,       "/camera/image_raw",   self._rgb_cb,   sensor_qos)
        self.create_subscription(Image,       "/depth_camera/depth", self._depth_cb, sensor_qos)
        self.create_subscription(PoseStamped, self.pose_topic,       self._pose_cb,  sensor_qos)

        # ── diagnostics timer ────────────────────────────────────────────────
        self.create_timer(3.0, self._diag_cb)

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  VLMap Data Recorder (manual-sync)\n"
            f"  output  : {self.output_dir}\n"
            f"  topics  : /camera/image_raw\n"
            f"            /depth_camera/depth\n"
            f"            {self.pose_topic}\n"
            f"  Δpos >= {self.min_trans:.2f} m   "
            f"Δrot >= {math.degrees(self.min_rot_rad):.1f} deg   "
            f"slop = {self.slop:.3f} s\n"
            f"{'='*60}"
        )

    # ── individual topic callbacks ────────────────────────────────────────────

    def _rgb_cb(self, msg: Image):
        self._rx["rgb"] += 1
        t = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._rgb_cache.append((t, msg))
            if len(self._rgb_cache) > self._cache_max:
                self._rgb_cache.pop(0)
        self._try_sync()

    def _depth_cb(self, msg: Image):
        self._rx["depth"] += 1
        t = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._depth_cache.append((t, msg))
            if len(self._depth_cache) > self._cache_max:
                self._depth_cache.pop(0)
        self._try_sync()

    def _pose_cb(self, msg: PoseStamped):
        self._rx["pose"] += 1
        t = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._pose_cache.append((t, msg))
            if len(self._pose_cache) > self._cache_max:
                self._pose_cache.pop(0)
        self._try_sync()

    # ── manual sync ───────────────────────────────────────────────────────────

    def _try_sync(self):
        """
        Takes the newest RGB stamp as reference, finds the closest depth and
        pose within `slop` seconds.  Falls back to wall-clock (newest-of-each)
        if all stamps are zero (publisher not setting header.stamp).
        """
        with self._lock:
            if not (self._rgb_cache and self._depth_cache and self._pose_cache):
                return

            t_ref, rgb_msg = self._rgb_cache[-1]

            # ── zero-stamp fallback ──────────────────────────────────────────
            if t_ref == 0.0:
                self.get_logger().warn(
                    "header.stamp is zero on RGB — using newest-of-each fallback. "
                    "Set use_sim_time:=true or stamp your messages.",
                    throttle_duration_sec=10.0,
                )
                _, depth_msg = self._depth_cache[-1]
                _, pose_msg  = self._pose_cache[-1]
                self._rgb_cache.clear()
                self._depth_cache.clear()
                self._pose_cache.clear()
            else:
                depth_match = self._closest(self._depth_cache, t_ref)
                pose_match  = self._closest(self._pose_cache,  t_ref)

                if depth_match is None or pose_match is None:
                    return   # nothing within slop yet — wait for more messages

                t_d, depth_msg = depth_match
                t_p, pose_msg  = pose_match

                # purge everything up to and including the matched stamps
                self._rgb_cache.clear()
                self._depth_cache = [x for x in self._depth_cache if x[0] > t_d]
                self._pose_cache  = [x for x in self._pose_cache  if x[0] > t_p]

        self._record(rgb_msg, depth_msg, pose_msg)

    def _closest(self, cache, t_ref):
        best, best_dt = None, float("inf")
        for t, msg in cache:
            dt = abs(t - t_ref)
            if dt < best_dt and dt <= self.slop:
                best_dt = dt
                best = (t, msg)
        return best

    # ── save one frame ────────────────────────────────────────────────────────

    def _record(self, rgb_msg: Image, depth_msg: Image, pose_msg: PoseStamped):
        p    = pose_msg.pose.position
        q    = pose_msg.pose.orientation
        pos  = np.array([p.x, p.y, p.z],      dtype=np.float64)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        # ── motion threshold ─────────────────────────────────────────────────
        with self._lock:
            if self.last_pos is not None:
                d_pos = float(np.linalg.norm(pos - self.last_pos))
                d_rot = angle_between_quats(quat, self.last_quat)
                if d_pos < self.min_trans and d_rot < self.min_rot_rad:
                    return
            self.last_pos  = pos.copy()
            self.last_quat = quat.copy()
            idx = self.frame_idx
            self.frame_idx += 1

        # ── RGB ──────────────────────────────────────────────────────────────
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            cv2.imwrite(os.path.join(self.color_dir, f"{self.session_name}_{idx:06d}.png"), bgr)
        except Exception as e:
            self.get_logger().error(f"[{idx}] RGB save error: {e}")
            return

        # ── Depth ────────────────────────────────────────────────────────────
        try:
            depth = depth_to_float32(depth_msg, self.bridge)
            np.save(os.path.join(self.depth_dir, f"{self.session_name}_{idx:06d}.npy"), depth)
        except Exception as e:
            self.get_logger().error(f"[{idx}] Depth save error: {e}")
            return

        # ── Pose ─────────────────────────────────────────────────────────────
        pose_file = open(os.path.join(self.pose_dir, f"{self.session_name}_{idx:06d}.txt"), "w")
        x, y, z         = pos
        qx, qy, qz, qw  = quat
        pose_file.write(
            f"{x:.8f} {y:.8f} {z:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
        )
        pose_file.close()

        self.get_logger().info(
            # f"[{idx:05d}] saved  pos=({x:.3f},{y:.3f},{z:.3f})  "
            # f"depth {depth.shape}  range=[{depth.min():.2f},{depth.max():.2f}] m"
            f'Saved data for frame {idx:05d} at pose ({x:.3f}, {y:.3f}, {z:.3f})'
        )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def _diag_cb(self):
        rx = self._rx
        self.get_logger().info(
            f"[DIAG] received → rgb:{rx['rgb']}  depth:{rx['depth']}  "
            f"pose:{rx['pose']}  |  frames saved: {self.frame_idx}"
        )

        for name, count in rx.items():
            if count == 0:
                self.get_logger().warn(
                    f"[DIAG] '{name}' = 0 messages received. "
                    f"Check topic name and QoS with:\n"
                    f"  ros2 topic hz  <topic>\n"
                    f"  ros2 topic info <topic> --verbose"
                )

        # sim-time vs wall-clock skew check
        with self._lock:
            for name, cache in [("rgb", self._rgb_cache),
                                 ("depth", self._depth_cache),
                                 ("pose", self._pose_cache)]:
                if cache:
                    t = cache[-1][0]
                    wall = time.time()
                    if t != 0.0 and abs(t - wall) > 30:
                        self.get_logger().warn(
                            f"[DIAG] '{name}' stamp ({t:.1f}) vs wall ({wall:.1f}): "
                            f"skew={abs(t-wall):.0f}s — "
                            f"run with --ros-args -p use_sim_time:=true if using Gazebo clock."
                        )

    # ── cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        # self.poses_file.close()
        self.get_logger().info(f"Stopped. Frames saved: {self.frame_idx}")
        super().destroy_node()


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()