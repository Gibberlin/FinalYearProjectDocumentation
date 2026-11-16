"""CARLA-only cooperative-perception prototype using compact late fusion.

This is a research/demo baseline, not a reproduction of V2X-ViT, Where2comm,
or CoBEVT, and not a vehicle controller. Both vehicles remain under CARLA's
Traffic Manager. The detector's COCO classes are projected into approximate
world positions using a paired depth camera, then exchanged over a simulated
delay/loss link and fused at the ego vehicle.

OPV2V motivates the CARLA multi-agent setting; Where2comm motivates recording
communication cost and freshness. This baseline does not implement their
published models, nor V2X-ViT or CoBEVT. It uses a generic COCO detector.

Requires a running CARLA server, its Python API, NumPy, OpenCV, and Ultralytics.
The first Ultralytics run may download the selected COCO detector weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


COCO_CLASSES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x: float
    y: float
    z: float
    box: tuple[int, int, int, int] | None = None


class CameraPair:
    """Wait for matching RGB and depth camera frames from one vehicle."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._rgb: dict[int, tuple[Any, float]] = {}
        self._depth: dict[int, tuple[Any, float]] = {}

    def rgb_callback(self, image: Any) -> None:
        import numpy as np

        frame = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = frame.reshape((image.height, image.width, 4))[:, :, :3].copy()
        with self._condition:
            self._rgb[image.frame] = (frame, image.timestamp)
            self._condition.notify_all()

    def depth_callback(self, image: Any) -> None:
        import numpy as np

        frame = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = frame.reshape((image.height, image.width, 4)).copy()
        with self._condition:
            self._depth[image.frame] = (frame, image.timestamp)
            self._condition.notify_all()

    def get(self, frame_number: int, timeout: float) -> tuple[Any, Any, float]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while frame_number not in self._rgb or frame_number not in self._depth:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for RGB/depth frame {frame_number}"
                    )
                self._condition.wait(remaining)

            rgb, timestamp = self._rgb.pop(frame_number)
            depth, _ = self._depth.pop(frame_number)
            old_frames = [
                number
                for number in set(self._rgb) | set(self._depth)
                if number < frame_number
            ]
            for number in old_frames:
                self._rgb.pop(number, None)
                self._depth.pop(number, None)
            return rgb, depth, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CARLA cooperative-perception baseline (simulation only)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Ultralytics COCO-compatible detector weights",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--width", type=int, default=640, help="Camera image width")
    parser.add_argument("--height", type=int, default=360, help="Camera image height")
    parser.add_argument("--fov", type=float, default=100.0, help="Camera field of view")
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=0,
        help="CARLA map spawn-point index for the ego vehicle",
    )
    parser.add_argument(
        "--peer-spawn-index",
        type=int,
        default=None,
        help="Peer spawn-point index (by default, choose a nearby aligned point)",
    )
    parser.add_argument(
        "--latency-ms", type=float, default=100.0, help="Simulated one-way link delay"
    )
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=0.0,
        help="Probability of dropping each peer message, from 0 to 1",
    )
    parser.add_argument(
        "--max-message-age-ms",
        type=float,
        default=500.0,
        help="Discard received peer detections older than this",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35, help="Detector confidence threshold"
    )
    parser.add_argument(
        "--fusion-distance-m",
        type=float,
        default=4.0,
        help="Maximum same-class distance for pairing detections",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=80.0,
        help="Ignore depth-camera detections farther away than this",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Synchronous CARLA step in seconds",
    )
    parser.add_argument(
        "--frames", type=int, default=0, help="Stop after this many frames; 0 runs until quit"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Run without an OpenCV display window"
    )
    args = parser.parse_args()
    if not 0.0 <= args.packet_loss <= 1.0:
        parser.error("--packet-loss must be between 0 and 1")
    if args.latency_ms < 0 or args.max_message_age_ms < 0:
        parser.error("latency and maximum message age must not be negative")
    if args.fixed_delta <= 0:
        parser.error("--fixed-delta must be greater than zero")
    if args.width <= 0 or args.height <= 0 or args.fov <= 0:
        parser.error("camera dimensions and field of view must be positive")
    if args.fov >= 180 or args.max_depth_m <= 0:
        parser.error("--fov must be below 180 and --max-depth-m must be positive")
    if not 0.0 <= args.confidence <= 1.0 or args.fusion_distance_m <= 0:
        parser.error("--confidence must be in [0, 1] and fusion distance must be positive")
    if args.frames < 0:
        parser.error("--frames must not be negative")
    return args


def choose_peer_spawn(
    spawn_points: list[Any], ego_index: int, requested_index: int | None
) -> int:
    if len(spawn_points) < 2:
        raise RuntimeError("The current CARLA map needs at least two vehicle spawn points.")
    if requested_index is not None:
        if not 0 <= requested_index < len(spawn_points):
            raise ValueError(
                f"--peer-spawn-index must be from 0 to {len(spawn_points) - 1}"
            )
        if requested_index == ego_index:
            raise ValueError("Ego and peer vehicles must use different spawn points.")
        return requested_index

    ego = spawn_points[ego_index]
    candidates: list[tuple[float, int]] = []
    for index, candidate in enumerate(spawn_points):
        if index == ego_index:
            continue
        distance = ego.location.distance(candidate.location)
        yaw_delta = abs(
            (ego.rotation.yaw - candidate.rotation.yaw + 180.0) % 360.0 - 180.0
        )
        if 8.0 <= distance <= 35.0 and yaw_delta <= 35.0:
            candidates.append((abs(distance - 18.0) + yaw_delta / 4.0, index))
    if not candidates:
        raise RuntimeError(
            "Could not find a nearby, similarly aligned peer spawn point. "
            "Choose --peer-spawn-index explicitly."
        )
    return min(candidates)[1]


def decode_depth(raw_depth: Any) -> Any:
    import numpy as np

    # CARLA encodes normalized 24-bit depth into the RGB channels of BGRA data.
    red = raw_depth[:, :, 2].astype(np.float32)
    green = raw_depth[:, :, 1].astype(np.float32)
    blue = raw_depth[:, :, 0].astype(np.float32)
    return (red + green * 256.0 + blue * 65536.0) * (1000.0 / 16777215.0)


def detect_objects(
    model: Any,
    image: Any,
    raw_depth: Any,
    camera_transform: Any,
    fov: float,
    confidence: float,
    max_depth_m: float,
    device: str | None,
) -> list[Detection]:
    import carla
    import numpy as np

    results = model.predict(
        source=image,
        conf=confidence,
        classes=list(COCO_CLASSES),
        device=device,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []

    depth_map = decode_depth(raw_depth)
    image_height, image_width = depth_map.shape
    focal_length = image_width / (2.0 * math.tan(math.radians(fov) / 2.0))
    detections: list[Detection] = []
    boxes = results[0].boxes
    coordinates = boxes.xyxy.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()

    for coords, class_id, score in zip(coordinates, class_ids, confidences):
        x1, y1, x2, y2 = (int(round(value)) for value in coords)
        x1 = max(0, min(image_width - 1, x1))
        x2 = max(x1 + 1, min(image_width, x2))
        y1 = max(0, min(image_height - 1, y1))
        y2 = max(y1 + 1, min(image_height, y2))
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        patch_width = max(1, (x2 - x1) // 5)
        patch_height = max(1, (y2 - y1) // 5)
        patch = depth_map[
            max(y1, center_y - patch_height) : min(y2, center_y + patch_height + 1),
            max(x1, center_x - patch_width) : min(x2, center_x + patch_width + 1),
        ]
        valid_depths = patch[
            np.isfinite(patch) & (patch > 0.1) & (patch <= max_depth_m)
        ]
        if valid_depths.size == 0:
            continue
        distance = float(np.median(valid_depths))
        camera_y = (center_x - image_width / 2.0) * distance / focal_length
        camera_z = -(center_y - image_height / 2.0) * distance / focal_length
        world_point = camera_transform.transform(
            # CARLA camera coordinates: forward is x, image-right is y, up is z.
            carla.Location(x=distance, y=camera_y, z=camera_z)
        )
        detections.append(
            Detection(
                class_id=int(class_id),
                confidence=float(score),
                x=float(world_point.x),
                y=float(world_point.y),
                z=float(world_point.z),
                box=(x1, y1, x2, y2),
            )
        )
    return detections


def serialize_message(frame: int, sim_time: float, detections: list[Detection]) -> bytes:
    payload = {
        "frame": frame,
        "sim_time": round(sim_time, 4),
        "detections": [
            {
                "class_id": detection.class_id,
                "confidence": round(detection.confidence, 3),
                "position": [
                    round(detection.x, 2),
                    round(detection.y, 2),
                    round(detection.z, 2),
                ],
            }
            for detection in detections
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def fuse_detections(
    local: list[Detection],
    remote: list[Detection],
    distance_threshold: float,
) -> list[Detection]:
    fused: list[Detection] = []
    matched_remote: set[int] = set()
    for local_detection in local:
        candidates = [
            (
                math.dist(
                    (local_detection.x, local_detection.y, local_detection.z),
                    (remote_detection.x, remote_detection.y, remote_detection.z),
                ),
                index,
            )
            for index, remote_detection in enumerate(remote)
            if index not in matched_remote
            and remote_detection.class_id == local_detection.class_id
        ]
        if not candidates:
            fused.append(local_detection)
            continue
        distance, remote_index = min(candidates)
        if distance > distance_threshold:
            fused.append(local_detection)
            continue
        remote_detection = remote[remote_index]
        matched_remote.add(remote_index)
        local_weight = max(local_detection.confidence, 1e-6)
        remote_weight = max(remote_detection.confidence, 1e-6)
        total_weight = local_weight + remote_weight
        fused.append(
            Detection(
                class_id=local_detection.class_id,
                confidence=1.0
                - (1.0 - local_detection.confidence)
                * (1.0 - remote_detection.confidence),
                x=(local_detection.x * local_weight + remote_detection.x * remote_weight)
                / total_weight,
                y=(local_detection.y * local_weight + remote_detection.y * remote_weight)
                / total_weight,
                z=(local_detection.z * local_weight + remote_detection.z * remote_weight)
                / total_weight,
                box=local_detection.box,
            )
        )
    fused.extend(
        detection for index, detection in enumerate(remote) if index not in matched_remote
    )
    return fused


def main() -> int:
    args = parse_args()
    try:
        import carla
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as error:
        print(
            f"Missing dependency: {error}. Install NumPy, OpenCV, Ultralytics, "
            "and the CARLA Python API compatible with your CARLA server.",
            file=sys.stderr,
        )
        return 2

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.fixed_delta
    traffic_manager = client.get_trafficmanager(args.tm_port)
    original_tm_sync = original_settings.synchronous_mode

    actors: list[Any] = []
    display_open = False
    inference_times: list[float] = []
    total_bytes = 0
    sent_messages = 0
    dropped_messages = 0
    received_messages = 0

    try:
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        spawn_points = world.get_map().get_spawn_points()
        ego_index = args.spawn_index
        if not 0 <= ego_index < len(spawn_points):
            raise ValueError(f"--spawn-index must be from 0 to {len(spawn_points) - 1}")
        peer_index = choose_peer_spawn(spawn_points, ego_index, args.peer_spawn_index)

        vehicle_blueprints = world.get_blueprint_library().filter("vehicle.*")
        preferred = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego_blueprint = preferred if preferred else random.choice(vehicle_blueprints)
        peer_blueprint = random.choice(vehicle_blueprints)
        ego_blueprint.set_attribute("role_name", "ego")
        peer_blueprint.set_attribute("role_name", "peer")
        ego_vehicle = world.try_spawn_actor(ego_blueprint, spawn_points[ego_index])
        if ego_vehicle is None:
            raise RuntimeError(f"Could not spawn ego vehicle at point {ego_index}.")
        actors.append(ego_vehicle)
        peer_vehicle = world.try_spawn_actor(peer_blueprint, spawn_points[peer_index])
        if peer_vehicle is None:
            raise RuntimeError(f"Could not spawn peer vehicle at point {peer_index}.")
        actors.append(peer_vehicle)

        ego_vehicle.set_autopilot(True, args.tm_port)
        peer_vehicle.set_autopilot(True, args.tm_port)
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 20.0)
        traffic_manager.vehicle_percentage_speed_difference(peer_vehicle, 20.0)

        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera_pairs: list[CameraPair] = []
        vehicle_sensors: list[tuple[Any, Any, CameraPair]] = []
        for vehicle in (ego_vehicle, peer_vehicle):
            pair = CameraPair()
            rgb_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
            depth_blueprint = world.get_blueprint_library().find("sensor.camera.depth")
            for blueprint in (rgb_blueprint, depth_blueprint):
                blueprint.set_attribute("image_size_x", str(args.width))
                blueprint.set_attribute("image_size_y", str(args.height))
                blueprint.set_attribute("fov", str(args.fov))
            rgb_sensor = world.spawn_actor(
                rgb_blueprint, camera_transform, attach_to=vehicle
            )
            depth_sensor = world.spawn_actor(
                depth_blueprint, camera_transform, attach_to=vehicle
            )
            actors.extend((rgb_sensor, depth_sensor))
            rgb_sensor.listen(pair.rgb_callback)
            depth_sensor.listen(pair.depth_callback)
            camera_pairs.append(pair)
            vehicle_sensors.append((rgb_sensor, depth_sensor, pair))

        print(f"Loading detector weights: {args.weights}")
        model = YOLO(args.weights)
        latency_frames = math.ceil(args.latency_ms / (args.fixed_delta * 1000.0))
        pending_messages: deque[tuple[int, float, list[Detection], int]] = deque()
        latest_peer: tuple[float, list[Detection]] | None = None
        rng = random.Random()
        frame_count = 0

        if not args.no_display:
            cv2.namedWindow("CARLA Cooperative Perception", cv2.WINDOW_NORMAL)
            display_open = True

        while args.frames == 0 or frame_count < args.frames:
            frame_number = world.tick()
            current_sensors = [
                pair.get(frame_number, timeout=5.0) for pair in camera_pairs
            ]
            local_detections: list[list[Detection]] = []
            started = time.perf_counter()
            for index, (image, raw_depth, _) in enumerate(current_sensors):
                transform = vehicle_sensors[index][0].get_transform()
                local_detections.append(
                    detect_objects(
                        model,
                        image,
                        raw_depth,
                        transform,
                        args.fov,
                        args.confidence,
                        args.max_depth_m,
                        args.device,
                    )
                )
            inference_times.append((time.perf_counter() - started) * 1000.0)
            sim_time = current_sensors[0][2]

            if rng.random() < args.packet_loss:
                dropped_messages += 1
            else:
                message = serialize_message(
                    frame_number, sim_time, local_detections[1]
                )
                total_bytes += len(message)
                sent_messages += 1
                pending_messages.append(
                    (
                        frame_number + latency_frames,
                        sim_time,
                        local_detections[1],
                        len(message),
                    )
                )
            while pending_messages and pending_messages[0][0] <= frame_number:
                _, sent_at, detections, _ = pending_messages.popleft()
                age_ms = max(0.0, (sim_time - sent_at) * 1000.0)
                if age_ms <= args.max_message_age_ms:
                    latest_peer = (sent_at, detections)
                    received_messages += 1

            remote_detections = (
                latest_peer[1]
                if latest_peer is not None
                and (sim_time - latest_peer[0]) * 1000.0 <= args.max_message_age_ms
                else []
            )
            fused = fuse_detections(
                local_detections[0], remote_detections, args.fusion_distance_m
            )
            frame_count += 1

            if frame_count % 50 == 0:
                print(
                    f"frame={frame_number} local={len(local_detections[0])} "
                    f"peer={len(remote_detections)} fused={len(fused)} "
                    f"inference_ms={inference_times[-1]:.1f} "
                    f"sim_link_delay_ms={args.latency_ms:.0f}"
                )
            if display_open:
                view = current_sensors[0][0].copy()
                for detection in local_detections[0]:
                    if detection.box is None:
                        continue
                    x1, y1, x2, y2 = detection.box
                    label = (
                        f"{COCO_CLASSES[detection.class_id]} "
                        f"{detection.confidence:.2f}"
                    )
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        view,
                        label,
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 0),
                        1,
                    )
                status = (
                    f"local={len(local_detections[0])} peer={len(remote_detections)} "
                    f"fused={len(fused)} bytes_sent={total_bytes}"
                )
                cv2.putText(
                    view,
                    status,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("CARLA Cooperative Perception", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        if inference_times:
            print(
                "Run summary: "
                f"frames={frame_count}, "
                f"inference_ms_mean={statistics.mean(inference_times):.1f}, "
                f"inference_ms_p95={float(np.percentile(inference_times, 95)):.1f}, "
                f"message_bytes={total_bytes}, sent={sent_messages}, "
                f"received={received_messages}, simulated_drops={dropped_messages}"
            )
        print(
            "Simulation demo only: detections do not control vehicle motion and "
            "must not be used to guide a real vehicle."
        )
        return 0
    except KeyboardInterrupt:
        print("\nStopping CARLA cooperative-perception demo.")
        return 0
    finally:
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        if display_open:
            cv2.destroyAllWindows()
        traffic_manager.set_synchronous_mode(original_tm_sync)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
