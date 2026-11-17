"""
Autonomous Vehicle Highway Simulation - Closed-Loop Control System
Connects the trained AV Neural Network Controller to a live physics simulator (highway-env).

Features:
- Closed-loop AI vehicle control: Steering Angle, Throttle %, Brake %
- Point A (0 m, dead stop) -> Point B (Destination)
- Autonomous Multi-Lane Overtaking (scans adjacent lanes and changes lane when blocked by slower traffic)
- Speed-dependent Adaptive Cruise Control & Emergency Braking
- Real-time Telemetry HUD with Speedometer, Steering Gauge, Pedals, Lane Status, and Trip Progress

Usage:
    python simulate_highway_drive.py
    python simulate_highway_drive.py --destination 3500 --speed-limit 100
    python simulate_highway_drive.py --save-video
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import highway_env
import numpy as np
import pygame
import torch

from predict_control import EdgeVehicleController

# --------------------------------------------------
# 1. VISUAL CONSTANTS & COLOR PALETTE
# --------------------------------------------------
BG_COLOR = (18, 20, 24)          # Sleek dark gray
PANEL_COLOR = (26, 29, 36)       # Subtle card background
TEXT_MAIN = (240, 243, 246)      # Off-white
TEXT_MUTED = (140, 146, 160)     # Gray for subtitles
ACCENT_CYAN = (0, 216, 255)      # Cyan highlight
ACCENT_GREEN = (46, 204, 113)    # Throttle / OK green
ACCENT_RED = (231, 76, 60)       # Brake / Alert red
ACCENT_AMBER = (241, 196, 15)    # Traffic caution yellow
ACCENT_PURPLE = (155, 89, 182)   # Destination highlight

HUD_HEIGHT = 160
SIM_WIDTH = 1000
SIM_HEIGHT = 300
TOTAL_HEIGHT = SIM_HEIGHT + HUD_HEIGHT


# --------------------------------------------------
# 2. DRAWING HELPERS FOR DASHBOARD HUD
# --------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    speed_kmh: float,
    accel_mps2: float,
    yaw_rate_deg_s: float,
    steering_deg: float,
    throttle: float,
    brake: float,
    dist_traveled: float,
    dist_remain: float,
    target_dist: float,
    status_text: str,
    status_color: tuple,
    cars_detected: int,
    lead_distance: float,
    current_lane: int,
    target_lane: int,
    inference_ms: float,
    loop_hz: float,
) -> np.ndarray:
    """
    Overlays a professional, high-tech telemetry HUD bar onto the bottom of the simulation frame.
    """
    h, w, c = frame.shape
    canvas = np.zeros((h + HUD_HEIGHT, w, 3), dtype=np.uint8)
    canvas[:h, :w] = frame
    canvas[h:, :w] = BG_COLOR

    # Draw separator line between simulation and HUD
    cv2.line(canvas, (0, h), (w, h), (45, 52, 65), 2)

    # --------------------------------------------------
    # SECTION 1: SPEEDOMETER & MOTION (Left, x: 20 to 180)
    # --------------------------------------------------
    cv2.putText(canvas, "SPEED", (25, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{speed_kmh:4.1f}", (25, h + 78), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "km/h", (140, h + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)

    acc_sign = "+" if accel_mps2 >= 0 else ""
    acc_color = ACCENT_GREEN if accel_mps2 >= 0 else ACCENT_RED
    cv2.putText(canvas, f"Accel: {acc_sign}{accel_mps2:.2f} m/s²", (25, h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, acc_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Yaw: {yaw_rate_deg_s:+.1f} deg/s", (25, h + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.line(canvas, (190, h + 15), (190, h + HUD_HEIGHT - 15), (40, 44, 52), 1)

    # --------------------------------------------------
    # SECTION 2: CIRCULAR STEERING GAUGE (x: 200 to 350)
    # --------------------------------------------------
    center_x, center_y = 275, h + 72
    radius = 42

    # Draw gauge bezel
    cv2.circle(canvas, (center_x, center_y), radius, (55, 62, 75), 2, cv2.LINE_AA)
    cv2.circle(canvas, (center_x, center_y), radius - 6, (32, 36, 44), -1)

    # Dial angle ticks (-30, -15, 0, +15, +30)
    for tick_deg in [-30, -15, 0, 15, 30]:
        rad = np.deg2rad(tick_deg - 90)
        p1 = (int(center_x + (radius - 5) * np.cos(rad)), int(center_y + (radius - 5) * np.sin(rad)))
        p2 = (int(center_x + (radius - 1) * np.cos(rad)), int(center_y + (radius - 1) * np.sin(rad)))
        cv2.line(canvas, p1, p2, (90, 100, 115), 1, cv2.LINE_AA)

    # Needle angle: In visual car gauges, turning Right points needle Right (+angle).
    # Since KITTI yaw convention has Left as positive, Right is negative steering_deg.
    # We display visual needle angle = -steering_deg
    vis_steer_angle = np.clip(-steering_deg, -45.0, 45.0)
    needle_rad = np.deg2rad(vis_steer_angle - 90)
    tip_x = int(center_x + (radius - 10) * np.cos(needle_rad))
    tip_y = int(center_y + (radius - 10) * np.sin(needle_rad))

    cv2.line(canvas, (center_x, center_y), (tip_x, tip_y), ACCENT_CYAN, 3, cv2.LINE_AA)
    cv2.circle(canvas, (center_x, center_y), 4, (255, 255, 255), -1)

    # Labels for L and R
    cv2.putText(canvas, "L", (center_x - radius + 2, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "R", (center_x + radius - 9, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)

    turn_tag = " (R)" if steering_deg < -0.3 else (" (L)" if steering_deg > 0.3 else " (0)")
    cv2.putText(canvas, "AI STEERING", (center_x - 38, h + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{steering_deg:+.1f}°{turn_tag}", (center_x - 36, h + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CYAN, 1, cv2.LINE_AA)

    cv2.line(canvas, (360, h + 15), (360, h + HUD_HEIGHT - 15), (40, 44, 52), 1)

    # --------------------------------------------------
    # SECTION 3: THROTTLE & BRAKE PEDALS (x: 375 to 480)
    # --------------------------------------------------
    bar_w, bar_h = 24, 70
    pedal_base_y = h + 115

    # Throttle Bar
    t_h = int(np.clip(throttle, 0.0, 1.0) * bar_h)
    cv2.rectangle(canvas, (385, pedal_base_y - bar_h), (385 + bar_w, pedal_base_y), (40, 45, 55), -1)
    if t_h > 0:
        cv2.rectangle(canvas, (385, pedal_base_y - t_h), (385 + bar_w, pedal_base_y), ACCENT_GREEN, -1)
    cv2.rectangle(canvas, (385, pedal_base_y - bar_h), (385 + bar_w, pedal_base_y), (80, 90, 105), 1)
    cv2.putText(canvas, f"{int(throttle*100)}%", (385, pedal_base_y - bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT_GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "THR", (385, pedal_base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    # Brake Bar
    b_h = int(np.clip(brake, 0.0, 1.0) * bar_h)
    cv2.rectangle(canvas, (435, pedal_base_y - bar_h), (435 + bar_w, pedal_base_y), (40, 45, 55), -1)
    if b_h > 0:
        cv2.rectangle(canvas, (435, pedal_base_y - b_h), (435 + bar_w, pedal_base_y), ACCENT_RED, -1)
    cv2.rectangle(canvas, (435, pedal_base_y - bar_h), (435 + bar_w, pedal_base_y), (80, 90, 105), 1)
    cv2.putText(canvas, f"{int(brake*100)}%", (435, pedal_base_y - bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT_RED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "BRK", (435, pedal_base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.line(canvas, (485, h + 15), (485, h + HUD_HEIGHT - 15), (40, 44, 52), 1)

    # --------------------------------------------------
    # SECTION 4: POINT A -> POINT B TRIP PROGRESS (x: 505 to 740)
    # --------------------------------------------------
    cv2.putText(canvas, "AUTONOMOUS MISSION: POINT A -> POINT B", (505, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, TEXT_MUTED, 1, cv2.LINE_AA)

    # Mission Status Pill
    badge_x, badge_y = 505, h + 34
    badge_w, badge_h = 230, 24
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (35, 40, 50), -1)
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), status_color, 1)
    cv2.putText(canvas, status_text, (badge_x + 8, badge_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

    # Trip Progress Bar
    p_bar_x, p_bar_y = 505, h + 74
    p_bar_w, p_bar_h = 230, 12
    pct = np.clip(dist_traveled / target_dist, 0.0, 1.0)
    fill_w = int(pct * p_bar_w)

    cv2.rectangle(canvas, (p_bar_x, p_bar_y), (p_bar_x + p_bar_w, p_bar_y + p_bar_h), (40, 45, 55), -1)
    if fill_w > 0:
        cv2.rectangle(canvas, (p_bar_x, p_bar_y), (p_bar_x + fill_w, p_bar_y + p_bar_h), ACCENT_CYAN, -1)
    cv2.rectangle(canvas, (p_bar_x, p_bar_y), (p_bar_x + p_bar_w, p_bar_y + p_bar_h), (80, 90, 105), 1)

    # Waypoint markers
    cv2.putText(canvas, "Point A (0m)", (p_bar_x, p_bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Point B ({target_dist:.0f}m)", (p_bar_x + p_bar_w - 85, p_bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)

    # Lane Navigation info
    if current_lane != target_lane:
        lane_str = f"Lane {current_lane+1} -> Overtaking to Lane {target_lane+1}"
        lane_col = ACCENT_AMBER
    else:
        lane_str = f"Lane {current_lane+1} of 4 (Locked)"
        lane_col = ACCENT_CYAN
    cv2.putText(canvas, lane_str, (505, h + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lane_col, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Dist: {dist_traveled:4.1f}m ({int(pct*100)}%)  |  Rem: {dist_remain:4.1f}m", (505, h + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MAIN, 1, cv2.LINE_AA)

    cv2.line(canvas, (750, h + 15), (750, h + HUD_HEIGHT - 15), (40, 44, 52), 1)

    # --------------------------------------------------
    # SECTION 5: RADAR & EDGE AI METRICS (x: 770 to 980)
    # --------------------------------------------------
    cv2.putText(canvas, "PERCEPTION & EDGE AI", (770, h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Radar Traffic Nearby: {cars_detected} cars", (770, h + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CYAN, 1, cv2.LINE_AA)

    lead_str = f"{lead_distance:.1f} m" if lead_distance < 100.0 else "Clear (>100m)"
    lead_color = ACCENT_RED if lead_distance < 20.0 else (ACCENT_AMBER if lead_distance < 40.0 else TEXT_MAIN)
    cv2.putText(canvas, f"Lead Car Distance: {lead_str}", (770, h + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, lead_color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Model Inference: {inference_ms:.2f} ms", (770, h + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_AMBER, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Edge Loop Rate: ~{int(loop_hz):,} Hz", (770, h + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_GREEN, 1, cv2.LINE_AA)

    return canvas


# --------------------------------------------------
# 3. MAIN SIMULATION LOOP
# --------------------------------------------------

def run_simulation(args):
    print("=" * 70)
    print("   AUTONOMOUS VEHICLE HIGHWAY SIMULATION: POINT A -> POINT B")
    print("   Closed-loop deployment of AV Neural Network Controller")
    print("=" * 70)

    # Load Edge AI Controller
    controller = EdgeVehicleController()
    print("✓ Loaded AVControlModel and normalisation parameters.")

    target_distance = float(args.destination)
    cruise_speed_mps = float(args.speed_limit) / 3.6  # convert km/h to m/s

    # Dynamic computation of maximum steps to ensure car can reach destination
    if args.max_steps is not None and args.max_steps > 0:
        max_simulation_steps = args.max_steps
        step_mode_str = f"{max_simulation_steps} steps (User limit)"
    else:
        # Run continuously until Point B is reached
        max_simulation_steps = 100000  # High safety ceiling
        step_mode_str = f"Continuous until Point B ({target_distance:.0f} m)"

    # Configure Highway Environment
    max_phys_speed = max(40.0, cruise_speed_mps + 10.0)
    sim_config = {
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi / 4, np.pi / 4],
            "speed_range": [0.0, max_phys_speed],
        },
        "simulation_frequency": 15,
        "policy_frequency": 15,
        "screen_width": SIM_WIDTH,
        "screen_height": SIM_HEIGHT,
        "centering_position": [0.3, 0.5],
        "scaling": 6.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": args.no_gui,
        "vehicles_count": args.vehicles_count,
        "vehicles_density": 1.0,
    }

    env_render_mode = "rgb_array"
    env = gym.make("highway-v0", render_mode=env_render_mode, config=sim_config)
    env.reset()

    ego = env.unwrapped.vehicle
    # Start ego vehicle from dead stop at Point A
    start_x = float(ego.position[0])
    ego.speed = 0.0

    target_lane_idx = int(ego.lane_index[2])
    lane_change_cooldown = 0

    print(f"✓ Point A initialized at X = {start_x:.1f} m (Speed: 0.0 km/h, Initial Lane: {target_lane_idx+1})")
    print(f"✓ Point B Destination set at X = {start_x + target_distance:.1f} m (Travel: {target_distance:.0f} m)")
    print(f"✓ Target Cruise Speed: {args.speed_limit:.0f} km/h ({cruise_speed_mps:.1f} m/s)")
    print(f"✓ Mission Mode: {step_mode_str}")

    # Pygame display setup if GUI is active
    pygame_screen = None
    clock = None
    if not args.no_gui:
        pygame.init()
        pygame.display.set_caption("AV Autonomous Highway Simulation - Point A to Point B")
        pygame_screen = pygame.display.set_mode((SIM_WIDTH, TOTAL_HEIGHT))
        clock = pygame.time.Clock()

    # Video recorder setup
    video_writer = None
    if args.save_video:
        out_dir = Path("visualizations")
        out_dir.mkdir(exist_ok=True)
        video_path = out_dir / "autonomous_highway_drive.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 15, (SIM_WIDTH, TOTAL_HEIGHT))
        print(f"✓ Video recording enabled -> {video_path}")

    # Simulation metrics
    total_steps = 0
    mission_completed = False
    stop_counter = 0

    try:
        for step in range(max_simulation_steps):
            if not args.no_gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[User exited simulation]")
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        print("\n[Escape pressed - exiting]")
                        return

            pos_x = float(ego.position[0])
            pos_y = float(ego.position[1])
            speed = float(ego.speed)
            speed_kmh = speed * 3.6
            heading = float(ego.heading)

            dist_traveled = max(0.0, pos_x - start_x)
            dist_remain = max(0.0, target_distance - dist_traveled)

            current_lane_idx = int(ego.lane_index[2])

            # --------------------------------------------------
            # SENSOR PERCEPTION & TRAFFIC DETECTION
            # --------------------------------------------------
            # Lead vehicle in currently targeted lane
            target_lane_key = ("0", "1", target_lane_idx)
            lead_vehicle, _ = env.unwrapped.road.neighbour_vehicles(ego, target_lane_key)
            d_lead = float(ego.front_distance_to(lead_vehicle)) if lead_vehicle else 999.0
            lead_speed = float(lead_vehicle.speed) if lead_vehicle else cruise_speed_mps

            # Count nearby radar-detected vehicles within 60 meters
            nearby_cars = 0
            for v in env.unwrapped.road.vehicles:
                if v is not ego and np.linalg.norm(v.position - ego.position) < 60.0:
                    nearby_cars += 1

            # --------------------------------------------------
            # AUTONOMOUS OVERTAKING & LANE CHANGE ENGINE
            # --------------------------------------------------
            lane_change_cooldown = max(0, lane_change_cooldown - 1)
            d_safe = max(28.0, 1.3 * speed)
            d_emergency = max(14.0, 0.6 * speed)
            d_approach = max(70.0, speed * 4.0)
            is_overtaking = False

            # Keep target lane strictly within the 4 highway lanes [0, 1, 2, 3]
            target_lane_idx = int(np.clip(target_lane_idx, 0, 3))
            target_lane = env.unwrapped.road.network.get_lane(("0", "1", target_lane_idx))
            lat_offset = float(target_lane.local_coordinates(ego.position)[1])
            lane_heading = float(target_lane.heading_at(pos_x))
            heading_err = heading - lane_heading

            # Only initiate lane change if on the road and not approaching destination
            if dist_remain > d_approach and lane_change_cooldown == 0 and abs(lat_offset) < 0.6 and d_lead < d_safe and lead_speed < (cruise_speed_mps - 1.5):
                candidates = []
                if target_lane_idx > 0:
                    candidates.append(target_lane_idx - 1)  # Prefer left overtaking lane
                if target_lane_idx < 3:
                    candidates.append(target_lane_idx + 1)  # Next option right lane

                for cand_lane in candidates:
                    cand_lead, cand_rear = env.unwrapped.road.neighbour_vehicles(ego, ("0", "1", cand_lane))
                    df = float(ego.front_distance_to(cand_lead)) if cand_lead else 999.0
                    dr = float(cand_rear.front_distance_to(ego)) if cand_rear else 999.0

                    if df > max(35.0, 1.2 * speed) and dr > 20.0:
                        print(f"   [Step {step:03d}] 🔄 Slower vehicle at {d_lead:.1f} m ({lead_speed*3.6:.0f} km/h)! Overtaking: Lane {target_lane_idx+1} -> Lane {cand_lane+1}")
                        target_lane_idx = cand_lane
                        lane_change_cooldown = 45  # 3 seconds cooldown before next change
                        is_overtaking = True
                        # Re-calculate target lane geometry
                        target_lane = env.unwrapped.road.network.get_lane(("0", "1", target_lane_idx))
                        lat_offset = float(target_lane.local_coordinates(ego.position)[1])
                        lane_heading = float(target_lane.heading_at(pos_x))
                        heading_err = heading - lane_heading
                        break

            # --------------------------------------------------
            # LANE CENTERING TO TARGET LANE
            # --------------------------------------------------
            # Firm, reliable lane centering - always keeps car locked to lane
            req_curv = np.clip(0.05 * lat_offset + 0.35 * np.sin(heading_err), -0.04, 0.04)
            req_yaw_rate = np.clip(speed * req_curv, -0.40, 0.40)
            yaw_rate_deg_s = np.rad2deg(req_yaw_rate)

            # --------------------------------------------------
            # AUTONOMOUS MISSION BEHAVIOR LOGIC
            # --------------------------------------------------
            if ego.crashed:
                target_speed = 0.0
                target_accel = -5.0
                status_text = "[XX] COLLISION DETECTED - STOPPED"
                status_color = ACCENT_RED
            elif dist_remain <= 0.5 or (dist_remain < 15.0 and speed < 1.0):
                # Phase 4: Arrived at Point B
                target_speed = 0.0
                target_accel = -4.5 if speed > 0.3 else 0.0
                status_text = "[OK] POINT B REACHED (STOPPED)"
                status_color = ACCENT_PURPLE
                if speed <= 0.3:
                    stop_counter += 1
                    if stop_counter >= 30:  # Hold stop for 2 seconds
                        mission_completed = True
            elif dist_remain < d_approach:
                # Phase 3: Approach Destination (smooth deceleration)
                target_speed = max(0.5, cruise_speed_mps * (dist_remain / d_approach))
                target_accel = np.clip(1.5 * (target_speed - speed), -4.0, 1.5)
                status_text = "[**] APPROACHING DESTINATION"
                status_color = ACCENT_PURPLE
            elif d_lead < d_emergency:
                # Emergency traffic braking
                target_accel = -5.0
                status_text = "[!!] EMERGENCY TRAFFIC BRAKE"
                status_color = ACCENT_RED
            elif abs(lat_offset) > 0.8:
                # Actively executing lane change
                target_accel = np.clip(1.0 * (cruise_speed_mps - speed), -2.5, 1.8)
                status_text = f"[>>] OVERTAKING -> LANE {target_lane_idx+1}"
                status_color = ACCENT_AMBER
            elif d_lead < d_safe and lead_speed < speed:
                # Slower car ahead, lane change blocked -> match lead speed safely
                target_accel = np.clip(0.4 * (d_lead - d_safe) + 1.2 * (lead_speed - speed), -4.0, 0.5)
                status_text = "[!] ADAPTIVE TRAFFIC BRAKE"
                status_color = ACCENT_AMBER
            elif dist_traveled < 20.0 and speed < 8.0:
                # Phase 1: Accelerating from rest
                target_accel = 2.0
                status_text = "[>>] ACCELERATING (POINT A)"
                status_color = ACCENT_GREEN
            else:
                # Phase 2: Autonomous cruise
                target_accel = np.clip(1.0 * (cruise_speed_mps - speed), -3.0, 2.0)
                status_text = "[==] CRUISING AUTONOMOUSLY"
                status_color = ACCENT_CYAN

            # Prevent reverse acceleration when stopped
            if speed <= 0.1 and target_accel < 0:
                target_accel = 0.0

            # --------------------------------------------------
            # EDGE AI INFERENCE
            # --------------------------------------------------
            sensor_dict = {
                "speed_kmh": speed_kmh,
                "forward_speed_mps": speed,
                "heading_deg": 86.0 + np.rad2deg(heading_err),
                "yaw_rate_rad_s": req_yaw_rate,
                "forward_acceleration_mps2": target_accel,
                "lateral_acceleration_mps2": speed * req_yaw_rate,
                "curvature_m_inv": req_curv,
                "persons_detected": 0,
                "bicycles_detected": 0,
                "cars_detected": nearby_cars,
                "trucks_detected": 0,
                "buses_detected": 0,
                "traffic_lights_detected": 0,
                "total_detections": nearby_cars,
            }

            t0 = time.perf_counter()
            preds = controller.predict(sensor_dict)
            t_infer = (time.perf_counter() - t0) * 1000.0  # ms
            loop_hz = 1000.0 / max(t_infer, 1e-4)

            ai_steer_deg = preds["steering_angle_deg"]
            ai_thr = preds["throttle_command"]
            ai_brk = preds["brake_command"]

            # --------------------------------------------------
            # SIMULATOR ACTUATION (CONTINUOUS ACTION)
            # --------------------------------------------------
            if mission_completed or (dist_remain <= 0.1 and speed <= 0.3) or ego.crashed:
                sim_accel = -5.0 if speed > 0.1 else 0.0
            elif ai_brk > 0.05:
                # Do not apply reverse acceleration when stopped
                sim_accel = -ai_brk * 5.0 if speed > 0.1 else 0.0
            else:
                sim_accel = ai_thr * 3.0

            action_accel = np.clip(sim_accel / 5.0, -1.0, 1.0)
            # Map KITTI CCW steering (+left, -right) to highway-env CW steering (-left, +right)
            action_steer = np.clip(-np.deg2rad(ai_steer_deg) / (np.pi / 4), -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(np.array([action_accel, action_steer], dtype=np.float32))
            ego.speed = max(0.0, float(ego.speed))
            total_steps += 1

            # --------------------------------------------------
            # RENDER FRAME & HUD DASHBOARD
            # --------------------------------------------------
            raw_frame = env.render()
            hud_frame = draw_hud(
                frame=raw_frame,
                speed_kmh=speed_kmh,
                accel_mps2=sim_accel,
                yaw_rate_deg_s=yaw_rate_deg_s,
                steering_deg=ai_steer_deg,
                throttle=ai_thr,
                brake=ai_brk,
                dist_traveled=dist_traveled,
                dist_remain=dist_remain,
                target_dist=target_distance,
                status_text=status_text,
                status_color=status_color,
                cars_detected=nearby_cars,
                lead_distance=d_lead,
                current_lane=current_lane_idx,
                target_lane=target_lane_idx,
                inference_ms=t_infer,
                loop_hz=loop_hz,
            )

            # Record video frame
            if video_writer:
                bgr_frame = cv2.cvtColor(hud_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            # Display on Pygame screen
            if pygame_screen:
                surf = pygame.surfarray.make_surface(hud_frame.swapaxes(0, 1))
                pygame_screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(15)  # 15 FPS simulation rate

            # Periodic terminal progress
            if step % 25 == 0 or mission_completed or ego.crashed:
                print(f"Step {step:04d} | Dist: {dist_traveled:6.1f}m/{target_distance:.0f}m | Lane: {current_lane_idx+1}/4 | Spd: {speed_kmh:5.1f} km/h | Acc: {sim_accel:+4.2f} | Steer: {ai_steer_deg:+5.1f}° | Thr: {ai_thr:.2f} | Brk: {ai_brk:.2f} | {status_text}")

            if ego.crashed:
                print(f"\n=======================================================")
                print(f"CRASH DETECTED: Vehicle collided at {dist_traveled:.1f} m!")
                print(f"=======================================================\n")
                if not args.no_gui:
                    pygame.time.wait(2500)
                break

            if mission_completed:
                print(f"\n=======================================================")
                print(f"MISSION SUCCESS: Reached Point B at {dist_traveled:.1f} m!")
                print(f"Vehicle brought to a safe, complete stop by AI brain.")
                print(f"=======================================================\n")
                if not args.no_gui:
                    pygame.time.wait(2000)
                break

    finally:
        if video_writer:
            video_writer.release()
            print(f"✓ Video saved successfully to: visualizations/autonomous_highway_drive.mp4")
        if not args.no_gui and pygame_screen:
            pygame.quit()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Highway Drive (Point A to Point B)")
    parser.add_argument("--destination", type=float, default=350.0, help="Target trip distance in meters (default: 350.0)")
    parser.add_argument("--speed-limit", type=float, default=65.0, help="Target cruise speed limit in km/h (default: 65.0)")
    parser.add_argument("--save-video", action="store_true", help="Record and save an MP4 video of the simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run simulation headless without opening a Pygame GUI window")
    parser.add_argument("--max-steps", type=int, default=None, help="Max simulation steps before termination (default: auto-computed from destination)")
    parser.add_argument("--vehicles-count", type=int, default=25, help="Number of traffic vehicles on highway (default: 25)")
    args = parser.parse_args()

    run_simulation(args)
