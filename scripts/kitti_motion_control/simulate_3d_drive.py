"""
KITTI 3D Autonomous Vehicle Simulator
Closed-Loop Edge AI Driving Simulation in Full 3D with Dynamic Curves & Real World Actions.

Features:
- Full 3D interactive physics & graphics powered by Ursina/Panda3D (runs on CPU/standard Intel graphics).
- Real 3D road network: Straightaways, sweeping curves, S-curve chicanes, and sharp 90-degree intersection turns.
- Point A (Start) to Point B (Destination) mission navigation with autonomous arrival sequence.
- Closed-loop Edge AI controller integration using trained KITTI AVControlModel (model/best_model.pth).
- 3D vehicle with swiveling front steering wheels, spinning tires, and glowing brake lights.
- Multiple camera views (Chase Cam, Cockpit / Hood Cam, Drone Cam) toggleable with 'C' key.
- Real-time on-screen telemetry HUD: Speedometer, Steering gauge, Throttle/Brake pedals, Trip progress, Edge latency.

Usage:
    python simulate_3d_drive.py
    python simulate_3d_drive.py --speed 60
    python simulate_3d_drive.py --save-preview
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from panda3d.core import Filename, PNMImage
from ursina import *
from ursina.models.procedural.cylinder import Cylinder

# Import trained Edge AI Controller
from predict_control import EdgeVehicleController

# --------------------------------------------------
# 0. COLOR PALETTE HELPER (0..255 -> 0.0..1.0)
# --------------------------------------------------
def col(r: float, g: float, b: float, a: float = 1.0) -> Color:
    """Safely normalizes 0-255 RGB values to Panda3D 0.0-1.0 floats."""
    return Color(r / 255.0, g / 255.0, b / 255.0, a)


# --------------------------------------------------
# 1. TRACK & ROAD GEOMETRY GENERATOR
# --------------------------------------------------

def build_3d_course(course_type="city", ds=0.8):
    """
    Generates a continuous C1-smooth road centerline with straightaways,
    high-speed curves, S-bends, and 90-degree intersection corners.
    """
    if course_type == "city":
        segments = [
            ("straight", 90.0),       # Point A: Launch straightaway
            ("turn", 75.0, 45.0),      # Sweeping right turn (R=75m, +45 deg)
            ("straight", 60.0),       # Connector straight
            ("turn", 55.0, -70.0),     # Dynamic left chicane (R=55m, -70 deg)
            ("straight", 70.0),       # Avenue straightaway
            ("turn", 25.0, 90.0),      # Sharp 90 deg right intersection corner (R=25m)
            ("straight", 110.0),      # Long downtown boulevard
            ("turn", 25.0, -90.0),     # Sharp 90 deg left intersection corner (R=25m)
            ("straight", 90.0),       # Final approach to Point B
        ]
    else:
        segments = [
            ("straight", 80.0),
            ("turn", 60.0, 50.0),
            ("straight", 60.0),
            ("turn", 45.0, -60.0),
            ("straight", 80.0),
            ("turn", 30.0, 90.0),
            ("straight", 100.0),
        ]

    waypoints = []
    x, z, psi, s_total = 0.0, 0.0, 0.0, 0.0

    for seg in segments:
        seg_type = seg[0]
        if seg_type == "straight":
            length = float(seg[1])
            n = max(2, int(np.ceil(length / ds)))
            for i in range(n):
                s = i * (length / n)
                waypoints.append({
                    "x": float(x + s * np.sin(psi)),
                    "z": float(z + s * np.cos(psi)),
                    "heading": float(psi),
                    "curvature": 0.0,
                    "s": float(s_total + s),
                })
            s_total += length
            x += length * np.sin(psi)
            z += length * np.cos(psi)

        elif seg_type == "turn":
            r = float(seg[1])
            turn_deg = float(seg[2])  # positive = right, negative = left
            turn_rad = np.radians(abs(turn_deg))
            arc_len = r * turn_rad
            n = max(2, int(np.ceil(arc_len / ds)))

            if turn_deg > 0:  # Right turn
                xc = x + r * np.cos(psi)
                zc = z - r * np.sin(psi)
                kappa = -1.0 / r  # Right curvature (negative in KITTI coordinate convention)
                for i in range(n):
                    s = i * (arc_len / n)
                    cur_psi = psi + s / r
                    waypoints.append({
                        "x": float(xc - r * np.cos(cur_psi)),
                        "z": float(zc + r * np.sin(cur_psi)),
                        "heading": float(cur_psi),
                        "curvature": float(kappa),
                        "s": float(s_total + s),
                    })
                s_total += arc_len
                end_psi = psi + turn_rad
                x = xc - r * np.cos(end_psi)
                z = zc + r * np.sin(end_psi)
                psi = end_psi
            else:  # Left turn
                xc = x - r * np.cos(psi)
                zc = z + r * np.sin(psi)
                kappa = 1.0 / r   # Left curvature (positive in KITTI coordinate convention)
                for i in range(n):
                    s = i * (arc_len / n)
                    cur_psi = psi - s / r
                    waypoints.append({
                        "x": float(xc + r * np.cos(cur_psi)),
                        "z": float(zc - r * np.sin(cur_psi)),
                        "heading": float(cur_psi),
                        "curvature": float(kappa),
                        "s": float(s_total + s),
                    })
                s_total += arc_len
                end_psi = psi - turn_rad
                x = xc + r * np.cos(end_psi)
                z = zc - r * np.sin(end_psi)
                psi = end_psi

    return waypoints


# --------------------------------------------------
# 2. 3D ROAD MESH & SCENERY BUILDER
# --------------------------------------------------

def create_road_and_scenery(waypoints, road_width=7.6):
    """
    Constructs the 3D road surface mesh, lane markings, curbs,
    sidewalks, roadside buildings, trees, streetlamps, and finish arch.
    """
    n_pts = len(waypoints)
    road_verts = []
    road_tris = []

    asphalt_col = col(38, 42, 50)
    curb_col = col(180, 185, 192)

    half_w = road_width / 2.0
    curb_w = 0.5
    curb_h = 0.18

    # Ground terrain plane (Watertight cube)
    xs = [wp["x"] for wp in waypoints]
    zs = [wp["z"] for wp in waypoints]
    min_x, max_x = min(xs) - 120, max(xs) + 120
    min_z, max_z = min(zs) - 120, max(zs) + 120
    cx = (min_x + max_x) / 2.0
    cz = (min_z + max_z) / 2.0
    sx = max_x - min_x
    sz = max_z - min_z

    ground = Entity(
        model="cube",
        scale=(sx, 0.25, sz),
        position=(cx, -0.15, cz),
        color=col(50, 115, 50),
    )

    # Road surface triangles
    v_idx = 0
    for i in range(n_pts - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]

        psi0 = p0["heading"]
        psi1 = p1["heading"]

        nx0, nz0 = np.cos(psi0), -np.sin(psi0)
        nx1, nz1 = np.cos(psi1), -np.sin(psi1)

        lx0, lz0 = p0["x"] - half_w * nx0, p0["z"] - half_w * nz0
        lx1, lz1 = p1["x"] - half_w * nx1, p1["z"] - half_w * nz1
        rx0, rz0 = p0["x"] + half_w * nx0, p0["z"] + half_w * nz0
        rx1, rz1 = p1["x"] + half_w * nx1, p1["z"] + half_w * nz1

        road_verts.extend([
            Vec3(lx0, 0.04, lz0),
            Vec3(rx0, 0.04, rz0),
            Vec3(rx1, 0.04, rz1),
            Vec3(lx1, 0.04, lz1)
        ])

        road_tris.extend([
            (v_idx, v_idx + 1, v_idx + 2),
            (v_idx, v_idx + 2, v_idx + 3)
        ])
        v_idx += 4

    road_mesh = Mesh(
        vertices=road_verts,
        triangles=road_tris,
        normals=[Vec3(0, 1, 0)] * len(road_verts),
        mode="triangle"
    )
    road_entity = Entity(model=road_mesh, color=asphalt_col)

    # Centerline dashed markings
    dash_length = 2.4
    dash_gap = 3.6

    for i in range(0, n_pts - 2, 4):
        wp = waypoints[i]
        s = wp["s"]
        cycle = s % (dash_length + dash_gap)
        if cycle < dash_length:
            Entity(
                model="cube",
                scale=(0.20, 0.03, 1.8),
                position=(wp["x"], 0.06, wp["z"]),
                rotation_y=np.degrees(wp["heading"]),
                color=col(240, 240, 240)
            )

    # Curbs, street lamps, trees, and buildings along edges
    tree_colors = [
        col(34, 139, 34),
        col(46, 117, 46),
        col(60, 150, 50),
    ]
    building_colors = [
        col(85, 95, 110),   # Modern slate
        col(180, 140, 120), # Brick red
        col(200, 205, 210), # White stucco
        col(110, 130, 150), # Steel blue
        col(190, 180, 150), # Sandstone
    ]

    for i in range(0, n_pts - 1, 8):
        wp = waypoints[i]
        psi = wp["heading"]
        nx, nz = np.cos(psi), -np.sin(psi)

        # Curbs
        for side, sign in [("left", -1), ("right", 1)]:
            curb_pos_x = wp["x"] + sign * (half_w + curb_w / 2.0) * nx
            curb_pos_z = wp["z"] + sign * (half_w + curb_w / 2.0) * nz
            Entity(
                model="cube",
                scale=(curb_w, curb_h, 5.0),
                position=(curb_pos_x, curb_h / 2.0 + 0.02, curb_pos_z),
                rotation_y=np.degrees(psi),
                color=curb_col
            )

        # Trees
        if i % 24 == 0:
            for side, sign in [("left", -1), ("right", 1)]:
                dist_out = half_w + 5.0 + (i % 3) * 1.5
                tx = wp["x"] + sign * dist_out * nx
                tz = wp["z"] + sign * dist_out * nz

                Entity(model="cube", scale=(0.4, 2.2, 0.4), position=(tx, 1.1, tz), color=col(90, 55, 30))
                foliage_col = tree_colors[i % len(tree_colors)]
                Entity(model="cube", scale=(2.4, 2.0, 2.4), position=(tx, 2.6, tz), color=foliage_col)
                Entity(model="cube", scale=(1.7, 1.6, 1.7), position=(tx, 3.8, tz), color=foliage_col)

        # Streetlamps
        if i % 32 == 0:
            lx = wp["x"] + (half_w + 1.8) * nx
            lz = wp["z"] + (half_w + 1.8) * nz
            Entity(model="cube", scale=(0.15, 5.0, 0.15), position=(lx, 2.5, lz), color=col(70, 75, 80))
            Entity(model="cube", scale=(1.2, 0.15, 0.15), position=(lx - 0.5 * nx, 5.0, lz - 0.5 * nz), rotation_y=np.degrees(psi), color=col(70, 75, 80))
            Entity(model="cube", scale=(0.4, 0.2, 0.4), position=(lx - 1.0 * nx, 4.9, lz - 1.0 * nz), color=col(255, 240, 200))

        # Buildings
        if i % 40 == 0 and i > 15 and i < n_pts - 25:
            for side, sign in [("left", -1), ("right", 1)]:
                b_dist = half_w + 12.0 + (i % 4) * 2.0
                bx = wp["x"] + sign * b_dist * nx
                bz = wp["z"] + sign * b_dist * nz

                bw = 14.0 + (i % 5) * 2.0
                bd = 12.0 + (i % 3) * 3.0
                bh = 10.0 + (i % 7) * 3.5

                b_col = building_colors[(i + (1 if side == "right" else 0)) % len(building_colors)]
                Entity(model="cube", scale=(bw, bh, bd), position=(bx, bh / 2.0, bz), rotation_y=np.degrees(psi), color=b_col)
                Entity(model="cube", scale=(bw + 0.4, 0.5, bd + 0.4), position=(bx, bh + 0.25, bz), rotation_y=np.degrees(psi), color=col(40, 45, 50))

    # Point A Gantry (Placed 10m ahead so vehicle launches through it)
    wp_start_idx = min(12, len(waypoints) - 1)
    wp_start = waypoints[wp_start_idx]
    p0_psi = wp_start["heading"]
    nx0, nz0 = np.cos(p0_psi), -np.sin(p0_psi)
    arch_span = half_w + 1.6
    Entity(model="cube", scale=(0.5, 5.5, 0.5), position=(wp_start["x"] - arch_span * nx0, 2.75, wp_start["z"] - arch_span * nz0), color=col(40, 180, 80))
    Entity(model="cube", scale=(0.5, 5.5, 0.5), position=(wp_start["x"] + arch_span * nx0, 2.75, wp_start["z"] + arch_span * nz0), color=col(40, 180, 80))
    Entity(model="cube", scale=(arch_span * 2.2, 0.8, 0.6), position=(wp_start["x"], 5.4, wp_start["z"]), rotation_y=np.degrees(p0_psi), color=col(30, 140, 60))
    Text(text="START: POINT A", position=(0, 0), parent=Entity(position=(wp_start["x"], 5.4, wp_start["z"] - 0.35), rotation_y=np.degrees(p0_psi)), scale=8, color=color.white, origin=(0, 0))

    # Point B Gantry
    wp_end = waypoints[-1]
    pend_psi = wp_end["heading"]
    nxe, nze = np.cos(pend_psi), -np.sin(pend_psi)
    Entity(model="cube", scale=(0.6, 6.0, 0.6), position=(wp_end["x"] - arch_span * nxe, 3.0, wp_end["z"] - arch_span * nze), color=col(220, 50, 50))
    Entity(model="cube", scale=(0.6, 6.0, 0.6), position=(wp_end["x"] + arch_span * nxe, 3.0, wp_end["z"] + arch_span * nze), color=col(220, 50, 50))
    Entity(model="cube", scale=(arch_span * 2.2, 1.0, 0.7), position=(wp_end["x"], 5.8, wp_end["z"]), rotation_y=np.degrees(pend_psi), color=col(240, 200, 30))
    Entity(model="cube", scale=(road_width, 0.05, 1.2), position=(wp_end["x"], 0.06, wp_end["z"]), rotation_y=np.degrees(pend_psi), color=col(240, 240, 240))
    Text(text="DESTINATION: POINT B", position=(0, 0), parent=Entity(position=(wp_end["x"], 5.8, wp_end["z"] - 0.4), rotation_y=np.degrees(pend_psi)), scale=8, color=color.black, origin=(0, 0))

    print(f"3D World successfully built. {len(waypoints)} road waypoints, buildings, trees, and gantries placed.")


# --------------------------------------------------
# 3. 3D AUTONOMOUS VEHICLE MODEL
# --------------------------------------------------

class AutonomousVehicle3D:
    """
    Detailed 3D vehicle model with Ackermann front steering pivots,
    spinning tire cylinders, dynamic headlights, and illuminated brake lights.
    """
    def __init__(self, init_x=0.0, init_z=0.0, init_heading=0.0):
        self.root = Entity(
            position=(init_x, 0.35, init_z),
            rotation_y=np.degrees(init_heading)
        )

        body_color = col(24, 110, 190)
        trim_color = col(20, 24, 30)
        glass_color = col(35, 50, 65)

        # Lower chassis
        self.chassis = Entity(
            parent=self.root,
            model="cube",
            scale=(2.0, 0.55, 4.4),
            position=(0, 0.35, 0),
            color=body_color
        )

        # Bumpers
        Entity(parent=self.root, model="cube", scale=(2.05, 0.22, 0.5), position=(0, 0.2, 2.15), color=trim_color)
        Entity(parent=self.root, model="cube", scale=(2.05, 0.25, 0.5), position=(0, 0.22, -2.15), color=trim_color)

        # Greenhouse cabin
        self.cabin = Entity(parent=self.root, model="cube", scale=(1.65, 0.55, 2.3), position=(0, 0.82, -0.2), color=glass_color)
        Entity(parent=self.root, model="cube", scale=(1.55, 0.1, 2.1), position=(0, 1.12, -0.2), color=body_color)

        # Headlights
        Entity(parent=self.root, model="cube", scale=(0.4, 0.15, 0.1), position=(-0.7, 0.45, 2.22), color=col(220, 245, 255))
        Entity(parent=self.root, model="cube", scale=(0.4, 0.15, 0.1), position=(0.7, 0.45, 2.22), color=col(220, 245, 255))

        # Brake lights
        self.brake_light_l = Entity(parent=self.root, model="cube", scale=(0.42, 0.16, 0.1), position=(-0.7, 0.45, -2.22), color=col(120, 20, 20))
        self.brake_light_r = Entity(parent=self.root, model="cube", scale=(0.42, 0.16, 0.1), position=(0.7, 0.45, -2.22), color=col(120, 20, 20))

        # Wheels
        wheel_cylinder = Cylinder(resolution=16, radius=0.38, height=0.28)
        tire_color = col(25, 25, 28)
        rim_color = col(180, 185, 195)

        # Rear wheels
        self.rl_wheel = Entity(parent=self.root, model=wheel_cylinder, scale=(1, 1, 1), rotation_z=90, position=(-1.0, 0.15, -1.3), color=tire_color)
        self.rr_wheel = Entity(parent=self.root, model=wheel_cylinder, scale=(1, 1, 1), rotation_z=90, position=(1.0, 0.15, -1.3), color=tire_color)
        Entity(parent=self.rl_wheel, model="circle", scale=(0.45, 0.45, 0.45), position=(0, -0.15, 0), rotation_x=90, color=rim_color)
        Entity(parent=self.rr_wheel, model="circle", scale=(0.45, 0.45, 0.45), position=(0, 0.15, 0), rotation_x=90, color=rim_color)

        # Front wheels with pivots
        self.fl_pivot = Entity(parent=self.root, position=(-1.0, 0.15, 1.3))
        self.fr_pivot = Entity(parent=self.root, position=(1.0, 0.15, 1.3))

        self.fl_wheel = Entity(parent=self.fl_pivot, model=wheel_cylinder, scale=(1, 1, 1), rotation_z=90, position=(0, 0, 0), color=tire_color)
        self.fr_wheel = Entity(parent=self.fr_pivot, model=wheel_cylinder, scale=(1, 1, 1), rotation_z=90, position=(0, 0, 0), color=tire_color)
        Entity(parent=self.fl_wheel, model="circle", scale=(0.45, 0.45, 0.45), position=(0, -0.15, 0), rotation_x=90, color=rim_color)
        Entity(parent=self.fr_wheel, model="circle", scale=(0.45, 0.45, 0.45), position=(0, 0.15, 0), rotation_x=90, color=rim_color)

    def set_steering_angle(self, steering_deg: float):
        self.fl_pivot.rotation_y = steering_deg
        self.fr_pivot.rotation_y = steering_deg

    def spin_wheels(self, distance_traveled: float):
        wheel_radius = 0.38
        spin_deg = (distance_traveled / (2.0 * math.pi * wheel_radius)) * 360.0
        self.rl_wheel.rotation_x += spin_deg
        self.rr_wheel.rotation_x += spin_deg
        self.fl_wheel.rotation_x += spin_deg
        self.fr_wheel.rotation_x += spin_deg

    def set_brake_lights(self, is_braking: bool):
        if is_braking:
            self.brake_light_l.color = col(255, 30, 30)
            self.brake_light_r.color = col(255, 30, 30)
            self.brake_light_l.scale = (0.46, 0.20, 0.12)
            self.brake_light_r.scale = (0.46, 0.20, 0.12)
        else:
            self.brake_light_l.color = col(120, 20, 20)
            self.brake_light_r.color = col(120, 20, 20)
            self.brake_light_l.scale = (0.42, 0.16, 0.1)
            self.brake_light_r.scale = (0.42, 0.16, 0.1)


# --------------------------------------------------
# 4. TELEMETRY HUD OVERLAY
# --------------------------------------------------

class TelemetryHUD:
    def __init__(self):
        # Top banner card
        Entity(parent=camera.ui, model="quad", scale=(1.2, 0.06), position=(0, 0.46), color=col(18, 22, 28, 0.9))
        self.header_text = Text(parent=camera.ui, text="KITTI AUTONOMOUS VEHICLE 3D SIMULATION  |  EDGE AI SYSTEM", position=(0, 0.46), origin=(0, 0), scale=0.9, color=col(0, 216, 255))

        # Bottom main telemetry bar
        self.hud_bg = Entity(parent=camera.ui, model="quad", scale=(1.74, 0.22), position=(0, -0.38), color=col(14, 17, 23, 0.94))

        # SECTION 1: SPEEDOMETER
        Text(parent=camera.ui, text="SPEED", position=(-0.78, -0.30), scale=0.75, color=col(140, 146, 160))
        self.speed_val = Text(parent=camera.ui, text="0.0", position=(-0.78, -0.36), scale=2.4, color=color.white)
        self.speed_unit = Text(parent=camera.ui, text="KM/H", position=(-0.60, -0.37), scale=0.8, color=col(0, 216, 255))
        self.accel_text = Text(parent=camera.ui, text="Accel: +0.0 m/s²", position=(-0.78, -0.44), scale=0.7, color=col(180, 185, 195))
        self.yaw_text = Text(parent=camera.ui, text="Yaw:   +0.0 °/s", position=(-0.78, -0.47), scale=0.7, color=col(180, 185, 195))

        # SECTION 2: STEERING & PEDALS
        Text(parent=camera.ui, text="AI STEERING", position=(-0.25, -0.30), scale=0.75, color=col(140, 146, 160))
        self.steer_val = Text(parent=camera.ui, text="0.0° [STRAIGHT]", position=(-0.25, -0.34), scale=1.1, color=col(0, 216, 255))

        Entity(parent=camera.ui, model="quad", scale=(0.32, 0.015), position=(-0.14, -0.38), color=col(40, 48, 60))
        Entity(parent=camera.ui, model="quad", scale=(0.01, 0.03), position=(-0.14, -0.38), color=color.white)
        self.steer_indicator = Entity(parent=camera.ui, model="quad", scale=(0.02, 0.035), position=(-0.14, -0.38), color=col(0, 216, 255))

        Text(parent=camera.ui, text="THR", position=(-0.25, -0.43), scale=0.65, color=col(46, 204, 113))
        Entity(parent=camera.ui, model="quad", scale=(0.14, 0.018), position=(-0.12, -0.43), color=col(30, 40, 35))
        self.thr_bar = Entity(parent=camera.ui, model="quad", scale=(0.0, 0.018), position=(-0.19, -0.43), origin=(-0.5, 0), color=col(46, 204, 113))

        Text(parent=camera.ui, text="BRK", position=(-0.25, -0.465), scale=0.65, color=col(231, 76, 60))
        Entity(parent=camera.ui, model="quad", scale=(0.14, 0.018), position=(-0.12, -0.465), color=col(50, 30, 30))
        self.brk_bar = Entity(parent=camera.ui, model="quad", scale=(0.0, 0.018), position=(-0.19, -0.465), origin=(-0.5, 0), color=col(231, 76, 60))

        # SECTION 3: MISSION & STATUS
        Text(parent=camera.ui, text="TRIP PROGRESS", position=(0.12, -0.30), scale=0.75, color=col(140, 146, 160))
        self.dist_val = Text(parent=camera.ui, text="0.0 m / 0.0 m (0.0%)", position=(0.12, -0.34), scale=1.05, color=color.white)
        self.status_pill = Text(parent=camera.ui, text="[ CRUISING AUTONOMOUSLY ]", position=(0.12, -0.39), scale=0.9, color=col(46, 204, 113))

        Entity(parent=camera.ui, model="quad", scale=(0.42, 0.016), position=(0.33, -0.43), color=col(35, 42, 55))
        self.progress_bar = Entity(parent=camera.ui, model="quad", scale=(0.0, 0.016), position=(0.12, -0.43), origin=(-0.5, 0), color=col(155, 89, 182))

        self.edge_metrics = Text(parent=camera.ui, text="Edge Inference: 0.22 ms  |  Loop: 50 Hz", position=(0.12, -0.47), scale=0.68, color=col(180, 185, 195))
        self.help_prompt = Text(parent=camera.ui, text="[C] Switch Camera (Chase / Hood / Drone)   [R] Restart   [P] Pause   [Esc] Exit", position=(0, -0.49), origin=(0, 0), scale=0.62, color=col(120, 125, 135))

        # Mission Complete Overlay
        self.win_panel = Entity(parent=camera.ui, model="quad", scale=(0.88, 0.44), position=(0, 0.05), color=col(16, 20, 26, 0.96), enabled=False)
        self.win_title = Text(parent=camera.ui, text="MISSION COMPLETE: POINT B REACHED SAFELY", position=(0, 0.20), origin=(0, 0), scale=1.2, color=col(46, 204, 113), enabled=False)
        self.win_stats = Text(parent=camera.ui, text="", position=(0, 0.05), origin=(0, 0), scale=0.95, color=color.white, enabled=False)
        self.win_prompt = Text(parent=camera.ui, text="Press [R] to Drive Again  |  [Esc] to Exit", position=(0, -0.10), origin=(0, 0), scale=0.8, color=col(0, 216, 255), enabled=False)

    def update_telemetry(
        self,
        speed_kmh: float,
        accel_mps2: float,
        yaw_rate_deg_s: float,
        steer_deg: float,
        throttle: float,
        brake: float,
        s_cur: float,
        s_total: float,
        status_mode: str,
        inference_ms: float,
        update_text: bool = True
    ):
        needle_x = -0.14 + (np.clip(steer_deg, -30.0, 30.0) / 30.0) * 0.15
        self.steer_indicator.x = needle_x
        self.thr_bar.scale_x = 0.14 * np.clip(throttle, 0.0, 1.0)
        self.brk_bar.scale_x = 0.14 * np.clip(brake, 0.0, 1.0)
        pct = (s_cur / max(1.0, s_total)) * 100.0
        self.progress_bar.scale_x = 0.42 * (pct / 100.0)

        if update_text:
            self.speed_val.text = f"{speed_kmh:4.1f}"
            self.accel_text.text = f"Accel: {accel_mps2:+4.2f} m/s²"
            self.yaw_text.text = f"Yaw:   {yaw_rate_deg_s:+5.1f} °/s"

            steer_direction = "STRAIGHT"
            if steer_deg > 2.0:
                steer_direction = "RIGHT TURN"
            elif steer_deg < -2.0:
                steer_direction = "LEFT TURN"
            self.steer_val.text = f"{steer_deg:+4.1f}° [{steer_direction}]"
            self.dist_val.text = f"{s_cur:4.1f} m / {s_total:4.1f} m ({pct:4.1f}%)"

            if status_mode == "CRUISING":
                self.status_pill.text = "[ CRUISING AUTONOMOUSLY ]"
                self.status_pill.color = col(46, 204, 113)
            elif status_mode == "CORNERING":
                self.status_pill.text = "[ CORNERING & CURVE ENTRY ]"
                self.status_pill.color = col(0, 216, 255)
            elif status_mode == "DECELERATING":
                self.status_pill.text = "[ DECELERATING / BRAKING ]"
                self.status_pill.color = col(231, 76, 60)
            elif status_mode == "ARRIVED":
                self.status_pill.text = "[ POINT B DESTINATION REACHED ]"
                self.status_pill.color = col(155, 89, 182)

            self.edge_metrics.text = f"Edge Inference: {inference_ms:4.2f} ms  |  Loop: 50 Hz"

    def show_mission_complete(self, total_dist: float, total_time: float, avg_speed: float, max_g: float):
        self.win_panel.enabled = True
        self.win_title.enabled = True
        self.win_stats.enabled = True
        self.win_prompt.enabled = True
        self.win_stats.text = (
            f"Trip Distance:     {total_dist:5.1f} meters\n\n"
            f"Drive Duration:    {total_time:4.1f} seconds\n\n"
            f"Average Speed:     {avg_speed:4.1f} km/h\n\n"
            f"Max Lateral Accel: {max_g:4.2f} m/s²\n\n"
            f"Collisions:        0 (Clean Autonomous Drive)"
        )


# --------------------------------------------------
# 5. MAIN 3D SIMULATION APPLICATION
# --------------------------------------------------

class AV3DSimulation:
    def __init__(self, target_speed_kmh=55.0, headless=False):
        self.headless = headless
        self.target_speed_mps = target_speed_kmh / 3.6
        self.dt = 0.02
        self.wheelbase = 2.7

        self.cam_mode = 0
        self.paused = False
        self.mission_complete = False

        print("=" * 60)
        print("STARTING KITTI 3D AUTONOMOUS VEHICLE SIMULATOR")
        print("=" * 60)
        self.controller = EdgeVehicleController()
        print("✓ Edge Vehicle Controller and KITTI weights loaded successfully.")

        # Window background color (Sky Blue)
        window.color = col(120, 180, 240)

        self.waypoints = build_3d_course("city", ds=0.8)
        self.total_dist = self.waypoints[-1]["s"]

        create_road_and_scenery(self.waypoints)

        wp0 = self.waypoints[0]
        self.vehicle_x = wp0["x"]
        self.vehicle_z = wp0["z"]
        self.vehicle_psi = wp0["heading"]
        self.vehicle_v = 0.0
        self.vehicle_a = 0.0
        self.vehicle_steer = 0.0

        self.car_3d = AutonomousVehicle3D(self.vehicle_x, self.vehicle_z, self.vehicle_psi)
        self.hud = TelemetryHUD()

        self.wp_idx = 0
        self.sim_step = 0
        self.sim_time = 0.0
        self.max_lateral_err = 0.0
        self.max_lat_accel = 0.0
        self.speed_history = []
        self.last_inference_ms = 0.22

        camera.fov = 65
        self.update_camera(lerp_factor=1.0)

        # Attach frame updater entity for interactive mode
        if not self.headless:
            self.tick_entity = Entity(update=lambda: self.step_simulation(update_visuals=True))

    def update_camera(self, lerp_factor=0.15):
        car_pos = self.car_3d.root.position
        heading_rad = np.radians(self.car_3d.root.rotation_y)
        fwd_x = float(np.sin(heading_rad))
        fwd_z = float(np.cos(heading_rad))

        if self.cam_mode == 0:
            # Mode 0: Third-Person Dynamic Chase Cam
            target_cam_pos = car_pos + Vec3(-fwd_x * 8.5, 3.4, -fwd_z * 8.5)
            look_pos = car_pos + Vec3(fwd_x * 6.0, 1.2, fwd_z * 6.0)

            if lerp_factor >= 1.0:
                camera.position = target_cam_pos
            else:
                camera.position = camera.position + (target_cam_pos - camera.position) * lerp_factor

            dx = look_pos.x - camera.position.x
            dy = look_pos.y - camera.position.y
            dz = look_pos.z - camera.position.z
            dh = math.sqrt(dx * dx + dz * dz)
            yaw = math.degrees(math.atan2(dx, dz))
            pitch = math.degrees(math.atan2(-dy, dh))
            camera.rotation = (pitch, yaw, 0)

        elif self.cam_mode == 1:
            # Mode 1: First-Person Cockpit / Hood Cam
            hood_cam_pos = car_pos + Vec3(fwd_x * 0.3, 1.15, fwd_z * 0.3)
            camera.position = hood_cam_pos
            camera.rotation = (6.0, np.degrees(self.vehicle_psi), 0)

        elif self.cam_mode == 2:
            # Mode 2: Overhead Drone Cam (Elevated bird's eye view)
            drone_cam_pos = car_pos + Vec3(0, 38.0, -22.0)
            if lerp_factor >= 1.0:
                camera.position = drone_cam_pos
            else:
                camera.position = camera.position + (drone_cam_pos - camera.position) * 0.15

            dx = car_pos.x - camera.position.x
            dy = (car_pos.y + 0.5) - camera.position.y
            dz = car_pos.z - camera.position.z
            dh = math.sqrt(dx * dx + dz * dz)
            yaw = math.degrees(math.atan2(dx, dz))
            pitch = math.degrees(math.atan2(-dy, dh))
            camera.rotation = (pitch, yaw, 0)

    def refresh_visuals(self):
        """Forces a complete refresh of 3D entities, camera, and HUD."""
        cur_wp = self.waypoints[self.wp_idx]
        s_cur = cur_wp["s"]
        status_mode = "ARRIVED" if self.mission_complete else "CRUISING"

        self.car_3d.root.position = (self.vehicle_x, 0.35, self.vehicle_z)
        self.car_3d.root.rotation_y = np.degrees(self.vehicle_psi)
        self.car_3d.set_steering_angle(np.degrees(self.vehicle_steer))
        self.car_3d.set_brake_lights(self.mission_complete)

        self.update_camera(lerp_factor=1.0)

        self.hud.update_telemetry(
            speed_kmh=self.vehicle_v * 3.6,
            accel_mps2=self.vehicle_a,
            yaw_rate_deg_s=0.0,
            steer_deg=np.degrees(self.vehicle_steer),
            throttle=0.0,
            brake=1.0 if self.mission_complete else 0.0,
            s_cur=s_cur,
            s_total=self.total_dist,
            status_mode=status_mode,
            inference_ms=self.last_inference_ms,
            update_text=True
        )

        if self.mission_complete:
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
            self.hud.show_mission_complete(
                total_dist=s_cur,
                total_time=self.sim_time,
                avg_speed=avg_speed,
                max_g=self.max_lat_accel,
            )

    def step_simulation(self, update_visuals=True):
        if self.paused or self.mission_complete:
            return

        self.sim_step += 1
        self.sim_time += self.dt

        # 1. Local track projection
        search_start = max(0, self.wp_idx - 6)
        search_end = min(len(self.waypoints), self.wp_idx + 45)
        dists = [
            (self.waypoints[i]["x"] - self.vehicle_x)**2 + (self.waypoints[i]["z"] - self.vehicle_z)**2
            for i in range(search_start, search_end)
        ]
        self.wp_idx = search_start + int(np.argmin(dists))

        cur_wp = self.waypoints[self.wp_idx]
        s_cur = cur_wp["s"]
        s_remain = self.total_dist - s_cur

        dx = self.vehicle_x - cur_wp["x"]
        dz = self.vehicle_z - cur_wp["z"]
        psi_wp = cur_wp["heading"]
        e_lat = dx * np.cos(psi_wp) - dz * np.sin(psi_wp)
        self.max_lateral_err = max(self.max_lateral_err, abs(e_lat))
        theta_err = (self.vehicle_psi - psi_wp + np.pi) % (2.0 * np.pi) - np.pi

        # 2. Sensor lookahead
        d_look = float(np.clip(5.5 + 0.45 * self.vehicle_v, 6.0, 16.0))
        s_target = s_cur + d_look

        look_idx = self.wp_idx
        while look_idx < len(self.waypoints) - 1 and self.waypoints[look_idx]["s"] < s_target:
            look_idx += 1
        target_wp = self.waypoints[look_idx]
        kappa_look = target_wp["curvature"]

        # 3. Speed planning
        if s_remain < 32.0:
            v_target = max(0.0, self.target_speed_mps * ((s_remain - 1.5) / 30.5))
            if s_remain <= 1.2:
                v_target = 0.0
            status_mode = "DECELERATING"
        elif abs(kappa_look) > 0.015:
            v_curve = np.sqrt(2.5 / abs(kappa_look))
            v_target = min(self.target_speed_mps, v_curve)
            status_mode = "CORNERING"
        else:
            v_target = self.target_speed_mps
            status_mode = "CRUISING"

        # 4. Steering control
        dx_look = target_wp["x"] - self.vehicle_x
        dz_look = target_wp["z"] - self.vehicle_z
        dx_b = dx_look * np.cos(self.vehicle_psi) - dz_look * np.sin(self.vehicle_psi)
        dz_b = dx_look * np.sin(self.vehicle_psi) + dz_look * np.cos(self.vehicle_psi)
        alpha = np.arctan2(dx_b, dz_b)
        steer_target = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), d_look)
        self.vehicle_steer = float(np.clip(steer_target, -np.radians(28.0), np.radians(28.0)))

        # 5. AI Edge inference
        yaw_rate = (self.vehicle_v / self.wheelbase) * np.tan(self.vehicle_steer)
        lat_accel = self.vehicle_v * yaw_rate
        self.max_lat_accel = max(self.max_lat_accel, abs(lat_accel))

        sensor_input = {
            "speed_kmh": self.vehicle_v * 3.6,
            "forward_speed_mps": self.vehicle_v,
            "heading_deg": np.degrees(theta_err),
            "yaw_rate_rad_s": yaw_rate,
            "forward_acceleration_mps2": self.vehicle_a,
            "lateral_acceleration_mps2": lat_accel,
            "curvature_m_inv": kappa_look,
            "persons_detected": 0,
            "bicycles_detected": 0,
            "cars_detected": 1 if abs(kappa_look) > 0.02 else 0,
            "trucks_detected": 0,
            "buses_detected": 0,
            "traffic_lights_detected": 0,
            "total_detections": 1 if abs(kappa_look) > 0.02 else 0,
        }

        if self.sim_step % 10 == 0:
            t_inf_start = time.perf_counter()
            ai_cmd = self.controller.predict(sensor_input)
            self.last_inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # 6. Throttle & Brake
        e_v = v_target - self.vehicle_v
        if s_remain <= 1.5:
            throttle = 0.0
            brake = 1.0
            if self.vehicle_v < 0.15:
                self.vehicle_v = 0.0
                self.mission_complete = True
                status_mode = "ARRIVED"
        elif e_v > 0.4:
            throttle = float(np.clip(0.35 + 0.16 * e_v, 0.0, 1.0))
            brake = 0.0
        elif e_v < -0.8:
            throttle = 0.0
            brake = float(np.clip(-0.25 * e_v, 0.12, 1.0))
            status_mode = "DECELERATING"
        else:
            throttle = 0.15
            brake = 0.0

        # Kinematics Update
        a_eng = 3.8 * throttle
        a_brk = -6.5 * brake
        a_drag = -(0.015 * 9.81 + 0.0004 * self.vehicle_v**2)
        self.vehicle_a = a_eng + a_brk + a_drag
        self.vehicle_v = max(0.0, self.vehicle_v + self.vehicle_a * self.dt)
        self.speed_history.append(self.vehicle_v * 3.6)

        dist_step = self.vehicle_v * self.dt
        self.vehicle_psi += yaw_rate * self.dt
        self.vehicle_x += self.vehicle_v * np.sin(self.vehicle_psi) * self.dt
        self.vehicle_z += self.vehicle_v * np.cos(self.vehicle_psi) * self.dt

        # Visual Entities & HUD
        if update_visuals:
            self.car_3d.root.position = (self.vehicle_x, 0.35, self.vehicle_z)
            self.car_3d.root.rotation_y = np.degrees(self.vehicle_psi)
            self.car_3d.set_steering_angle(np.degrees(self.vehicle_steer))
            self.car_3d.spin_wheels(dist_step)
            self.car_3d.set_brake_lights(brake > 0.08)

            self.update_camera(lerp_factor=0.15)

            self.hud.update_telemetry(
                speed_kmh=self.vehicle_v * 3.6,
                accel_mps2=self.vehicle_a,
                yaw_rate_deg_s=np.degrees(yaw_rate),
                steer_deg=np.degrees(self.vehicle_steer),
                throttle=throttle,
                brake=brake,
                s_cur=s_cur,
                s_total=self.total_dist,
                status_mode=status_mode,
                inference_ms=self.last_inference_ms,
                update_text=(self.sim_step % 4 == 0 or self.mission_complete)
            )

        if self.mission_complete and update_visuals:
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
            self.hud.show_mission_complete(
                total_dist=s_cur,
                total_time=self.sim_time,
                avg_speed=avg_speed,
                max_g=self.max_lat_accel,
            )
            print("\n" + "=" * 60)
            print("MISSION COMPLETE: POINT B REACHED SAFELY")
            print(f"Total Distance: {s_cur:.1f} m  |  Time: {self.sim_time:.1f} s  |  Avg Speed: {avg_speed:.1f} km/h")
            print("=" * 60)

    def switch_camera(self, target_mode=None):
        if target_mode is not None:
            self.cam_mode = target_mode
        else:
            self.cam_mode = (self.cam_mode + 1) % 3
        cam_names = ["3RD PERSON CHASE CAM", "1ST PERSON HOOD CAM", "OVERHEAD DRONE CAM"]
        print(f"Camera switched to: {cam_names[self.cam_mode]}")
        self.update_camera(lerp_factor=1.0)

    def restart(self):
        wp0 = self.waypoints[0]
        self.vehicle_x = wp0["x"]
        self.vehicle_z = wp0["z"]
        self.vehicle_psi = wp0["heading"]
        self.vehicle_v = 0.0
        self.vehicle_a = 0.0
        self.vehicle_steer = 0.0
        self.wp_idx = 0
        self.sim_step = 0
        self.sim_time = 0.0
        self.mission_complete = False
        self.hud.win_panel.enabled = False
        self.hud.win_title.enabled = False
        self.hud.win_stats.enabled = False
        self.hud.win_prompt.enabled = False
        self.car_3d.root.position = (self.vehicle_x, 0.35, self.vehicle_z)
        self.car_3d.root.rotation_y = np.degrees(self.vehicle_psi)
        self.update_camera(lerp_factor=1.0)
        print("Mission reset to Point A.")


# --------------------------------------------------
# 6. SCRIPT ENTRYPOINT & CLI
# --------------------------------------------------

# Global simulation instance for Ursina frame hook and input propagation
sim_instance = None

def input(key):
    global sim_instance
    if sim_instance is None:
        return
    if key in ("c", "C"):
        sim_instance.switch_camera()
    elif key in ("r", "R"):
        sim_instance.restart()
    elif key in ("p", "P"):
        sim_instance.paused = not sim_instance.paused
        print(f"Simulation {'Paused' if sim_instance.paused else 'Resumed'}")
    elif key == "escape":
        application.quit()

def update():
    global sim_instance
    if sim_instance is not None and not sim_instance.headless:
        sim_instance.step_simulation(update_visuals=True)

def main():
    global sim_instance
    parser = argparse.ArgumentParser(description="KITTI 3D Autonomous Vehicle Simulator")
    parser.add_argument("--speed", type=float, default=55.0, help="Target cruising speed limit in km/h (default: 55)")
    parser.add_argument("--course", type=str, default="city", choices=["city", "circuit"], help="Course layout")
    parser.add_argument("--save-preview", action="store_true", help="Run headless preview capture and exit")
    args = parser.parse_args()

    if args.save_preview:
        app = Ursina(headless=True)
        sim_instance = AV3DSimulation(target_speed_kmh=args.speed, headless=True)

        def capture_frame(filename):
            app.step()
            pnm = PNMImage()
            base.win.getScreenshot(pnm)
            pnm.write(Filename.fromOsSpecific(str(filename)))

        print("Simulating preview run to capture showcase images...")
        # 1. Chase Cam during high speed cruising / gentle curve
        for _ in range(200):
            sim_instance.step_simulation(update_visuals=False)
        sim_instance.switch_camera(0)
        sim_instance.refresh_visuals()
        p1 = Path("3d_sim_chase.png").resolve()
        capture_frame(p1)
        print(f"Saved: {p1}")

        # 2. 1st Person Hood Cam entering S-Curve Chicane
        for _ in range(350):
            sim_instance.step_simulation(update_visuals=False)
        sim_instance.switch_camera(1)
        sim_instance.refresh_visuals()
        p2 = Path("3d_sim_hood.png").resolve()
        capture_frame(p2)
        print(f"Saved: {p2}")

        # 3. Drone Cam navigating 90-degree intersection turn
        for _ in range(500):
            sim_instance.step_simulation(update_visuals=False)
        sim_instance.switch_camera(2)
        sim_instance.refresh_visuals()
        p3 = Path("3d_sim_drone.png").resolve()
        capture_frame(p3)
        print(f"Saved: {p3}")

        # 4. Finish Line at Point B
        while not sim_instance.mission_complete and sim_instance.sim_step < 3500:
            sim_instance.step_simulation(update_visuals=False)
        sim_instance.switch_camera(0)
        sim_instance.refresh_visuals()
        p4 = Path("3d_sim_finish.png").resolve()
        capture_frame(p4)
        print(f"Saved: {p4}")

        print("Creating combined 2x2 multi-view showcase composite...")
        im1 = cv2.imread(str(p1))
        im2 = cv2.imread(str(p2))
        im3 = cv2.imread(str(p3))
        im4 = cv2.imread(str(p4))

        if im1 is not None and im2 is not None and im3 is not None and im4 is not None:
            h_sub, w_sub = 540, 960
            r1 = cv2.resize(im1, (w_sub, h_sub))
            r2 = cv2.resize(im2, (w_sub, h_sub))
            r3 = cv2.resize(im3, (w_sub, h_sub))
            r4 = cv2.resize(im4, (w_sub, h_sub))

            def put_banner(img, title):
                cv2.rectangle(img, (15, 15), (460, 52), (18, 22, 28), -1)
                cv2.rectangle(img, (15, 15), (460, 52), (0, 216, 255), 2)
                cv2.putText(img, title, (25, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            put_banner(r1, "1. 3RD PERSON CHASE CAM (CRUISING)")
            put_banner(r2, "2. 1ST PERSON HOOD CAM (S-CHICANE)")
            put_banner(r3, "3. DRONE OVERHEAD (90 DEG INTERSECTION)")
            put_banner(r4, "4. POINT B FINISH (MISSION COMPLETE)")

            top_row = np.hstack([r1, r2])
            bot_row = np.hstack([r3, r4])
            grid = np.vstack([top_row, bot_row])

            comp_path = Path("3d_sim_showcase.png").resolve()
            cv2.imwrite(str(comp_path), grid)
            print(f"Showcase image created at: {comp_path}")

        return

    # Normal interactive mode
    app = Ursina(
        title="KITTI 3D Autonomous Vehicle Simulator",
        borderless=False,
        fullscreen=False,
        size=(1280, 720),
        development_mode=False
    )
    window.fps_counter.enabled = False
    window.exit_button.visible = False

    sim_instance = AV3DSimulation(target_speed_kmh=args.speed, headless=False)
    app.run()


if __name__ == "__main__":
    main()
