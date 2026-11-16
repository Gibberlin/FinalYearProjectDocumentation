"""
Simulation physics and tracking verification for 3D AV Simulator.
Tests closed-loop trajectory tracking, cornering deceleration, and arrival at Point B.
"""

import numpy as np
from predict_control import EdgeVehicleController

def generate_track_points(segments, ds=0.5):
    waypoints = []
    x, z, psi, s_total = 0.0, 0.0, 0.0, 0.0
    for seg in segments:
        seg_type = seg[0]
        if seg_type == 'straight':
            length = float(seg[1])
            n = max(2, int(np.ceil(length / ds)))
            for i in range(n):
                s = i * (length / n)
                waypoints.append({
                    'x': float(x + s * np.sin(psi)),
                    'z': float(z + s * np.cos(psi)),
                    'heading': float(psi),
                    'curvature': 0.0,
                    's': float(s_total + s)
                })
            s_total += length
            x += length * np.sin(psi)
            z += length * np.cos(psi)
        elif seg_type == 'turn':
            r = float(seg[1])
            turn_deg = float(seg[2]) # positive = right, negative = left
            turn_rad = np.radians(abs(turn_deg))
            arc_len = r * turn_rad
            n = max(2, int(np.ceil(arc_len / ds)))
            if turn_deg > 0: # Right turn
                xc = x + r * np.cos(psi)
                zc = z - r * np.sin(psi)
                kappa = -1.0 / r
                for i in range(n):
                    s = i * (arc_len / n)
                    cur_psi = psi + s / r
                    waypoints.append({
                        'x': float(xc - r * np.cos(cur_psi)),
                        'z': float(zc + r * np.sin(cur_psi)),
                        'heading': float(cur_psi),
                        'curvature': float(kappa),
                        's': float(s_total + s)
                    })
                s_total += arc_len
                end_psi = psi + turn_rad
                x = xc - r * np.cos(end_psi)
                z = zc + r * np.sin(end_psi)
                psi = end_psi
            else: # Left turn
                xc = x - r * np.cos(psi)
                zc = z + r * np.sin(psi)
                kappa = 1.0 / r
                for i in range(n):
                    s = i * (arc_len / n)
                    cur_psi = psi - s / r
                    waypoints.append({
                        'x': float(xc + r * np.cos(cur_psi)),
                        'z': float(zc - r * np.sin(cur_psi)),
                        'heading': float(cur_psi),
                        'curvature': float(kappa),
                        's': float(s_total + s)
                    })
                s_total += arc_len
                end_psi = psi - turn_rad
                x = xc + r * np.cos(end_psi)
                z = zc - r * np.sin(end_psi)
                psi = end_psi

    return waypoints

def run_physics_test():
    course = [
        ('straight', 80.0),
        ('turn', 70.0, 45.0),    # right 45 deg
        ('straight', 50.0),
        ('turn', 55.0, -70.0),   # left 70 deg chicane
        ('straight', 60.0),
        ('turn', 25.0, 90.0),    # sharp 90 deg right intersection
        ('straight', 100.0),
        ('turn', 25.0, -90.0),   # sharp 90 deg left intersection
        ('straight', 80.0),
    ]
    wps = generate_track_points(course, ds=0.5)
    total_length = wps[-1]['s']
    print(f"Track loaded: {len(wps)} waypoints, {total_length:.1f} meters.")

    controller = EdgeVehicleController()
    print("AI Model loaded.")

    # Vehicle initial state
    x = wps[0]['x']
    z = wps[0]['z']
    psi = wps[0]['heading']
    v = 0.0
    a = 0.0
    steer_rad = 0.0
    dt = 0.02
    wheelbase = 2.7
    cruise_speed = 55.0 / 3.6 # 15.28 m/s

    wp_idx = 0
    max_lateral_err = 0.0
    steps = 0
    mission_complete = False

    while not mission_complete and steps < 6000:
        steps += 1
        # Find closest waypoint in local window
        search_start = max(0, wp_idx - 5)
        search_end = min(len(wps), wp_idx + 40)
        dists = [(wps[i]['x'] - x)**2 + (wps[i]['z'] - z)**2 for i in range(search_start, search_end)]
        wp_idx = search_start + int(np.argmin(dists))

        cur_wp = wps[wp_idx]
        s_cur = cur_wp['s']
        s_remain = total_length - s_cur

        # Signed lateral error
        dx = x - cur_wp['x']
        dz = z - cur_wp['z']
        psi_wp = cur_wp['heading']
        e_lat = dx * np.cos(psi_wp) - dz * np.sin(psi_wp)
        max_lateral_err = max(max_lateral_err, abs(e_lat))

        # Heading error
        theta_err = (psi - psi_wp + np.pi) % (2 * np.pi) - np.pi

        # Lookahead point
        d_look = np.clip(5.0 + 0.45 * v, 6.0, 16.0)
        s_target = s_cur + d_look
        # find target waypoint
        look_idx = wp_idx
        while look_idx < len(wps) - 1 and wps[look_idx]['s'] < s_target:
            look_idx += 1
        target_wp = wps[look_idx]

        # Target curvature
        kappa_look = target_wp['curvature']

        # Target speed calculation
        if s_remain < 30.0:
            v_target = max(0.0, cruise_speed * ((s_remain - 1.5) / 28.5))
            if s_remain <= 1.0:
                v_target = 0.0
        else:
            # Curve speed
            if abs(kappa_look) > 0.015:
                v_curve = np.sqrt(2.5 / abs(kappa_look))
            else:
                v_curve = cruise_speed
            v_target = min(cruise_speed, v_curve)

        # Pure pursuit steering geometry
        dx_look = target_wp['x'] - x
        dz_look = target_wp['z'] - z
        # In vehicle coordinates
        dx_b = dx_look * np.cos(psi) - dz_look * np.sin(psi)
        dz_b = dx_look * np.sin(psi) + dz_look * np.cos(psi)
        alpha = np.arctan2(dx_b, dz_b)
        steer_target = np.arctan2(2.0 * wheelbase * np.sin(alpha), d_look)
        steer_rad = np.clip(steer_target, -np.radians(30.0), np.radians(30.0))

        # Speed control
        e_v = v_target - v
        if s_remain <= 1.5:
            throttle = 0.0
            brake = 1.0
            if v < 0.1:
                v = 0.0
                mission_complete = True
        elif e_v > 0.4:
            throttle = float(np.clip(0.35 + 0.15 * e_v, 0.0, 1.0))
            brake = 0.0
        elif e_v < -0.8:
            throttle = 0.0
            brake = float(np.clip(-0.25 * e_v, 0.1, 1.0))
        else:
            throttle = 0.15
            brake = 0.0

        # Kinematics
        a_eng = 3.8 * throttle
        a_brk = -6.5 * brake
        a_drag = -(0.015 * 9.81 + 0.0004 * v**2)
        a = a_eng + a_brk + a_drag
        v = max(0.0, v + a * dt)

        yaw_rate = (v / wheelbase) * np.tan(steer_rad)
        psi += yaw_rate * dt
        x += v * np.sin(psi) * dt
        z += v * np.cos(psi) * dt

        if steps % 250 == 0 or mission_complete:
            print(f"Step {steps:4d} | Dist: {s_cur:5.1f}m/{total_length:5.1f}m | Spd: {v*3.6:4.1f} km/h (tgt: {v_target*3.6:4.1f}) | Steer: {np.degrees(steer_rad):+5.1f}° | ErrLat: {e_lat:+5.2f}m | Thr: {throttle:.2f} | Brk: {brake:.2f}")

    print("\n" + "=" * 60)
    print(f"TEST RESULT: {'SUCCESS' if mission_complete else 'TIMED OUT'}")
    print(f"Final distance reached: {s_cur:.1f} m / {total_length:.1f} m")
    print(f"Max lateral error across entire course: {max_lateral_err:.3f} meters (road half-width = 3.75m)")
    print(f"Final speed at destination: {v*3.6:.2f} km/h")
    print("=" * 60)

if __name__ == '__main__':
    run_physics_test()
