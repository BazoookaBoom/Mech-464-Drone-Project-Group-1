import mujoco
import mujoco.viewer
import numpy as np
import time

# Load scene
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 0.5 m cube centred at origin, bottom face at Z=0.5, top at Z=1.0
WAYPOINTS = np.array([
    [-0.25, -0.25, 0.5],
    [ 0.25, -0.25, 0.5],
    [ 0.25,  0.25, 0.5],
    [-0.25,  0.25, 0.5],
    [-0.25,  0.25, 1.0],
    [ 0.25,  0.25, 1.0],
    [ 0.25, -0.25, 1.0],
    [-0.25, -0.25, 1.0],
])
WAYPOINT_THRESHOLD = 0.12  # metres — advance to next when within this distance
wp_idx = 0
target_pos = WAYPOINTS[wp_idx].copy()

# Sensor addresses (looked up once at startup)
SENSOR_NAMES = ['range_fwd', 'range_back', 'range_left', 'range_right', 'range_up', 'range_down']
sensor_addrs = {}
for name in SENSOR_NAMES:
    try:
        sid = model.sensor(name).id
        sensor_addrs[name] = model.sensor_adr[sid]
    except Exception as e:
        print(f"Warning: sensor '{name}' not found: {e}")
        sensor_addrs[name] = None

# ── Terminal display ───────────────────────────────────────────────────────────
DISPLAY_LINES = 11  # must match exactly the number of print() calls in display()
_first_display = True

def _bar(d, width=20):
    if d < 0:
        return '─' * width
    filled = int(min(d, 4.0) / 4.0 * width)
    return '█' * filled + '░' * (width - filled)

def display(dists, target, pos, yaw_deg, wp_i):
    global _first_display
    if not _first_display:
        print(f'\033[{DISPLAY_LINES}F', end='')  # cursor up N lines, column 0
    _first_display = False

    def fmt(d):
        return f'{d:5.2f} m' if d >= 0 else '  oor  '

    n = len(WAYPOINTS)
    print(f'  Drone    X={pos[0]:+.2f}  Y={pos[1]:+.2f}  Z={pos[2]:+.2f}  Yaw={yaw_deg:+5.1f}°   ')
    print(f'  Waypoint {wp_i % n + 1}/{n}  →  X={target[0]:+.2f}  Y={target[1]:+.2f}  Z={target[2]:+.2f}   ')
    print(f'  {"─"*48}')
    print(f'  Forward  [{_bar(dists["range_fwd"])}]  {fmt(dists["range_fwd"])}')
    print(f'  Back     [{_bar(dists["range_back"])}]  {fmt(dists["range_back"])}')
    print(f'  Left     [{_bar(dists["range_left"])}]  {fmt(dists["range_left"])}')
    print(f'  Right    [{_bar(dists["range_right"])}]  {fmt(dists["range_right"])}')
    print(f'  Up       [{_bar(dists["range_up"])}]  {fmt(dists["range_up"])}')
    print(f'  Down     [{_bar(dists["range_down"])}]  {fmt(dists["range_down"])}')
    print(f'  {"─"*48}')
    print(f'  Ctrl+C to quit                                    ')

# ── Simulation loop ────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    last_display_time = -1.0

    while viewer.is_running():
        step_start = time.time()

        # Advance waypoint when close enough
        dist_to_wp = np.linalg.norm(data.qpos[:3] - target_pos)
        if dist_to_wp < WAYPOINT_THRESHOLD:
            wp_idx += 1
            target_pos = WAYPOINTS[wp_idx % len(WAYPOINTS)].copy()

        # PID controller
        z_err = target_pos[2] - data.qpos[2]
        z_vel = data.qvel[2]
        thrust = 0.26487 + (z_err * 3.0) - (z_vel * 1.5)
        data.ctrl[0] = np.clip(thrust, 0, 0.35)

        x_err = target_pos[0] - data.qpos[0]
        y_err = target_pos[1] - data.qpos[1]
        data.ctrl[1] = (y_err * 0.05) - (data.qvel[1] * 0.02)   # x_moment (Roll)
        data.ctrl[2] = -(x_err * 0.05) + (data.qvel[0] * 0.02)  # y_moment (Pitch)
        data.ctrl[3] = 0

        # Read sensors
        dists = {
            name: float(data.sensordata[addr]) if addr is not None else -1.0
            for name, addr in sensor_addrs.items()
        }

        # Update terminal at ~5 Hz
        if data.time - last_display_time >= 0.2:
            last_display_time = data.time
            q = data.qpos[3:7]
            yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
            display(dists, target_pos, data.qpos[:3], np.degrees(yaw), wp_idx)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
