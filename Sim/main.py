import mujoco
import mujoco.viewer
import numpy as np
import time

# Load scene
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)
# Initialize PID controller state
x_errPrev = 0.0
y_errPrev = 0.0
z_errPrev = 0.0
roll_errPrev = 0.0
pitch_errPrev = 0.0
dt = 0.0
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
DISPLAY_LINES = 12  # must match exactly the number of print() calls in display()
_first_display = True

def _bar(d, width=20):
    if d < 0:
        return '─' * width
    filled = int(min(d, 4.0) / 4.0 * width)
    return '█' * filled + '░' * (width - filled)

def display(dists, target, pos, yaw_deg, wp_i, drone_roll, drone_pitch):
    global _first_display
    if not _first_display:
        print(f'\033[{DISPLAY_LINES}F', end='')  # cursor up N lines, column 0
    _first_display = False

    def fmt(d):
        return f'{d:5.2f} m' if d >= 0 else '  oor  '

    n = len(WAYPOINTS)
    print(f'  Drone    X={pos[0]:+.2f}  Y={pos[1]:+.2f}  Z={pos[2]:+.2f}  Yaw={yaw_deg:+5.1f}° ')
    print(f'  Attitude Roll={np.rad2deg(drone_roll):+5.2f} Pitch={np.rad2deg(drone_pitch):+5.2f}                 ')
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

        # Set desired position
        x_des = target_pos[0] #0.5
        y_des = target_pos[1] #0.0
        z_des = target_pos[2] #0.5

        # PID controller
        # Calculate time step
        dt = model.opt.timestep

        # Set thrust to control altitude (Z)
        z_err = z_des - data.qpos[2]
        z_vel = data.qvel[2]
        thrust = 0.26487 + (z_err * 3.0) - (z_vel * 1.5)
        data.ctrl[0] = np.clip(thrust, 0, 0.35)

        # PID loop for desired axis angles (roll and pitch)
        Kp_acc, Ki_acc, Kd_acc = 0.15, 0.01, 0.5
        x_err = x_des - data.qpos[0]
        y_err = y_des - data.qpos[1]
        x_errI = x_errPrev + x_err*dt
        y_errI = y_errPrev + y_err*dt
        x_errI = np.clip(x_errI, -0.5, 0.5)
        y_errI = np.clip(y_errI, -0.5, 0.5)
        x_errPrev = x_err
        y_errPrev = y_err
        ax_des = (x_err * Kp_acc) + (x_errI * Ki_acc) - (data.qvel[0] * Kd_acc)
        ay_des = (y_err * Kp_acc) + (y_errI * Ki_acc) - (data.qvel[1] * Kd_acc)

        #Set desired roll and pitch based on desired accelerations
        roll_des = np.clip(ay_des / 9.81, -0.5, 0.5)  # Roll controls lateral (Y) acceleration
        pitch_des = np.clip(-ax_des / 9.81, -0.5, 0.5) # Pitch controls longitudinal (X) acceleration

        # Calculate drone current roll and pitch from orientation quaternion
        q = data.qpos[3:7]  # (w, x, y, z)
        roll  = -np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
        pitch = -np.arcsin(2*(q[0]*q[2] - q[3]*q[1]))
        yaw   = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

        roll_err = roll_des - roll
        pitch_err = pitch_des - pitch
        roll_errI = roll_errPrev + roll_err*dt
        pitch_errI = pitch_errPrev + pitch_err*dt
        roll_D = data.qvel[3]  # roll rate
        pitch_D = data.qvel[4]  # pitch rate
        roll_errPrev = roll_err
        pitch_errPrev = pitch_err
        roll_errI = np.clip(roll_errI, -0.5, 0.5)
        pitch_errI = np.clip(pitch_errI, -0.5, 0.5)

        # Implement control through roll and pitch
        disturbance = 0.5 if 5.0< data.time < 5.5 else 0.0
        Kp, Ki, Kd = 5.0, 0.00, 3.0
        data.ctrl[1] = np.clip(roll_err * Kp + roll_errI * Ki + roll_D * Kd, -0.5, 0.5)  # x_moment (Roll)
        data.ctrl[2] = np.clip(pitch_err * Kp + pitch_errI * Ki + pitch_D * Kd, -0.5, 0.5) # y_moment (Pitch)
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
            display(dists, target_pos, data.qpos[:3], np.degrees(yaw), wp_idx, roll,pitch)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start


        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
