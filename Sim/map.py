"""
map.py  —  Drone ToF point-cloud mapper
============================================
Usage (from your main.py sim loop):

    from map import DroneMapper
    mapper = DroneMapper()

    # inside your while viewer.is_running() loop:
    mapper.update(data, model)

    # when done (Ctrl+C or loop exit):
    mapper.save('flight_map')
    mapper.plot_topdown('flight_map_topdown.png', waypoints=WAYPOINTS)
    mapper.plot_3d(waypoints=WAYPOINTS)
"""

import numpy as np
import csv
from pathlib import Path

# ── Sensor body-frame unit vectors ────────────────────────────────────────────
# Each ToF sensor fires a ray in a fixed direction relative to the drone body.
# +X = forward, +Y = left, +Z = up  (standard aerospace body frame)
SENSOR_DIRS = {
    'range_fwd':   np.array([ 1.0,  0.0,  0.0]),
    'range_back':  np.array([-1.0,  0.0,  0.0]),
    'range_left':  np.array([ 0.0,  1.0,  0.0]),
    'range_right': np.array([ 0.0, -1.0,  0.0]),
    # 'range_up':   np.array([ 0.0,  0.0,  1.0]),   # disabled — traces ceiling above path
    # 'range_down': np.array([ 0.0,  0.0, -1.0]),   # disabled — traces floor below path
}

# Readings above this value are treated as out-of-range and discarded
OOR_THRESHOLD = 3.9   # metres (MuJoCo rangefinder returns max when nothing hit)

# Minimum meaningful range — readings below this are likely noise
MIN_RANGE = 0.02      # metres


def _yaw_from_quat(q):
    """
    Extract yaw angle (rotation about world Z) from a MuJoCo quaternion [w, x, y, z].
    Returns yaw in radians.
    """
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(vec, yaw):
    """Rotate a 3-vector about the Z axis by yaw radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])
    return R @ vec


def tof_to_world_point(drone_pos, yaw, sensor_dir, distance):
    """
    Convert a single ToF reading into a world-frame 3D point.

    Parameters
    ----------
    drone_pos   : (3,) array  — drone XYZ in world frame
    yaw         : float       — drone yaw in radians
    sensor_dir  : (3,) array  — sensor ray direction in drone body frame
    distance    : float       — range reading in metres

    Returns
    -------
    (3,) array  — hit point in world frame, or None if out of range
    """
    if distance < MIN_RANGE or distance > OOR_THRESHOLD:
        return None
    world_dir = _rotate_z(sensor_dir, yaw)
    return drone_pos + world_dir * distance


class DroneMapper:
    """
    Collects world-frame 3D hit points from all ToF sensors each sim step.

    Points are stored as a list of dicts:
        { 'time', 'sensor', 'x', 'y', 'z', 'drone_x', 'drone_y', 'drone_z' }
    """

    def __init__(self, oor_threshold=OOR_THRESHOLD):
        self.oor_threshold = oor_threshold
        self.points = []
        self._sensor_addrs = {}
        self._model_id = None

    # ── Sensor address lookup ─────────────────────────────────────────────────
    def _build_sensor_map(self, model):
        self._sensor_addrs = {}
        for name in SENSOR_DIRS:
            try:
                sid = model.sensor(name).id
                self._sensor_addrs[name] = model.sensor_adr[sid]
            except Exception:
                self._sensor_addrs[name] = None
        self._model_id = id(model)

    # ── Main update call (call once per sim step) ─────────────────────────────
    def update(self, data, model):
        """
        Read drone state + all sensor values and append hit points to self.points.
        Call this every sim step inside your main loop.
        """
        if id(model) != self._model_id:
            self._build_sensor_map(model)

        drone_pos = np.array(data.qpos[:3])
        q         = data.qpos[3:7]
        yaw       = _yaw_from_quat(q)
        sim_time  = float(data.time)

        for name, body_dir in SENSOR_DIRS.items():
            addr = self._sensor_addrs.get(name)
            if addr is None:
                continue
            dist = float(data.sensordata[addr])
            pt   = tof_to_world_point(drone_pos, yaw, body_dir, dist)
            if pt is not None:
                self.points.append({
                    'time':    sim_time,
                    'sensor':  name,
                    'x':       float(pt[0]),
                    'y':       float(pt[1]),
                    'z':       float(pt[2]),
                    'drone_x': float(drone_pos[0]),
                    'drone_y': float(drone_pos[1]),
                    'drone_z': float(drone_pos[2]),
                })

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def xyz(self):
        """Return all hit points as an (N, 3) numpy array."""
        if not self.points:
            return np.empty((0, 3))
        return np.array([[p['x'], p['y'], p['z']] for p in self.points])

    @property
    def count(self):
        return len(self.points)

    # ── Save ─────────────────────────────────────────────────────────────────
    def save(self, stem='flight_map'):
        """
        Save collected points to both .npz and .csv.

        Parameters
        ----------
        stem : str — filename without extension (e.g. 'flight_map')
        """
        if not self.points:
            print('[mapper] No points to save.')
            return

        # NPZ (compact, fast to reload)
        npz_path  = Path(stem + '.npz')
        arr       = self.xyz
        drone_xyz = np.array([[p['drone_x'], p['drone_y'], p['drone_z']] for p in self.points])
        times     = np.array([p['time']   for p in self.points])
        sensors   = np.array([p['sensor'] for p in self.points])
        np.savez_compressed(
            npz_path,
            xyz=arr,
            drone_xyz=drone_xyz,
            time=times,
            sensor=sensors,
        )
        print(f'[mapper] Saved {len(self.points)} points → {npz_path}')

        # CSV (human-readable)
        csv_path   = Path(stem + '.csv')
        fieldnames = ['time', 'sensor', 'x', 'y', 'z', 'drone_x', 'drone_y', 'drone_z']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.points)
        print(f'[mapper] Saved CSV          → {csv_path}')

    # ── Top-down 2D occupancy map ─────────────────────────────────────────────
    def plot_topdown(
        self,
        output_path='flight_map_topdown.png',
        resolution=0.05,
        z_min=None,
        z_max=None,
        show=False,
        waypoints=None,
    ):
        """
        Generate a top-down occupancy map (bird's-eye view, X-Y plane).

        Parameters
        ----------
        output_path : str   — where to save the PNG
        resolution  : float — metres per pixel (smaller = finer grid)
        z_min/z_max : float — optional Z-slice filter
        show        : bool  — call plt.show() after saving
        waypoints   : array — optional (N,3) waypoints to plot as markers
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('[mapper] matplotlib not found — pip install matplotlib')
            return

        if not self.points:
            print('[mapper] No points to plot.')
            return

        pts = self.xyz

        # Optional Z slice
        mask = np.ones(len(pts), dtype=bool)
        if z_min is not None:
            mask &= pts[:, 2] >= z_min
        if z_max is not None:
            mask &= pts[:, 2] <= z_max
        pts = pts[mask]

        if len(pts) == 0:
            print('[mapper] No points in Z slice.')
            return

        x, y = pts[:, 0], pts[:, 1]

        # Build grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        pad    = resolution * 4
        x_min -= pad; x_max += pad
        y_min -= pad; y_max += pad

        cols = max(1, int(np.ceil((x_max - x_min) / resolution)))
        rows = max(1, int(np.ceil((y_max - y_min) / resolution)))
        grid = np.zeros((rows, cols), dtype=np.int32)
        ix   = np.clip(((x - x_min) / resolution).astype(int), 0, cols - 1)
        iy   = np.clip(((y - y_min) / resolution).astype(int), 0, rows - 1)
        np.add.at(grid, (iy, ix), 1)

        drone_xy = np.array([[p['drone_x'], p['drone_y']] for p in self.points])

        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        display_grid = np.log1p(grid.astype(float))
        im = ax.imshow(
            display_grid,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='plasma',
            interpolation='nearest',
            aspect='equal',
        )

        ax.plot(drone_xy[:, 0], drone_xy[:, 1],
                color='#00e5ff', linewidth=0.8, alpha=0.7, label='Drone path')
        ax.plot(*drone_xy[0],  'o', color='#69ff47', markersize=6, label='Start')
        ax.plot(*drone_xy[-1], 's', color='#ff4081', markersize=6, label='End')

        if waypoints is not None:
            wp = np.array(waypoints)
            ax.scatter(wp[:, 0], wp[:, 1], c='white', s=40, zorder=5,
                       marker='x', linewidths=1.5, label='Waypoints')

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('log(hit count + 1)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        z_label = ''
        if z_min is not None or z_max is not None:
            lo = f'{z_min:.2f}' if z_min is not None else '−∞'
            hi = f'{z_max:.2f}' if z_max is not None else '+∞'
            z_label = f'  |  Z slice [{lo}, {hi}] m'

        ax.set_title(
            f'Top-down occupancy map  —  {len(pts)} pts  |  {resolution*100:.0f} cm/cell{z_label}',
            color='white', fontsize=11, pad=10,
        )
        ax.set_xlabel('X (m)  →  East', color='white')
        ax.set_ylabel('Y (m)  →  North', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.legend(loc='upper left', fontsize=8,
                  facecolor='#2a2a3e', edgecolor='#555', labelcolor='white')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'[mapper] Saved top-down map → {output_path}')
        if show:
            plt.show()
        plt.close()

    # ── Interactive 3D point cloud ────────────────────────────────────────────
    def plot_3d(self, point_size=2.0, waypoints=None):
        """
        Open an interactive rotatable 3D scatter plot of all hit points.

        Each sensor gets its own colour. The drone flight path is drawn as a
        thin line. Click and drag to rotate, scroll to zoom.

        Parameters
        ----------
        point_size : float — scatter marker size
        waypoints  : array — optional (N,3) waypoints to plot as markers
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            print('[mapper] matplotlib not found — pip install matplotlib')
            return

        if not self.points:
            print('[mapper] No points to plot.')
            return

        SENSOR_COLOURS = {
            'range_fwd':   '#00e5ff',   # cyan
            'range_back':  '#ff4081',   # pink
            'range_left':  '#69ff47',   # green
            'range_right': '#ffab40',   # amber
            'range_up':    '#ea80fc',   # purple
            'range_down':  '#ffff00',   # yellow
        }

        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1a1a2e')

        # Plot each sensor's points separately so legend works
        pts_by_sensor = {}
        for p in self.points:
            pts_by_sensor.setdefault(p['sensor'], []).append((p['x'], p['y'], p['z']))

        for sensor_name, coords in pts_by_sensor.items():
            arr    = np.array(coords)
            colour = SENSOR_COLOURS.get(sensor_name, '#ffffff')
            label  = sensor_name.replace('range_', '')
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                       c=colour, s=point_size, alpha=0.6,
                       label=label, depthshade=True)

        # Drone flight path
        drone_xyz = np.array([[p['drone_x'], p['drone_y'], p['drone_z']] for p in self.points])
        path = drone_xyz[::10]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color='white', linewidth=0.8, alpha=0.5, label='drone path')
        ax.scatter(*drone_xyz[0],  color='#69ff47', s=60, zorder=5, marker='o', label='start')
        ax.scatter(*drone_xyz[-1], color='#ff4081', s=60, zorder=5, marker='s', label='end')

        if waypoints is not None:
            wp = np.array(waypoints)
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                       c='white', s=50, marker='x', linewidths=1.5,
                       label='waypoints', zorder=6)

        ax.set_xlabel('X (m)', color='white', labelpad=8)
        ax.set_ylabel('Y (m)  →  North', color='white', labelpad=8)
        ax.set_zlabel('Z (m)', color='white', labelpad=8)
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333')
        ax.yaxis.pane.set_edgecolor('#333')
        ax.zaxis.pane.set_edgecolor('#333')
        ax.grid(True, color='#333333', linewidth=0.5)

        ax.set_title(
            f'3D point cloud  —  {len(self.points)} hits  |  drag to rotate  |  scroll to zoom',
            color='white', fontsize=11, pad=12,
        )
        ax.legend(loc='upper left', fontsize=8, markerscale=3,
                  facecolor='#2a2a3e', edgecolor='#555', labelcolor='white')

        print(f'[mapper] Showing 3D scatter ({len(self.points)} points) — close window to continue.')
        plt.tight_layout()
        plt.show()