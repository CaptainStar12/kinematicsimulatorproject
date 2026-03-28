import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ========================== Kalman Filter Class ==========================
class KalmanFilter:
    """Simple 4-state (x,y,vx,vy) constant-velocity Kalman filter."""
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000.0          # initial uncertainty

        # State transition matrix (constant velocity)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix (position only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance — physically correct discrete-time Q for a
        # constant-velocity model: couples position and velocity noise so that
        # position uncertainty correctly grows from integrated velocity uncertainty.
        self.Q = process_noise * np.array([
            [dt**3/3, 0,       dt**2/2, 0      ],
            [0,       dt**3/3, 0,       dt**2/2],
            [dt**2/2, 0,       dt,      0      ],
            [0,       dt**2/2, 0,       dt     ]
        ])

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise**2

    def predict(self):
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update step with measurement z (2x1 vector: x, y)."""
        y = z - self.H @ self.x                # innovation
        S = self.H @ self.P @ self.H.T + self.R
        # Use solve instead of inv for better numerical stability
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# ========================== Simulation Parameters ==========================
dt = 0.01                     # time step [s]
duration = 20.0               # simulation duration [s]
steps = int(duration / dt)

# Interceptor parameters
interceptor_speed = 200.0     # constant speed [m/s]
max_accel = 50.0              # maximum acceleration [m/s^2]
interceptor_pos = np.array([0.0, 0.0])
interceptor_vel = np.array([interceptor_speed, 0.0])   # initial heading east

# Target parameters (true motion)
target_pos = np.array([1000.0, 0.0])
target_vel = np.array([-100.0, 50.0])                  # moving roughly toward interceptor
target_accel = np.array([0.0, 0.0])
maneuver_amplitude = 5.0      # [m/s^2] for weaving

# Sensor noise
meas_sigma = 10.0              # standard deviation of position measurement [m]

# Kalman filter setup
process_noise = 5.0            # process noise intensity (for unknown target maneuvers)
kf = KalmanFilter(dt, process_noise, meas_sigma)

# Initial Kalman state guess (based on first measurement)
initial_meas = target_pos + np.random.normal(0, meas_sigma, 2)
kf.x[:2, 0] = initial_meas     # initial position estimate
kf.x[2:, 0] = [0.0, 0.0]       # initial velocity guess zero

# Intercept condition
hit_distance = 5.0             # [m]

# Data storage for plotting
history = {
    't': [0.0],
    'target_true': [target_pos.copy()],
    'target_est': [kf.x[:2, 0].flatten().copy()],
    'interceptor': [interceptor_pos.copy()],
}
hit = False
sim_time = 0.0

# ========================== Simulation Loop ==========================
for step in range(1, steps + 1):
    if hit:
        break

    sim_time = step * dt

    # -------- True target dynamics (with weaving maneuver) ----------
    target_accel = np.array([0.0, maneuver_amplitude * np.sin(0.5 * sim_time)])
    target_vel += target_accel * dt
    target_pos += target_vel * dt

    # -------- Interceptor guidance (pure pursuit with acceleration limit) -----
    target_est_pos = kf.x[:2, 0].flatten()

    desired_vel = target_est_pos - interceptor_pos
    if np.linalg.norm(desired_vel) > 0:
        desired_vel = (desired_vel / np.linalg.norm(desired_vel)) * interceptor_speed

    acc = (desired_vel - interceptor_vel) / dt
    if np.linalg.norm(acc) > max_accel:
        acc = acc / np.linalg.norm(acc) * max_accel

    interceptor_vel += acc * dt
    interceptor_pos += interceptor_vel * dt

    # -------- Sensor measurement (noisy absolute position of target) -----
    z = target_pos + np.random.normal(0, meas_sigma, 2)

    # -------- Kalman filter update -----
    kf.predict()
    kf.update(z.reshape(2, 1))

    # -------- Check for intercept -----
    if np.linalg.norm(interceptor_pos - target_pos) < hit_distance:
        hit = True

    # -------- Store data (downsample to every 5th step for animation) -----
    if step % 5 == 0 or hit:
        history['t'].append(sim_time)
        history['target_true'].append(target_pos.copy())
        history['target_est'].append(kf.x[:2, 0].flatten().copy())
        history['interceptor'].append(interceptor_pos.copy())

# Convert history to numpy arrays for easier plotting
t_arr = np.array(history['t'])
target_true_arr = np.array(history['target_true'])
target_est_arr = np.array(history['target_est'])
interceptor_arr = np.array(history['interceptor'])

# ========================== Animation ==========================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-200, 1200)
ax.set_ylim(-200, 800)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Real-Time Kinematic Interceptor Simulator\nwith Kalman Filter Tracking')
ax.grid(True)

# Plot elements
traj_true, = ax.plot([], [], 'b-', lw=1.5, label='Target (true)')
traj_est, = ax.plot([], [], 'r--', lw=1.5, label='Target (Kalman est.)')
traj_inter, = ax.plot([], [], 'g-', lw=1.5, label='Interceptor')
point_true, = ax.plot([], [], 'bo', markersize=6)
point_est, = ax.plot([], [], 'rs', markersize=6)
point_inter, = ax.plot([], [], 'g^', markersize=8)

hit_circle = Circle((0, 0), hit_distance, fill=False, color='gray', linestyle=':')
ax.add_patch(hit_circle)

def init():
    traj_true.set_data([], [])
    traj_est.set_data([], [])
    traj_inter.set_data([], [])
    point_true.set_data([], [])
    point_est.set_data([], [])
    point_inter.set_data([], [])
    hit_circle.center = (0, 0)
    return traj_true, traj_est, traj_inter, point_true, point_est, point_inter, hit_circle

def animate(i):
    idx = min(i, len(t_arr) - 1)

    traj_true.set_data(target_true_arr[:idx, 0], target_true_arr[:idx, 1])
    traj_est.set_data(target_est_arr[:idx, 0], target_est_arr[:idx, 1])
    traj_inter.set_data(interceptor_arr[:idx, 0], interceptor_arr[:idx, 1])

    point_true.set_data([target_true_arr[idx, 0]], [target_true_arr[idx, 1]])
    point_est.set_data([target_est_arr[idx, 0]], [target_est_arr[idx, 1]])
    point_inter.set_data([interceptor_arr[idx, 0]], [interceptor_arr[idx, 1]])

    hit_circle.center = (interceptor_arr[idx, 0], interceptor_arr[idx, 1])

    return traj_true, traj_est, traj_inter, point_true, point_est, point_inter, hit_circle

# Interval set to 50 ms (20 fps) for reliable cross-platform playback.
# Frames are already downsampled 5x above, so wall-clock time stays close
# to real-time (5 * 10 ms per sim step = 50 ms per frame).
ani = animation.FuncAnimation(fig, animate, frames=len(t_arr), init_func=init,
                              interval=50, blit=True, repeat=False)

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Print final result
if hit:
    print(f"Intercept achieved at t = {sim_time:.2f} s")
else:
    print("Simulation ended without intercept.")
