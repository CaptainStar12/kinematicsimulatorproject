Real-Time Kinematic Interceptor Simulator
A 2D physics simulation of a guided interceptor pursuing a maneuvering target. This uses a Kalman filter providing real-time state estimation from noisy sensor measurements. Produces an animated visualization showing the true target trajectory, the Kalman filter's estimate, and the interceptor's pursuit path.

What It Does
Simulates a missile intercept scenario where:

A target moves with constant velocity plus a sinusoidal weaving maneuver
A sensor measures the target's position with Gaussian noise
A Kalman filter fuses the noisy measurements to estimate the target's true position and velocity in real time
An interceptor uses the Kalman estimate to steer toward the predicted target location using a pure pursuit guidance law with an acceleration limit

The simulation runs for up to 20 seconds or until intercept (within 5 meters).

Kalman Filter Design
The filter tracks a 4-dimensional state vector:
x = [x_position, y_position, x_velocity, y_velocity]
State transition model (constant velocity):
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
Measurement model (position only — velocity is not directly observed):
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
Process noise covariance Q uses the physically correct discrete-time form for a constant-velocity model. This couples position and velocity uncertainty so that position uncertainty grows correctly from integrated velocity uncertainty over time:
Q = q * [[dt³/3,  0,      dt²/2,  0    ],
         [0,      dt³/3,  0,      dt²/2],
         [dt²/2,  0,      dt,     0    ],
         [0,      dt²/2,  0,      dt   ]]
Key design choices:

process_noise = 5.0 — accounts for unknown target maneuvers. Higher values make the filter more responsive to maneuvers but noisier. Lower values produce smoother estimates but lag during sudden maneuvers.
measurement_noise = 10.0 m — standard deviation of the sensor position measurement
Kalman gain computed via np.linalg.solve rather than matrix inversion for numerical stability


Guidance Law
The interceptor uses pure pursuit with an acceleration limit:

Compute the desired velocity direction toward the Kalman-estimated target position
Scale to constant interceptor speed (200 m/s)
Compute required acceleration to achieve that velocity
Clip acceleration to max_accel = 50 m/s²
Integrate to update interceptor velocity and position

Pure pursuit is a simple and historically significant guidance law — the interceptor always points directly at the estimated target position. It is not optimal (proportional navigation is more efficient) but it is intuitive and sufficient for this simulation.

Target Motion
The target starts at (1000, 0) moving toward the interceptor with an initial velocity of (-100, +50) m/s. It performs a sinusoidal weaving maneuver:
target_accel_y = 5.0 * sin(0.5 * t)    [m/s²]
This lateral maneuver tests the Kalman filter's ability to track a non-constant-velocity target. The process noise parameter q controls how well the filter adapts to this unmodeled acceleration.

Simulation Parameters
Parameter          Value          Description
dt                 0.01 s         Time step
duration           20.0 s         Maximum simulation time
interceptor_speed  200 m/s        Constant interceptor 
max_accel          50 m/s²        Interceptor acceleration limit
target_vel        (-100, 50) m/s  Initial target velocity
maneuver_amplitude 5.0 m/s²       Lateral weaving amplitude
meas_sigma         10.0 m         Sensor position noise
process_noise      5.0            Kalman process noise intensity
hit_distance       5.0 m          Intercept radius       

Installation
bashpip install numpy matplotlib
python interceptor_sim.py

Animation
The visualization shows three trajectories in real time:

Blue solid — true target trajectory
Red dashed — Kalman filter estimate of target trajectory
Green solid — interceptor trajectory
Gray circle — intercept radius around the interceptor's current position

The animation runs at approximately real-time speed (20 fps, downsampled 5× from the simulation timestep).

Key Concepts Demonstrated
State estimation under noise: The Kalman filter recovers smooth position and velocity estimates from measurements corrupted by 10-meter Gaussian noise. The estimated trajectory (red) closely tracks the true trajectory (blue) despite the noise.
Velocity estimation from position measurements: The filter infers target velocity from the sequence of position measurements even though velocity is never directly measured. This is the core value of the Kalman filter over simple averaging.
Tracking a maneuvering target: The sinusoidal weaving introduces unmodeled acceleration. The process noise parameter q acts as a tuning knob. Increase it to track aggressive maneuvers faster, decrease it to reduce noise in the estimate during straight flight.
Guidance under estimation error: The interceptor steers toward the estimated position rather than the true position. Errors in the Kalman estimate directly affect intercept geometry. This demonstrates why state estimation quality matters for real guidance systems.

Limitations

Constant-velocity motion model. A constant-acceleration or Singer model would better handle the sinusoidal maneuver.
Pure pursuit guidance is not optimal. Proportional navigation (N = 4 gain) is standard in real systems and significantly more efficient in terms of control effort.
No seeker field-of-view constraint. A real missile seeker has a limited angular range and loses lock if the target moves outside it.
No missile aerodynamic model. The interceptor is treated as a point mass with instantaneous velocity changes, ignoring drag, lift, and airframe dynamics.
2D only. Real intercept problems are 3D.
