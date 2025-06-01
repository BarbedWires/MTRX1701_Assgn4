import numpy as np
import matplotlib.pyplot as plt
# Simulation parameters
dt = 0.1         # time step
T = 20           # total time (seconds)
N = int(T / dt)  # number of steps
# Bus state variables
x = 0.0                    # position along the path
y = -1.0                   # lateral error (start 1m below line)
theta = np.deg2rad(10)     # heading angle (10 degrees off)

# Desired path: y = 0 (a straight black line on the road)
desired_y = 0.0
desired_theta = 0.0
# Controller gains
Kp = 2.0       # proportional gain
Kd = 1.0       # derivative gain
Ki = 0.5       # integral gain (new)
Ktheta = 2.0   # heading correction gain

# Data storage for plotting
x_history = [x]
y_history = [y]
theta_history = [theta]
e_history = []
de_history = []
ie_history = []
# Initialize error terms
e_prev = y - desired_y
integral_e = 0.0

# Main simulation loop
for _ in range(N):
    # Errors
    e = y - desired_y
    de = (e - e_prev) / dt
    integral_e += e * dt
    theta_error = theta - desired_theta
    # PID control law
    delta = -Kp * e - Kd * de - Ki * integral_e - Ktheta * theta_error
    # vehicle kinematics 
    v = 1.0  # constant forward velocity
    L = 2.5  # wheelbase
    theta += (v / L) * np.tan(delta) * dt
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt

    # Store history
    x_history.append(x)
    y_history.append(y)
    theta_history.append(theta)
    e_history.append(e)
    de_history.append(de)
    ie_history.append(integral_e)
    # Update previous error
    e_prev = e
# --- Plotting ---
time = np.arange(0, T, dt)
plt.figure(figsize=(15, 5))
# Lateral Error
plt.subplot(1, 3, 1)
plt.plot(time, e_history)
plt.title("Lateral Error Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Lateral Error (m)")
plt.grid(True)

# Heading Error
plt.subplot(1, 3, 2)
theta_errors = [np.rad2deg(t - desired_theta) for t in theta_history[:-1]]
plt.plot(time, theta_errors)
plt.title("Heading Error Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Heading Error (degrees)")
plt.grid(True)

# Bus Trajectory
plt.subplot(1, 3, 3)
plt.plot(x_history, y_history, label="Bus Path")
plt.axhline(y=0.0, color='r', linestyle='--', label="Desired Line (y=0)")
plt.title("2D Trajectory of Bus")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('Ideal Conditions.png')
plt.show()
