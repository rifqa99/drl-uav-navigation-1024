    
import numpy as np
import math


class AdvancedUAVDynamics:
    def __init__(self, rng=None):
        # Kinematic state parameters
        self.dt = 0.1
        self.mass = 1.5          # Total mass of the quadcopter (kg)
        self.I_z = 0.05          # Rotational moment of inertia (kg*m^2)
        self.drag_linear = 0.05  # Linear aerodynamic drag coefficient (lambda)
        self.drag_angular = 0.1  # Angular drag coefficient
        self.arm_length = 0.25   # Distance from center to rotors (m)

        # Ambient continuous wind vector
        self.wind_speed = np.array([0.5, -0.2])
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_rng(self, rng):
        self.rng = rng

    def update_physics(self, pos, vel, theta, omega, action):
        """
        Updates the rigid body state using forces, torques, and external wind.
        Actions: 0=Hover, 1=Forward Thrust, 2=Reverse Thrust, 3=Clockwise Torque, 4=Counter-Clockwise Torque
        """
        thrust = 0.0
        torque = 0.0

        if action == 1:
            thrust = 3.0    # Linear forward force
        elif action == 2:
            thrust = -1.5  # Linear breaking force
        elif action == 3:
            torque = 0.25   # Clockwise rotational torque
        elif action == 4:
            torque = -0.25  # Counter-clockwise rotational torque

        # 1. Stochastic Wind Noise Addition
        wind_noise = self.rng.normal(0.0, 0.1, size=(2,))
        total_wind = self.wind_speed + wind_noise


        # 2. Linear Second-Order Dynamics Equations
        # Thrust force vector resolution based on current heading angle (theta)
        f_x = thrust * math.cos(theta) + total_wind[0]
        f_y = thrust * math.sin(theta) + total_wind[1]

        # v_{t+1} = v_t + (F/m)*dt - lambda * v_t
        accel_x = (f_x / self.mass) - self.drag_linear * vel[0]
        accel_y = (f_y / self.mass) - self.drag_linear * vel[1]

        vel[0] += accel_x * self.dt
        vel[1] += accel_y * self.dt
        pos[0] += vel[0] * self.dt
        pos[1] += vel[1] * self.dt

        # 3. Rotational Dynamics Equations (Inertia)
        # omega_{t+1} = omega_t + (Torque/I_z)*dt - drag * omega_t
        alpha = (torque / self.I_z) - self.drag_angular * omega
        omega += alpha * self.dt
        theta += omega * self.dt

        # Normalize heading angle between [-pi, pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        return pos, vel, theta, omega, np.array([accel_x, accel_y])
