import numpy as np

class UAVRewardShaping:
    def __init__(self, world_size=10.0):
        self.world_size = world_size
        # The physical collision envelope of the quadcopter is 0.25 meters
        self.collision_radius = 0.25 
        # Early-warning boundary: exponential penalty triggers when entering this 1.2m zone
        self.warning_threshold = 1.2 

    def compute_reward(
        self,
        progress,
        action,
        lidar_readings,
        collision,
        reached_goal,
        speed=0.0,
        omega=0.0
    ):
        # 1. Catastrophic Terminal Triggers
        if collision:
            return -1000.0

        if reached_goal:
            terminal_reward = 1000.0
            safe_landing_speed = 0.4
            
            # Kinematic Braking Constraint for target touchdown stability
            if speed > safe_landing_speed:
                terminal_reward -= 200.0 * (speed - safe_landing_speed)
            return float(terminal_reward)

        reward = 0.0

        # 2. Potential-Based Progress & Decisiveness Constraints
        reward += 5.0 * float(progress)
        reward -= 0.005  # Standard frame time penalty

        # 3. Dual Rotational Smoothness Constraints (Anti-Spinning Fix)
        if action in [3, 4]:
            reward -= 0.20  # Actuator selection penalty
            
        reward -= 0.10 * abs(float(omega))  # Kinetic angular velocity penalty

        # 4. Proximity Risk Envelope (Smooth Exponential Penalty Field)
        if lidar_readings is not None and len(lidar_readings) > 0:
            # Extract minimum sensor clearance mapped into true meters
            min_lidar_m = float(np.min(lidar_readings)) * self.world_size
            
            # Risk warning activates well before a catastrophic physical hit occurs
            if min_lidar_m < self.warning_threshold:
                # Exponential penalty scales smoothly relative to localized hazard proximity.
                # The gradient increases smoothly as distance drops toward the collision radius.
                risk_penalty = 1.5 * np.exp(self.warning_threshold - min_lidar_m)
                reward -= risk_penalty

        return float(reward)