# import numpy as np

class UAVRewardShaping:
    def __init__(self, world_size=10.0):
        self.world_size = world_size

    def compute_reward(
        self,
        progress,
        action,
        lidar_readings,
        collision,
        reached_goal,
        speed=0.0,
        omega=0.0,
        full_spin_penalty=0.0
    ):

        if collision:
            return -1000.0

        if reached_goal:
            return 1000.0

        reward = 0.0

        # Progress reward
        reward += 5.0 * float(progress)

        # Time penalty
        reward -= 0.005

        # if action in [3, 4]:
        #     reward -= 0.10  # Actuator selection penalty
            
        # reward -= 0.05 * abs(float(omega))  # Kinetic angular velocity penalty

        # if abs(omega) > 5:
        #     reward -= 1.0

        return float(reward)