import numpy as np
import matplotlib.pyplot as plt
from env.dynamics import AdvancedUAVDynamics
from env.rewards_riskaware import UAVRewardShaping

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class UAVLiDAREnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_size=10.0,
        n_lidar=64,
        max_steps=3000,
        dt=0.1,
        drag=0.2,
        thrust=1.0,
        max_speed=1.2,
        goal_radius=0.6,
        collision_radius=0.25,
        n_obstacles=2,
        obstacle_radius_range=(0.3, 0.8),
        seed=None,
        reward_mode="standard",
    ):
        if gym is None:
            raise ImportError("Install gymnasium first: pip install gymnasium")

        super().__init__()

        self.world_size = world_size
        self.n_lidar = n_lidar
        self.max_steps = max_steps
        self.dt = dt
        self.drag = drag
        self.thrust = thrust
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.collision_radius = collision_radius
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.reward_mode = reward_mode

        if self.reward_mode not in ["standard", "risk_aware"]:
            raise ValueError("reward_mode must be 'standard' or 'risk_aware'")

        self.rng = np.random.default_rng(seed)

        try:
            self.dynamics = AdvancedUAVDynamics(rng=self.rng)
        except TypeError:
            self.dynamics = AdvancedUAVDynamics()
            if hasattr(self.dynamics, "set_rng"):
                self.dynamics.set_rng(self.rng)

        self.reward_shaper = UAVRewardShaping(world_size=self.world_size)

        self.theta = 0.0
        self.omega = 0.0

        self.action_space = spaces.Discrete(5)

        obs_dim = self.n_lidar + 5
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.pos = None
        self.vel = None
        self.goal = None
        self.obstacles = None
        self.steps = 0
        self.prev_distance = None
        self.prev_action = 0
        self.trajectory = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if hasattr(self.dynamics, "set_rng"):
                self.dynamics.set_rng(self.rng)

        self.steps = 0
        self.theta = 0.0
        self.omega = 0.0
        self.vel = np.zeros(2, dtype=np.float32)
        self.prev_action = 0

        self.pos = self.rng.uniform(1.0, 3.0, size=2).astype(np.float32)
        self.goal = self.rng.uniform(
            self.world_size - 3.0,
            self.world_size - 1.0,
            size=2
        ).astype(np.float32)

        self.obstacles = self._generate_obstacles()
        self.prev_distance = self._distance_to_goal()
        self.trajectory = [self.pos.copy()]

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        self.pos, self.vel, self.theta, self.omega, accel = self.dynamics.update_physics(
            self.pos,
            self.vel,
            self.theta,
            self.omega,
            action
        )

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / (speed + 1e-8) * self.max_speed

        self.pos = np.clip(self.pos, 0.0, self.world_size)
        self.trajectory.append(self.pos.copy())

        distance = self._distance_to_goal()
        progress = self.prev_distance - distance
        self.prev_distance = distance

        lidar = self._lidar_scan()
        speed = float(np.linalg.norm(self.vel))
        omega = float(self.omega)

        collision = self._check_collision()
        reached_goal = distance <= self.goal_radius
        timeout = self.steps >= self.max_steps

        if self.reward_mode == "standard":
            reward = self._standard_reward(
                progress=progress,
                collision=collision,
                reached_goal=reached_goal
            )
        else:
            reward = self.reward_shaper.compute_reward(
                progress=progress,
                action=action,
                lidar_readings=lidar,
                collision=collision,
                reached_goal=reached_goal,
                speed=speed,
                omega=omega
            )

        energy_use = 1.0 if action in [1, 2] else (0.2 if action in [3, 4] else 0.0)
        smoothness_penalty = 1.0 if action != self.prev_action else 0.0

        self.prev_action = action

        terminated = collision or reached_goal
        truncated = timeout

        info = {
            "distance_to_goal": float(distance),
            "progress": float(progress),
            "collision": bool(collision),
            "reached_goal": bool(reached_goal),
            "speed": float(speed),
            "omega": float(omega),
            "energy_use": float(energy_use),
            "smoothness_violation": float(smoothness_penalty),
            "raw_lidar": lidar,
            "min_lidar_distance": float(np.min(lidar)) * self.world_size,
            "reward_mode": self.reward_mode,
            "n_obstacles": self.n_obstacles,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _standard_reward(self, progress, collision, reached_goal):
        """
        Standard baseline reward.
        No risk-aware LiDAR shaping.
        No angular velocity penalty.
        No landing speed constraint.
        """

        if collision:
            return -1000.0

        if reached_goal:
            return 1000.0

        reward = 0.0
        reward += 5.0 * float(progress)
        reward -= 0.005

        return float(reward)

    def _get_obs(self):
        lidar = self._lidar_scan()

        vx = self.vel[0] / self.max_speed
        vy = self.vel[1] / self.max_speed

        norm_theta = self.theta / np.pi
        norm_omega = self.omega / np.pi

        target_vec = self.goal - self.pos
        target_angle = np.arctan2(target_vec[1], target_vec[0]) / np.pi

        obs = np.concatenate([
            lidar,
            np.array(
                [vx, vy, norm_theta, norm_omega, target_angle],
                dtype=np.float32
            )
        ])

        return obs.astype(np.float32)

    def _lidar_scan(self):
        angles = np.linspace(0, 2 * np.pi, self.n_lidar, endpoint=False)
        max_range = self.world_size
        readings = np.ones(self.n_lidar, dtype=np.float32)

        for i, angle in enumerate(angles):
            direction = np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32
            )

            min_dist = max_range

            for center, radius in self.obstacles:
                dist = self._ray_circle_distance(
                    self.pos,
                    direction,
                    center,
                    radius
                )

                if dist is not None:
                    min_dist = min(min_dist, dist)

            wall_dist = self._ray_wall_distance(self.pos, direction)
            min_dist = min(min_dist, wall_dist)

            readings[i] = np.clip(min_dist / max_range, 0.0, 1.0)

        return readings

    def _ray_circle_distance(self, origin, direction, center, radius):
        oc = origin - center
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius ** 2
        discriminant = b ** 2 - 4 * c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)

        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        valid = [t for t in [t1, t2] if t >= 0]

        return min(valid) if valid else None

    def _ray_wall_distance(self, origin, direction):
        distances = []

        if direction[0] > 1e-6:
            distances.append((self.world_size - origin[0]) / direction[0])
        elif direction[0] < -1e-6:
            distances.append((0.0 - origin[0]) / direction[0])

        if direction[1] > 1e-6:
            distances.append((self.world_size - origin[1]) / direction[1])
        elif direction[1] < -1e-6:
            distances.append((0.0 - origin[1]) / direction[1])

        distances = [d for d in distances if d >= 0]

        return min(distances) if distances else self.world_size

    def _generate_obstacles(self):
        obstacles = []
        start = np.array([1.0, 1.0])
        goal = np.array([self.world_size - 1.0, self.world_size - 1.0])

        for _ in range(self.n_obstacles):
            for _attempt in range(100):
                radius = self.rng.uniform(*self.obstacle_radius_range)

                center = self.rng.uniform(
                    radius,
                    self.world_size - radius,
                    size=2
                )

                too_close_start = np.linalg.norm(center - start) < 1.5
                too_close_goal = np.linalg.norm(center - goal) < 1.5

                if not too_close_start and not too_close_goal:
                    obstacles.append(
                        (center.astype(np.float32), float(radius))
                    )
                    break

        return obstacles

    def _distance_to_goal(self):
        return float(np.linalg.norm(self.goal - self.pos))

    def _check_collision(self):
        if np.any(self.pos <= 0.0) or np.any(self.pos >= self.world_size):
            return True

        for center, radius in self.obstacles:
            if np.linalg.norm(self.pos - center) <= radius + self.collision_radius:
                return True

        return False

    def render(self):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for center, radius in self.obstacles:
            circle = plt.Circle(center, radius, alpha=0.4)
            ax.add_patch(circle)

        goal_circle = plt.Circle(self.goal, self.goal_radius, alpha=0.6)
        ax.add_patch(goal_circle)
        ax.text(self.goal[0], self.goal[1] + 0.4, "Goal", ha="center")

        uav_circle = plt.Circle(self.pos, self.collision_radius, alpha=0.9)
        ax.add_patch(uav_circle)
        ax.text(self.pos[0], self.pos[1] + 0.35, "UAV", ha="center")

        trajectory = np.array(self.trajectory)

        if len(trajectory) > 1:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=2,
                alpha=0.7
            )

        plt.show()