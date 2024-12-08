import gym
import numpy as np
import pygame
import math
import random

class AGVEnvironment(gym.Env):
    def __init__(self):
        super(AGVEnvironment, self).__init__()

        # Field dimensions (15x15 meters)
        self.field_size = 15
        self.resolution = 50
        self.window_size = int(self.field_size * self.resolution)

        # AGV parameters
        self.agv_length = 0.7
        self.agv_width = 0.5

        # Define possible actions
        self.turn_angles = [5, 10, 15, 30, 45]  # Turn by these angles
        self.move_steps = [0.01, 0.05, 0.1, 0.25, 1]  # Move by these distances
        self.num_turn_actions = len(self.turn_angles) * 2  # Left and right turns
        self.num_move_actions = len(self.move_steps)       # Forward movements
        self.action_space = gym.spaces.Discrete(self.num_turn_actions + self.num_move_actions)

        # Expanded sensor angles to get better view in front
        # Front area: -10, 0, 10 degrees. This will help determine if the path ahead is clear.
        self.sensor_angles = [-90, -45, -10, 0, 10, 45, 90]
        self.num_sensors = len(self.sensor_angles)
        self.max_sensor_distance = 5.0  # Maximum sensor range in meters

        self.obstacles = []

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0] + [0] * self.num_sensors),
            high=np.array([self.field_size, self.field_size, 360] + [self.max_sensor_distance] * self.num_sensors),
            dtype=np.float32,
        )

        # Initial position and goal
        self.start_pos = np.array([0.5, 7.5, 0])
        self.goal_pos = np.array([14.0, 14.0])
        self.agv_pos = self.start_pos.copy()

        self.turn_only_steps = 0

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("AGV Environment")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (230, 230, 230)
        self.grid_color = (200, 200, 200)
        self.agv_color = (0, 0, 255)
        self.goal_color = (0, 255, 0)
        self.obstacle_color = (255, 0, 0)
        self.sensor_color = (255, 165, 0)

    def reset(self):
        self.agv_pos = self.start_pos.copy()

        # Randomize obstacles
        self.obstacles = []
        num_obstacles = random.randint(5, 15)
        for _ in range(num_obstacles):
            obs_x = random.uniform(1, self.field_size - 1)
            obs_y = random.uniform(1, self.field_size - 1)
            width = random.uniform(0.3, 1.0)
            height = random.uniform(0.3, 1.0)

            dist_to_start = math.hypot(obs_x - self.start_pos[0], obs_y - self.start_pos[1])
            dist_to_goal = math.hypot(obs_x - self.goal_pos[0], obs_y - self.goal_pos[1])

            # Retry if too close to start or goal
            if dist_to_start < 1.0 or dist_to_goal < 1.0:
                continue

            self.obstacles.append({
                "type": "rectangle",
                "position": (obs_x, obs_y),
                "size": (width, height)
            })

        sensor_readings = self.get_sensor_readings()
        observation = np.concatenate((self.agv_pos, sensor_readings))
        self.turn_only_steps = 0
        return observation

    def step(self, action):
        previous_distance = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)
        previous_obstacle_distance = self.min_distance_to_obstacles()

        # Apply action
        if action < self.num_turn_actions:  # Turn
            direction = -1 if action % 2 == 0 else 1
            angle_idx = action // 2
            self.agv_pos[2] = (self.agv_pos[2] + direction * self.turn_angles[angle_idx]) % 360
            chosen_step = 0.0
            self.turn_only_steps += 1
        else:  # Move forward
            move_idx = action - self.num_turn_actions
            chosen_step = self.move_steps[move_idx]
            dx = chosen_step * math.cos(math.radians(self.agv_pos[2]))
            dy = chosen_step * math.sin(math.radians(self.agv_pos[2]))
            self.agv_pos[0] += dx
            self.agv_pos[1] += dy
            self.turn_only_steps = 0

        # Keep in bounds
        self.agv_pos[0] = np.clip(self.agv_pos[0], 0, self.field_size)
        self.agv_pos[1] = np.clip(self.agv_pos[1], 0, self.field_size)

        # Check collision
        collided = self.check_collision()

        # Distances
        current_distance = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)
        distance_improvement = previous_distance - current_distance
        current_obstacle_distance = self.min_distance_to_obstacles()
        obstacle_distance_improvement = current_obstacle_distance - previous_obstacle_distance

        # Check goal
        done = current_distance < 0.5

        # Get sensor readings
        sensor_readings = self.get_sensor_readings()

        # Identify front sensors for checking obstacles ahead (-10, 0, 10)
        front_sensor_indices = [self.sensor_angles.index(a) for a in [-10, 0, 10]]
        front_distances = sensor_readings[front_sensor_indices]
        min_front_distance = np.min(front_distances)

        reward = 0.0
        if collided:
            # Strong penalty for collision
            reward = -300.0
            done = True
        elif done:
            # Reached goal
            reward = 100.0
            # Also add some bonus for good obstacle avoidance or quick approach
            reward += 5.0 * distance_improvement
            reward += max(0, obstacle_distance_improvement) * 2.0
        else:
            # Distance-based reward:
            if distance_improvement > 0:
                # If front is clear, we add an extra bonus for larger forward steps
                if chosen_step > 0 and min_front_distance > 2.0:
                    # More bonus if we can move freely
                    reward += 5.0 * distance_improvement * (1.0 + chosen_step * 2.0)
                else:
                    reward += 5.0 * distance_improvement
            else:
                # Slight penalty if moving away from goal
                reward += 2.0 * distance_improvement  # negative

            # Obstacle avoidance:
            # Reward getting farther from obstacles
            if obstacle_distance_improvement > 0:
                # Encourage getting away from obstacles
                reward += 2.0 * obstacle_distance_improvement * (1 + max(0, distance_improvement))
            else:
                # Penalize getting closer to obstacles
                closeness_penalty_factor = 1.0
                if current_obstacle_distance < 0.5:
                    closeness_penalty_factor = 5.0
                elif current_obstacle_distance < 1.0:
                    closeness_penalty_factor = 2.0
                reward += closeness_penalty_factor * obstacle_distance_improvement  # negative

            # Additional penalty if very close to obstacles continuously
            if current_obstacle_distance < 1.0:
                reward -= (1.0 - current_obstacle_distance) * 0.5

            # Small step penalty to discourage random moves without purpose
            reward -= 0.005

            # Penalty for too many turns in place
            if self.turn_only_steps > 3:
                reward -= 0.1 * self.turn_only_steps

            # If moving forward with clear front and improving distance to goal, small bonus
            if chosen_step > 0 and distance_improvement > 0 and min_front_distance > 2.0:
                reward += 0.1  # Encourage fast forward motion in clear path

        observation = np.concatenate((self.agv_pos, sensor_readings))
        return observation, reward, done, {}

    def get_sensor_readings(self):
        agv_x, agv_y, agv_angle = self.agv_pos
        sensor_readings = []

        for angle in self.sensor_angles:
            sensor_angle = (agv_angle + angle) % 360
            sensor_rad = math.radians(sensor_angle)

            distance = self.max_sensor_distance
            for d in np.linspace(0, self.max_sensor_distance, num=100):
                test_x = agv_x + d * math.cos(sensor_rad)
                test_y = agv_y + d * math.sin(sensor_rad)

                # Check bounds
                if test_x < 0 or test_x > self.field_size or test_y < 0 or test_y > self.field_size:
                    distance = d
                    break

                # Check obstacle collision
                collision = False
                for obstacle in self.obstacles:
                    if obstacle["type"] == "rectangle":
                        obs_x, obs_y = obstacle["position"]
                        width, height = obstacle["size"]
                        half_width, half_height = width / 2, height / 2
                        if (
                            obs_x - half_width <= test_x <= obs_x + half_width
                            and obs_y - half_height <= test_y <= obs_y + half_height
                        ):
                            collision = True
                            break
                if collision:
                    distance = d
                    break

            sensor_readings.append(distance)

        return np.array(sensor_readings, dtype=np.float32)

    def check_collision(self):
        agv_x, agv_y, agv_angle = self.agv_pos
        half_length = self.agv_length / 2
        half_width = self.agv_width / 2
        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width),
        ]
        rotated_corners = [
            (
                agv_x + x * math.cos(math.radians(agv_angle)) - y * math.sin(math.radians(agv_angle)),
                agv_y + x * math.sin(math.radians(agv_angle)) + y * math.cos(math.radians(agv_angle)),
            )
            for x, y in corners
        ]

        # Check field boundaries
        for cx, cy in rotated_corners:
            if cx < 0 or cx > self.field_size or cy < 0 or cy > self.field_size:
                return True

        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle["type"] == "rectangle":
                obs_x, obs_y = obstacle["position"]
                width, height = obstacle["size"]
                hw, hh = width / 2, height / 2
                for cx, cy in rotated_corners:
                    if (obs_x - hw <= cx <= obs_x + hw) and (obs_y - hh <= cy <= obs_y + hh):
                        return True

        return False

    def min_distance_to_obstacles(self):
        agv_x, agv_y, _ = self.agv_pos
        min_dist = float('inf')
        for obstacle in self.obstacles:
            if obstacle["type"] == "rectangle":
                obs_x, obs_y = obstacle["position"]
                width, height = obstacle["size"]
                half_w, half_h = width/2, height/2

                closest_x = np.clip(agv_x, obs_x - half_w, obs_x + half_w)
                closest_y = np.clip(agv_y, obs_y - half_h, obs_y + half_h)
                dist = math.hypot(agv_x - closest_x, agv_y - closest_y)
                if dist < min_dist:
                    min_dist = dist

        return min_dist if min_dist != float('inf') else self.max_sensor_distance

    def render(self, mode="human"):
        self.screen.fill(self.bg_color)

        # Draw grid
        for x in range(0, self.window_size, self.resolution):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.window_size))
            pygame.draw.line(self.screen, self.grid_color, (0, x), (self.window_size, x))

        # Draw goal
        goal_rect = pygame.Rect(
            int(self.goal_pos[0] * self.resolution) - 5,
            int(self.goal_pos[1] * self.resolution) - 5,
            10,
            10,
        )
        pygame.draw.rect(self.screen, self.goal_color, goal_rect)

        # Draw obstacles
        for obstacle in self.obstacles:
            if obstacle["type"] == "rectangle":
                obs_x, obs_y = obstacle["position"]
                width, height = obstacle["size"]
                pygame.draw.rect(
                    self.screen,
                    self.obstacle_color,
                    pygame.Rect(
                        int((obs_x - width / 2) * self.resolution),
                        int((obs_y - height / 2) * self.resolution),
                        int(width * self.resolution),
                        int(height * self.resolution),
                    ),
                )

        # Draw AGV
        agv_x = int(self.agv_pos[0] * self.resolution)
        agv_y = int(self.agv_pos[1] * self.resolution)
        agv_angle = self.agv_pos[2]
        half_length = self.agv_length / 2 * self.resolution
        half_width = self.agv_width / 2 * self.resolution
        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width),
        ]
        rotated_corners = [
            (
                agv_x + x * math.cos(math.radians(agv_angle)) - y * math.sin(math.radians(agv_angle)),
                agv_y + x * math.sin(math.radians(agv_angle)) + y * math.cos(math.radians(agv_angle)),
            )
            for x, y in corners
        ]
        pygame.draw.polygon(self.screen, self.agv_color, rotated_corners)

        # Draw sensors
        sensor_readings = self.get_sensor_readings()
        for angle, distance in zip(self.sensor_angles, sensor_readings):
            sensor_angle = (agv_angle + angle) % 360
            end_x = agv_x + distance * self.resolution * math.cos(math.radians(sensor_angle))
            end_y = agv_y + distance * self.resolution * math.sin(math.radians(sensor_angle))
            pygame.draw.line(
                self.screen,
                self.sensor_color,
                (agv_x, agv_y),
                (end_x, end_y),
                1,
            )

        pygame.display.flip()

    def close(self):
        pygame.quit()
