import gym
import numpy as np
import pygame
import math


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
        self.move_step = 0.05  # Increased for better movement
        self.turn_step = 5.0   # Increased for smoother turning

        # Sensor parameters
        self.num_sensors = 5
        self.sensor_angles = [-90, -45, 0, 45, 90]  # Relative angles in degrees
        self.max_sensor_distance = 5.0  # Maximum sensor range in meters

        # Custom obstacles: Each obstacle has type, position, and size
        self.obstacles = [
            {"type": "circle", "position": (5, 5), "radius": 0.5},  # Circle obstacle
            {"type": "rectangle", "position": (10, 10), "size": (1.0, 0.5)},  # Rectangle obstacle
            {"type": "rectangle", "position": (7, 8), "size": (0.5, 0.5)},  # Rectangle obstacle
        ]

        # Action space and observation space
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0] + [0]*self.num_sensors),
            high=np.array([self.field_size, self.field_size, 360] + [self.max_sensor_distance]*self.num_sensors),
            dtype=np.float32,
        )

        # Initial position and goal
        self.start_pos = np.array([0.5, 7.5, 0])
        self.goal_pos = np.array([14.0, 14.0])
        self.agv_pos = self.start_pos.copy()

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("AGV Environment")
        self.clock = pygame.time.Clock()

        # Colors for visualization
        self.bg_color = (230, 230, 230)
        self.grid_color = (200, 200, 200)
        self.agv_color = (0, 0, 255)
        self.goal_color = (0, 255, 0)
        self.obstacle_color = (255, 0, 0)
        self.sensor_color = (255, 165, 0)

    def reset(self):
        self.agv_pos = self.start_pos.copy()
        sensor_readings = self.get_sensor_readings()
        observation = np.concatenate((self.agv_pos, sensor_readings))
        return observation

    def step(self, action):
        """
        Execute an action and update the environment.

        Args:
            action (int): 0 = move forward, 1 = turn left, 2 = turn right

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Save the previous distance to the goal
        previous_distance = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)

        # Apply the action
        if action == 0:  # Move forward
            dx = self.move_step * math.cos(math.radians(self.agv_pos[2]))
            dy = self.move_step * math.sin(math.radians(self.agv_pos[2]))
            self.agv_pos[0] += dx
            self.agv_pos[1] += dy
        elif action == 1:  # Turn left
            self.agv_pos[2] = (self.agv_pos[2] - self.turn_step) % 360
        elif action == 2:  # Turn right
            self.agv_pos[2] = (self.agv_pos[2] + self.turn_step) % 360

        # Ensure the AGV stays within the field bounds
        self.agv_pos[0] = np.clip(self.agv_pos[0], 0, self.field_size)
        self.agv_pos[1] = np.clip(self.agv_pos[1], 0, self.field_size)

        # Check for collision with obstacles
        if self.check_collision():
            reward = -100  # Large penalty for collision
            done = True
            sensor_readings = self.get_sensor_readings()
            observation = np.concatenate((self.agv_pos, sensor_readings))
            return observation, reward, done, {}

        # Calculate the current distance to the goal
        current_distance = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)

        # Calculate distance-based reward/penalty
        distance_reward = previous_distance - current_distance
        if distance_reward > 0:
            distance_reward = 1.0  # Reward for moving closer
        elif distance_reward < 0:
            distance_reward = -1.0  # Penalty for moving farther
        else:
            distance_reward = 0

        # Check if the AGV reached the goal
        done = current_distance < 0.5  # Goal reached if within 0.5 meters
        if done:
            reward = 100  # Large reward for reaching the goal
        else:
            # Small time penalty to encourage efficiency
            reward = distance_reward - 0.1

        # Get sensor readings
        sensor_readings = self.get_sensor_readings()

        # Return observation
        observation = np.concatenate((self.agv_pos, sensor_readings))
        return observation, reward, done, {}

    def get_sensor_readings(self):
        agv_x, agv_y, agv_angle = self.agv_pos
        sensor_readings = []

        for angle in self.sensor_angles:
            # Convert sensor angle to global angle
            sensor_angle = (agv_angle + angle) % 360
            sensor_rad = math.radians(sensor_angle)

            # Initialize distance
            distance = self.max_sensor_distance

            # Step along the sensor line
            for d in np.linspace(0, self.max_sensor_distance, num=100):
                test_x = agv_x + d * math.cos(sensor_rad)
                test_y = agv_y + d * math.sin(sensor_rad)

                # Check if out of bounds
                if test_x < 0 or test_x > self.field_size or test_y < 0 or test_y > self.field_size:
                    distance = d
                    break

                # Check collision with obstacles
                collision = False
                for obstacle in self.obstacles:
                    if obstacle["type"] == "circle":
                        obs_x, obs_y = obstacle["position"]
                        radius = obstacle["radius"]
                        if math.hypot(test_x - obs_x, test_y - obs_y) <= radius:
                            collision = True
                            break
                    elif obstacle["type"] == "rectangle":
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

        # Calculate AGV corners
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

        # Check collision for each obstacle
        for obstacle in self.obstacles:
            if obstacle["type"] == "circle":
                obs_x, obs_y = obstacle["position"]
                radius = obstacle["radius"]

                # Check if any corner is within the circle
                for corner_x, corner_y in rotated_corners:
                    if math.hypot(corner_x - obs_x, corner_y - obs_y) <= radius:
                        return True

            elif obstacle["type"] == "rectangle":
                obs_x, obs_y = obstacle["position"]
                width, height = obstacle["size"]
                half_width_obs, half_height_obs = width / 2, height / 2

                # Check if any corner is within the rectangle
                for corner_x, corner_y in rotated_corners:
                    if (
                        obs_x - half_width_obs <= corner_x <= obs_x + half_width_obs
                        and obs_y - half_height_obs <= corner_y <= obs_y + half_height_obs
                    ):
                        return True

        return False

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
            if obstacle["type"] == "circle":
                obs_x, obs_y = obstacle["position"]
                radius = obstacle["radius"]
                pygame.draw.circle(
                    self.screen,
                    self.obstacle_color,
                    (int(obs_x * self.resolution), int(obs_y * self.resolution)),
                    int(radius * self.resolution),
                )
            elif obstacle["type"] == "rectangle":
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
