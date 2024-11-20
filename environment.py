import gym
import numpy as np
import pygame
import math


class AGVEnvironment(gym.Env):
    def __init__(self):
        super(AGVEnvironment, self).__init__()

        # Field dimensions (15x15 meters)
        self.field_size = 15  # The field is 15x15 meters
        self.resolution = 50  # 50 pixels per meter for rendering
        self.window_size = self.field_size * self.resolution  # Size of the Pygame window

        # AGV parameters
        self.agv_length = 0.7  # Length of the AGV in meters
        self.agv_width = 0.5  # Width of the AGV in meters
        self.move_step = 0.002  # Forward movement per step in meters (1 cm)
        self.turn_step = 0.2  # Rotation per step in degrees

        # Obstacles (positions in meters)
        self.obstacles = [
            (5, 5),  # Obstacle at (5, 5)
            (10, 10),  # Obstacle at (10, 10)
            (7, 8),  # Obstacle at (7, 8)
        ]

        # Action space: 0 = move forward, 1 = turn left, 2 = turn right
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: AGV's position (x, y, angle)
        # x, y range from 0 to field_size, and angle from 0 to 360 degrees
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.field_size, self.field_size, 360]),
            dtype=np.float32,
        )

        # Initial position and goal
        self.start_pos = np.array([0.5, 7.5, 0])  # AGV starts at (7.5, 7.5), facing 0 degrees
        self.goal_pos = np.array([14.0, 14.0])  # Goal is located at (14, 14)
        self.agv_pos = self.start_pos.copy()  # AGV's current position and orientation

        # Pygame setup for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("AGV Environment")
        self.clock = pygame.time.Clock()

        # Colors for visualization
        self.bg_color = (230, 230, 230)  # Background color: light grey
        self.grid_color = (200, 200, 200)  # Grid color: grey
        self.agv_color = (0, 0, 255)  # AGV color: blue
        self.goal_color = (0, 255, 0)  # Goal color: green

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.agv_pos = self.start_pos.copy()
        return self.agv_pos

    def step(self, action):
        """
        Execute an action and update the environment.

        Args:
            action (int): 0 = move forward, 1 = turn left, 2 = turn right

        Returns:
            tuple: (observation, reward, done, info)
        """
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
            reward = -10  # Penalize collision
            done = True
            return self.agv_pos, reward, done, {}

        # Check if the AGV reached the goal
        distance_to_goal = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)
        done = distance_to_goal < 0.1  # Goal reached if within 10 cm
        reward = 10 if done else -0.01  # Reward for reaching the goal, small penalty otherwise

        return self.agv_pos, reward, done, {}

    def check_collision(self):
        """
        Check if the AGV collides with any obstacle.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        agv_x = self.agv_pos[0]
        agv_y = self.agv_pos[1]
        agv_angle = self.agv_pos[2]

        # Calculate AGV's corners based on its position and orientation
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

        # Check collision with obstacles
        for obs_x, obs_y in self.obstacles:
            for corner_x, corner_y in rotated_corners:
                # Check if any corner is within the obstacle's radius
                if math.dist((corner_x, corner_y), (obs_x, obs_y)) < 0.2:  # Adjust collision radius
                    return True

            # Check if any part of the AGV rectangle intersects the obstacle's bounding circle
            agv_center = np.array([agv_x, agv_y])
            obstacle_center = np.array([obs_x, obs_y])
            distance_to_obstacle = np.linalg.norm(agv_center - obstacle_center)

            # Effective AGV radius approximation for collision detection
            effective_agv_radius = math.sqrt((half_length ** 2) + (half_width ** 2))

            if distance_to_obstacle <= effective_agv_radius + 0.2:  # 20 cm obstacle radius
                return True

        return False


    def render(self, mode="human"):
        """
        Render the environment using Pygame.
        """
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
        for obs_x, obs_y in self.obstacles:
            obs_rect = pygame.Rect(
                int(obs_x * self.resolution) - 10,
                int(obs_y * self.resolution) - 10,
                20,
                20,
            )
            pygame.draw.rect(self.screen, (255, 0, 0), obs_rect)  # Red for obstacles

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
        pygame.display.flip()

    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()
