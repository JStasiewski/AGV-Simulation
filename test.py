import gym
from gym import spaces
import numpy as np
import pygame
import math


class AGVEnvironment(gym.Env):
    def __init__(self):
        super(AGVEnvironment, self).__init__()

        # Field dimensions (15x15 meters)
        self.field_size = 15  # meters
        self.resolution = 50  # pixels per meter (1 meter = 100 pixels)
        self.window_size = self.field_size * self.resolution

        # AGV parameters
        self.agv_length = 0.5  # meters (length of AGV)
        self.agv_width = 0.3  # meters (width of AGV)
        self.move_step = 0.05  # meters (1 cm)
        self.turn_step = 5  # degrees per turn

        # Action space: 0=move forward, 1=turn left, 2=turn right
        self.action_space = spaces.Discrete(3)

        # Observation space: AGV's position (x, y, angle)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.field_size, self.field_size, 360]),
            dtype=np.float32,
        )

        # Initial position and goal
        self.start_pos = np.array([7.5, 7.5, 0])  # x, y in meters, angle in degrees
        self.goal_pos = np.array([14.0, 14.0])  # x, y in meters
        self.agv_pos = self.start_pos.copy()

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("AGV Environment")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (230, 230, 230)  # Light grey
        self.grid_color = (200, 200, 200)  # Grey
        self.agv_color = (0, 0, 255)  # Blue
        self.goal_color = (0, 255, 0)  # Green

    def reset(self):
        """Resets the environment."""
        self.agv_pos = self.start_pos.copy()
        return self.agv_pos

    def step(self, action):
        """Executes an action and updates the environment."""
        # Update position and orientation based on action
        if action == 0:  # Move forward
            dx = self.move_step * math.cos(math.radians(self.agv_pos[2]))
            dy = self.move_step * math.sin(math.radians(self.agv_pos[2]))
            self.agv_pos[0] += dx
            self.agv_pos[1] += dy
        elif action == 1:  # Turn left
            self.agv_pos[2] = (self.agv_pos[2] + self.turn_step) % 360
        elif action == 2:  # Turn right
            self.agv_pos[2] = (self.agv_pos[2] - self.turn_step) % 360

        # Check if AGV is within bounds
        self.agv_pos[0] = np.clip(self.agv_pos[0], 0, self.field_size)
        self.agv_pos[1] = np.clip(self.agv_pos[1], 0, self.field_size)

        # Calculate reward
        distance_to_goal = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)
        done = distance_to_goal < 0.1  # Goal is reached if within 10 cm
        reward = 10 if done else -0.01  # Reward for reaching the goal

        return self.agv_pos, reward, done, {}

    def render(self, mode="human"):
        """Render the environment using Pygame."""
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

        # Draw AGV
        agv_x = int(self.agv_pos[0] * self.resolution)
        agv_y = int(self.agv_pos[1] * self.resolution)
        agv_angle = self.agv_pos[2]

        # Calculate AGV corners
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
        """Closes the Pygame window."""
        pygame.quit()


# Run the environment with keyboard control
if __name__ == "__main__":
    env = AGVEnvironment()
    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action = 0  # Move forward
        elif keys[pygame.K_LEFT]:
            action = 1  # Turn left
        elif keys[pygame.K_RIGHT]:
            action = 2  # Turn right

        # Execute action
        if action is not None:
            obs, reward, done, _ = env.step(action)
            print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
            if done:
                print("Goal reached!")
                running = False

        env.render()
        env.clock.tick(30)  # Limit to 30 frames per second

    env.close()
