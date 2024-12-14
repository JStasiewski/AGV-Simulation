import gym
import numpy as np
import pygame
import math
import random

class AGVLineFollowingEnvironment(gym.Env):
    def __init__(self):
        super(AGVLineFollowingEnvironment, self).__init__()

        # Field dimensions (15x15 meters)
        self.field_size = 15
        self.resolution = 50
        self.window_size = int(self.field_size * self.resolution)

        # AGV parameters
        self.agv_length = 0.7
        self.agv_width = 0.5

        # Define actions: turning and forward movement
        self.turn_angles = [5, 10, 15, 30, 45]  # degrees per action
        self.move_steps = [0.01, 0.05, 0.1, 0.25, 1]  # meters per step
        self.num_turn_actions = len(self.turn_angles) * 2  # Left and right
        self.num_move_actions = len(self.move_steps)       # Forward
        self.action_space = gym.spaces.Discrete(self.num_turn_actions + self.num_move_actions)

        # Observation: [x, y, heading, lateral_error, heading_error, distance_to_goal]
        # We'll keep the original observation structure, but note we may internally consider more info
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, -self.field_size, -180, 0]),
            high=np.array([self.field_size, self.field_size, 360, self.field_size, 180, self.field_size]),
            dtype=np.float32,
        )

        # Start and goal positions (will be set in reset)
        self.start_pos = np.array([0.5, 7.5, 0])
        self.goal_pos = None

        # Path points defining the line
        self.path_points = None

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("AGV Line Following Environment")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (230, 230, 230)
        self.grid_color = (200, 200, 200)
        self.agv_color = (0, 0, 255)
        self.goal_color = (0, 255, 0)
        self.path_color = (0, 0, 0)

        # Track how long the AGV stays in the same place
        self.stuck_steps = 0
        self.prev_pos = None

        # Parameters for rewarding corrective turns
        self.turn_reward_threshold = 1.0  # Lateral error threshold for rewarding turns
        self.turn_reward_scale = 0.5  # Scaling factor for rewarding turns

    def reset(self):
        # Generate a random path and goal
        self.generate_random_path()

        # Place AGV at start of path
        self.agv_pos = self.start_pos.copy()

        # Reset stuck steps counter and previous position
        self.stuck_steps = 0
        self.prev_pos = self.agv_pos[:2].copy()
        if hasattr(self, 'prev_distance_to_goal'):
            del self.prev_distance_to_goal

        observation = self.get_observation()
        return observation

    def step(self, action):
        # Apply action
        is_turning = action < self.num_turn_actions
        if is_turning:  # Turn in place
            direction = -1 if action % 2 == 0 else 1
            angle_idx = action // 2
            self.agv_pos[2] = (self.agv_pos[2] + direction * self.turn_angles[angle_idx]) % 360
        else:
            # Move forward
            move_idx = action - self.num_turn_actions
            chosen_step = self.move_steps[move_idx]
            dx = chosen_step * math.cos(math.radians(self.agv_pos[2]))
            dy = chosen_step * math.sin(math.radians(self.agv_pos[2]))
            self.agv_pos[0] += dx
            self.agv_pos[1] += dy

        # Check collision with wall (field boundaries)
        if (self.agv_pos[0] < 0 or self.agv_pos[0] > self.field_size or
            self.agv_pos[1] < 0 or self.agv_pos[1] > self.field_size):
            # Collision detected
            obs = self.get_observation()
            reward = -100.0  # Penalty for collision
            done = True
            return obs, reward, done, {}

        # Compute new observation
        obs = self.get_observation()

        # Check if reached goal
        dist_to_goal = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)
        done = dist_to_goal < 0.5

        # Reward system
        # Extract elements from observation
        agv_x, agv_y, agv_angle, lateral_error, heading_error, distance_to_goal = obs

        # Determine desired heading, blending next segment direction if near a junction
        closest_point, current_segment_heading, closest_dist, segment_idx, seg_progress = self.get_line_closest_point(self.agv_pos[:2], return_segment_info=True)
        desired_heading = current_segment_heading

        # If we are close to the end of the current segment and there's a next segment,
        # blend the desired heading towards the next segment's direction
        if segment_idx < len(self.path_points) - 2:
            # Next segment heading
            next_segment_heading = self.get_segment_heading(segment_idx + 1)
            # How far along current segment are we? seg_progress in [0,1]
            # If seg_progress > 0.8 (80% along the segment), start blending
            blend_start = 0.8
            if seg_progress > blend_start:
                blend_factor = (seg_progress - blend_start) / (1.0 - blend_start)
                # Smooth blend current heading to next segment heading
                desired_heading = self.angle_blend(current_segment_heading, next_segment_heading, blend_factor)

        # Compute a heading error based on the desired heading rather than just line heading
        adjusted_heading_error = (agv_angle - desired_heading + 180) % 360 - 180

        # Distance improvement
        old_dist = self.prev_distance_to_goal if hasattr(self, 'prev_distance_to_goal') else distance_to_goal + 1.0
        distance_improvement = old_dist - distance_to_goal
        self.prev_distance_to_goal = distance_to_goal

        reward = 0.0

        # Encourage moving closer to the goal
        reward += 1.0 * distance_improvement

        # Penalize lateral error with an exponential factor
        reward -= (1.1 ** abs(lateral_error)) * 0.5

        # Penalize the adjusted heading error
        reward -= abs(adjusted_heading_error) * 0.05

        # Reward for turning towards the path if lateral error is large
        if abs(lateral_error) > self.turn_reward_threshold and is_turning:
            # Check if the chosen turn reduces adjusted heading error
            predicted_heading = (self.agv_pos[2]) % 360
            predicted_heading_error = (predicted_heading - desired_heading + 180) % 360 - 180
            if abs(predicted_heading_error) < abs(adjusted_heading_error):
                reward += abs(lateral_error) * self.turn_reward_scale

        # Reward finishing
        if done:
            reward += 100.0

        # Check movement to apply penalty for staying in the same place
        move_distance = np.linalg.norm(self.agv_pos[:2] - self.prev_pos)
        if move_distance < 0.001:  # Not moved significantly
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        self.prev_pos = self.agv_pos[:2].copy()

        # Small penalty if stuck too long (e.g., more than 5 steps)
        if self.stuck_steps > 5:
            reward -= 0.1 * (self.stuck_steps - 5)

        return obs, reward, done, {}

    def generate_random_path(self):
        # Start fixed
        start = self.start_pos[:2]
        # Random goal point far from start
        goal = np.array([random.uniform(10, 14), random.uniform(10, 14)])
        self.goal_pos = goal

        # Generate a few intermediate waypoints
        mid_points = []
        for _ in range(random.randint(3, 6)):
            px = random.uniform(1, self.field_size - 1)
            py = random.uniform(1, self.field_size - 1)
            mid_points.append(np.array([px, py]))

        # Insert start and goal
        mid_points.append(start)
        mid_points.append(goal)

        # Sort by projection onto start->goal line
        start_to_goal = goal - start
        def projection_val(p):
            v = p - start
            return np.dot(v, start_to_goal) / np.linalg.norm(start_to_goal)

        mid_points.sort(key=projection_val)
        self.path_points = mid_points

    def get_line_closest_point(self, pos, return_segment_info=False):
        x, y = pos
        closest_dist = float('inf')
        closest_point = None
        closest_heading = 0.0
        closest_segment_idx = 0
        segment_progress = 0.0

        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            seg_dir = seg_vec / seg_len

            v = pos - p1
            proj = np.dot(v, seg_dir)
            if proj < 0:
                proj_point = p1
                proj_dist = np.linalg.norm(pos - p1)
                seg_prog = 0.0
            elif proj > seg_len:
                proj_point = p2
                proj_dist = np.linalg.norm(pos - p2)
                seg_prog = 1.0
            else:
                proj_point = p1 + seg_dir * proj
                proj_dist = np.linalg.norm(pos - proj_point)
                seg_prog = proj / seg_len

            if proj_dist < closest_dist:
                closest_dist = proj_dist
                closest_point = proj_point
                closest_heading = math.degrees(math.atan2(seg_dir[1], seg_dir[0]))
                closest_segment_idx = i
                segment_progress = seg_prog

        if return_segment_info:
            return closest_point, closest_heading, closest_dist, closest_segment_idx, segment_progress
        else:
            return closest_point, closest_heading, closest_dist

    def get_segment_heading(self, segment_idx):
        # Returns the heading of the segment indexed by segment_idx
        # segment_idx refers to the segment between path_points[segment_idx] and path_points[segment_idx+1]
        if segment_idx < 0 or segment_idx >= len(self.path_points)-1:
            return 0.0  # default if out of range
        p1 = self.path_points[segment_idx]
        p2 = self.path_points[segment_idx+1]
        seg_vec = p2 - p1
        return math.degrees(math.atan2(seg_vec[1], seg_vec[0]))

    def angle_blend(self, angle1, angle2, factor):
        # Blend two angles (in degrees) smoothly
        # Convert to radians
        a1 = math.radians(angle1)
        a2 = math.radians(angle2)
        # Find difference
        diff = ((a2 - a1 + math.pi) % (2*math.pi)) - math.pi
        a_blend = a1 + diff * factor
        return (math.degrees(a_blend) + 360) % 360

    def get_observation(self):
        agv_x, agv_y, agv_angle = self.agv_pos
        closest_point, line_heading, lateral_error = self.get_line_closest_point(self.agv_pos[:2])

        # Compute heading error
        heading_error = (agv_angle - line_heading + 180) % 360 - 180
        distance_to_goal = np.linalg.norm(self.agv_pos[:2] - self.goal_pos)

        obs = np.array([
            agv_x,
            agv_y,
            agv_angle,
            lateral_error,
            heading_error,
            distance_to_goal
        ], dtype=np.float32)

        return obs

    def render(self, mode="human"):
        self.screen.fill(self.bg_color)

        # Draw grid
        for x in range(0, self.window_size, self.resolution):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.window_size))
            pygame.draw.line(self.screen, self.grid_color, (0, x), (self.window_size, x))

        # Draw path
        if self.path_points is not None:
            scaled_points = [(int(px * self.resolution), int(py * self.resolution)) for px, py in self.path_points]
            pygame.draw.lines(self.screen, self.path_color, False, scaled_points, 3)

        # Draw goal
        if self.goal_pos is not None:
            gx = int(self.goal_pos[0] * self.resolution)
            gy = int(self.goal_pos[1] * self.resolution)
            pygame.draw.circle(self.screen, self.goal_color, (gx, gy), 5)

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
        pygame.quit()
