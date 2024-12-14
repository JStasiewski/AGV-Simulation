import pyglet
import numpy as np
import math

class AGVEnvironment:
    def __init__(
        self, 
        width=800, 
        height=800, 
        track_width=60,
        segment_length=100,
        num_segments=8,
        angle_variance=1,  # ~30 degrees max turn each segment
        goal_reward=100.0
    ):
        self.width = width
        self.height = height
        self.track_width = track_width
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.angle_variance = angle_variance
        self.goal_reward = goal_reward

        self.state = None
        self.state_prev = None
        self.action_space = 3

        # Generate a smooth track inside the screen without large immediate turns
        self.track = self._generate_track()

        # The goal is at the end of the track
        self.goal_x, self.goal_y = self.track[-1]

        # Create pyglet window
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.batch = pyglet.graphics.Batch()

        # Car
        self.car_size = 10  # Size of the car
        self.car_color = (255, 0, 0)
        self.car = pyglet.shapes.Rectangle(
            x=self.width // 2 - self.car_size // 2,  # Center horizontally
            y=50 - self.car_size // 2,              # Center vertically at the track start
            width=self.car_size,
            height=self.car_size,
            color=self.car_color,
            batch=self.batch
        )

        
        # Sensors 
        self.sensor_angles = [-math.pi/2, -math.pi/3, -math.pi/6, 0, math.pi/6, math.pi/3, math.pi/2]
        self.sensor_range = 200
        self.sensors = []  # To store computed sensor distances
        self.sensor_circles = [] 
        self.sensor_lines = []


        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.key_handler)

        self.left_lines = []
        self.right_lines = []
        self.segment_lines = []
        self.crossed_lines = set()
        self.finish_line = None

        self.reset()

    def _generate_track(self):
        track = [(self.width//2, 50)]
        direction = 0.0  # facing "up" initially
        margin = 50

        for _ in range(self.num_segments):
            # Try several attempts to find a valid next segment direction
            success = False
            for attempt in range(50):
                # Generate a small turn increment
                # This ensures no large (>90 deg) turns:
                dir_increment = (np.random.rand() - 0.5) * self.angle_variance
                # Add the small turn to the current direction
                dir_attempt = direction + dir_increment

                new_x = track[-1][0] + math.sin(dir_attempt)*self.segment_length
                new_y = track[-1][1] + math.cos(dir_attempt)*self.segment_length

                # Check if out of bounds
                if (new_x < margin or new_x > self.width - margin or 
                    new_y < margin or new_y > self.height - margin):
                    # If out of bounds, try a straighter line: reduce turn increment
                    # Move closer to no-turn scenario
                    dir_increment /= 2.0
                    dir_attempt = direction + dir_increment
                    new_x = track[-1][0] + math.sin(dir_attempt)*self.segment_length
                    new_y = track[-1][1] + math.cos(dir_attempt)*self.segment_length
                    # Clamp within bounds
                    new_x = min(max(new_x, margin), self.width - margin)
                    new_y = min(max(new_y, margin), self.height - margin)

                # Check overlap
                if not self._intersects_previous(track, (track[-1][0], track[-1][1], new_x, new_y)):
                    track.append((new_x, new_y))
                    direction = dir_attempt
                    success = True
                    break
                else:
                    # If intersection found, try reducing turn even more
                    # Move direction closer to straight line
                    direction = direction  # Keep same direction and try smaller turn next attempt
                    self.angle_variance *= 0.9  # Reduce variance slightly to help find a solution

            # If no success after attempts, just go straight
            if not success:
                new_x = track[-1][0] + math.sin(direction)*self.segment_length
                new_y = track[-1][1] + math.cos(direction)*self.segment_length
                new_x = min(max(new_x, margin), self.width - margin)
                new_y = min(max(new_y, margin), self.height - margin)
                track.append((new_x, new_y))

        return track
    
    def _compute_sensor_distances(self):
        car_x, car_y, car_angle, _ = self.state
        distances = []
        intersections = []

        for angle_offset in self.sensor_angles:
            # Compute the sensor's angle relative to the car
            sensor_angle = car_angle + angle_offset
            
            # Calculate the sensor's end point (maximum range)
            sensor_x2 = car_x + math.sin(sensor_angle) * self.sensor_range
            sensor_y2 = car_y + math.cos(sensor_angle) * self.sensor_range

            min_distance = self.sensor_range  # Start with max range
            closest_point = None

            # Check intersection with all track boundaries
            for i in range(len(self.left_track) - 1):
                # Left boundary segment
                x1, y1 = self.left_track[i]
                x2, y2 = self.left_track[i + 1]
                if self._lines_intersect(car_x, car_y, sensor_x2, sensor_y2, x1, y1, x2, y2):
                    intersect_x, intersect_y = self._line_intersection_point(
                        car_x, car_y, sensor_x2, sensor_y2, x1, y1, x2, y2
                    )
                    distance = math.sqrt((intersect_x - car_x) ** 2 + (intersect_y - car_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = (intersect_x, intersect_y)

                # Right boundary segment
                x1, y1 = self.right_track[i]
                x2, y2 = self.right_track[i + 1]
                if self._lines_intersect(car_x, car_y, sensor_x2, sensor_y2, x1, y1, x2, y2):
                    intersect_x, intersect_y = self._line_intersection_point(
                        car_x, car_y, sensor_x2, sensor_y2, x1, y1, x2, y2
                    )
                    distance = math.sqrt((intersect_x - car_x) ** 2 + (intersect_y - car_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = (intersect_x, intersect_y)

            distances.append(min_distance)
            intersections.append(closest_point)  # Store the closest intersection point

        return distances, intersections

    def _line_intersection_point(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Calculate the intersection point of two lines
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None  # Lines are parallel or coincident

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return x, y

    def _intersects_previous(self, track, new_segment):
        x1, y1, x2, y2 = new_segment
        for i in range(len(track)-1):
            x3, y3 = track[i]
            x4, y4 = track[i+1]
            if self._lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                # Check if it's not just touching at endpoints
                if not ((x3, y3) == (x1, y1) or (x4, y4) == (x1, y1)):
                    return True
        return False

    def _lines_intersect(self, x1,y1,x2,y2,x3,y3,x4,y4):
        denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
        if denom == 0:
            return False
        ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
        ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom
        return 0 < ua < 1 and 0 < ub < 1

    def _compute_track_boundaries(self):
        left_points = []
        right_points = []
        
        for i in range(len(self.track)-1):
            x1, y1 = self.track[i]
            x2, y2 = self.track[i+1]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            
            if length == 0:  # Skip if the segment length is zero
                continue

            # Compute normalized perpendicular vector
            px = -dy / length
            py = dx / length

            # Offset points by half the track width
            lwx1 = x1 + px * (self.track_width / 2)
            lwy1 = y1 + py * (self.track_width / 2)
            lwx2 = x2 + px * (self.track_width / 2)
            lwy2 = y2 + py * (self.track_width / 2)

            rwx1 = x1 - px * (self.track_width / 2)
            rwy1 = y1 - py * (self.track_width / 2)
            rwx2 = x2 - px * (self.track_width / 2)
            rwy2 = y2 - py * (self.track_width / 2)

            # Add to boundary lists
            left_points.append((lwx1, lwy1))
            right_points.append((rwx1, rwy1))

            # Also add the end points of the segment
            if i == len(self.track) - 2:
                left_points.append((lwx2, lwy2))
                right_points.append((rwx2, rwy2))

        return left_points, right_points

    def reset(self):
        self.state = np.array([self.track[0][0], self.track[0][1], 0.0, 0.0], dtype=np.float32)
        self.car.x = self.state[0] - self.car_size // 2
        self.car.y = self.state[1] - self.car_size // 2
        self.car.rotation = self.state[2] * (180.0 / math.pi)

        self.left_track, self.right_track = self._compute_track_boundaries()

        self.sensors = self._compute_sensor_distances()

        self.left_lines.clear()
        self.right_lines.clear()
        self.segment_lines.clear()
        self.crossed_lines.clear()
        self.state_prev = self.state.copy()

        for i in range(len(self.left_track)-1):
            x1, y1 = self.left_track[i]
            x2, y2 = self.left_track[i+1]
            self.left_lines.append(
                pyglet.shapes.Line(x1, y1, x2, y2, width=2, color=(255,255,255), batch=self.batch)
            )
        for i in range(len(self.right_track)-1):
            x1, y1 = self.right_track[i]
            x2, y2 = self.right_track[i+1]
            self.right_lines.append(
                pyglet.shapes.Line(x1, y1, x2, y2, width=2, color=(255,255,255), batch=self.batch)
            )

        # Create segment lines
        for i in range(1, len(self.track)):
            x1, y1 = self.track[i-1]
            x2, y2 = self.track[i]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                px = -dy / length
                py = dx / length
                line_half_width = self.track_width // 2
                lx1 = x2 + px * line_half_width
                ly1 = y2 + py * line_half_width
                lx2 = x2 - px * line_half_width
                ly2 = y2 - py * line_half_width

                line_id = i-1
                line_shape = pyglet.shapes.Line(
                    lx1, ly1, lx2, ly2, width=2, color=(255, 255, 0), batch=self.batch
                )

                self.segment_lines.append({
                    'id': line_id,
                    'shape': line_shape,
                    'start': (lx1, ly1),
                    'end': (lx2, ly2)
                })

        return self._get_observation()

    def step(self, action):
        forward_speed = 5.0
        turn_speed = 0.1

        self.sensors = self._compute_sensor_distances()

        # Save previous state for line crossing detection
        if self.state_prev is None:
            self.state_prev = self.state.copy()
        x_prev, y_prev = self.state_prev[0], self.state_prev[1]

        x, y, angle, dist = self.state

        if action == 1:  # turn left
            angle += turn_speed
        elif action == 2:  # turn right
            angle -= turn_speed

        x += math.sin(angle) * forward_speed
        y += math.cos(angle) * forward_speed
        dist += forward_speed

        self.state = np.array([x, y, angle, dist], dtype=np.float32)

        # Initialize reward
        reward = 0

        # Check for line crossings
        for line in self.segment_lines:
            if line['id'] in self.crossed_lines:
                continue

            x1_prev, y1_prev = x_prev, y_prev
            x1_curr, y1_curr = x, y
            x2_line_start, y2_line_start = line['start']
            x2_line_end, y2_line_end = line['end']

            if self._lines_intersect(
                x1_prev, y1_prev, x1_curr, y1_curr,
                x2_line_start, y2_line_start, x2_line_end, y2_line_end
            ):
                self.crossed_lines.add(line['id'])
                reward += 10.0  # Reward for crossing the line
                # print(f"Crossed line {line['id']}")

        # Update previous state
        self.state_prev = self.state.copy()

        # Check goal condition
        goal_threshold = 10.0
        dx = x - self.goal_x
        dy = y - self.goal_y
        goal_dist = math.sqrt(dx*dx + dy*dy)

        done = False

        if goal_dist < goal_threshold:
            reward += self.goal_reward
            # print("Goal reached")
            done = True
        elif not self._within_track(x, y):
            reward = -50.0
            # print("Off track")
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}

    def _within_track(self, x, y):
        closest_dist = float('inf')
        for i in range(len(self.track)-1):
            x1, y1 = self.track[i]
            x2, y2 = self.track[i+1]
            px, py = self._closest_point_on_line_segment(x1,y1,x2,y2,x,y)
            d = math.sqrt((x - px)**2 + (y - py)**2)
            if d < closest_dist:
                closest_dist = d
        return closest_dist <= (self.track_width/2)

    def _closest_point_on_line_segment(self, x1,y1,x2,y2,px,py):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return x1,y1
        t = ((px - x1)*dx+(py - y1)*dy)/(dx*dx+dy*dy)
        t = max(0,min(1,t))
        return x1 + t*dx, y1 + t*dy

    def _get_observation(self):
        sensor_distances, _ = self._compute_sensor_distances()
        return np.concatenate([self.state.copy(), np.array(sensor_distances, dtype=np.float32)])

    def render(self):
        self.window.clear()
        self.car.x = self.state[0] - self.car_size // 2
        self.car.y = self.state[1] - self.car_size // 2
        self.car.rotation = self.state[2] * (180.0 / math.pi)
        self.batch.draw()

        # Clear old sensor lines and circles
        self.sensor_lines.clear()
        self.sensor_circles.clear()

        # Draw sensor lines and their intersections
        car_x, car_y, car_angle, _ = self.state
        distances, sensor_intersections = self._compute_sensor_distances()

        for angle_offset, (distance, intersection) in zip(self.sensor_angles, zip(distances, sensor_intersections)):
            # Compute sensor line end point
            sensor_angle = car_angle + angle_offset
            sensor_x2 = car_x + math.sin(sensor_angle) * distance
            sensor_y2 = car_y + math.cos(sensor_angle) * distance

            # Create the line for the sensor
            line = pyglet.shapes.Line(
                car_x, car_y, sensor_x2, sensor_y2, width=1, color=(0, 255, 0), batch=self.batch
            )
            self.sensor_lines.append(line)  # Store the line

            # Draw a circle at the intersection point (if detected)
            if intersection is not None:
                intersect_x, intersect_y = intersection
                circle = pyglet.shapes.Circle(
                    intersect_x, intersect_y, radius=5, color=(255, 0, 0), batch=self.batch
                )
                self.sensor_circles.append(circle)


    def close(self):
        self.window.close()

    def sample_action(self):
        return np.random.randint(0, self.action_space)

    def on_key_press(self, symbol, modifiers):
        pass
