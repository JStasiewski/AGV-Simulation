import pyglet
from pyglet.window import key
from environment import AGVEnvironment

# Human can control the car using arrow keys:
# LEFT arrow = turn left
# RIGHT arrow = turn right
# UP arrow = forward

def main():
    env = AGVEnvironment()
    action_map = {
        (False, False, True): 0,  # forward only
        (True, False, True): 1,   # turn left + forward
        (False, True, True): 2,   # turn right + forward
        (True, False, False): 1,  # turn left only
        (False, True, False): 2,  # turn right only
        (False, False, False): 0  # no input default forward?
    }

    # Create a label to display the reward
    reward_label = pyglet.text.Label(
        text="Reward: 0",
        font_name="Arial",
        font_size=14,
        x=10, y=env.height - 20,  # Top-left corner
        anchor_x="left", anchor_y="top",
        color=(255, 255, 255, 255)  # White color
    )

    def update(dt):
        left = env.key_handler[key.RIGHT]
        right = env.key_handler[key.LEFT]
        up = env.key_handler[key.UP]

        action = action_map.get((left, right, up), 0)
        obs, reward, done, info = env.step(action)
        
        # Print the reward to the console
        print(f"Reward: {reward}")
        
        # Update the label text
        reward_label.text = f"Reward: {reward}"
        
        if done:
            env.reset()
        env.render()
        reward_label.draw()  # Draw the label on top of the environment

    pyglet.clock.schedule_interval(update, 1/30.0)  # 30 FPS
    pyglet.app.run()

if __name__ == "__main__":
    main()
