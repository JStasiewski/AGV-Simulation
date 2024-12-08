from environment import AGVEnvironment
import pygame
import time


def human_control():
    env = AGVEnvironment()
    obs = env.reset()

    print("Use arrow keys and modifiers to control the AGV:")
    print("  ↑: Move forward by 1 cm")
    print("  SHIFT + ↑: Move forward by 5 cm")
    print("  CTRL + ↑: Move forward by 10 cm")
    print("  ALT + ↑: Move forward by 25 cm")
    print("  ←: Turn left by 1°")
    print("  SHIFT + ←: Turn left by 5°")
    print("  CTRL + ←: Turn left by 10°")
    print("  ALT + ←: Turn left by 25°")
    print("  →: Turn right by 1°")
    print("  SHIFT + →: Turn right by 5°")
    print("  CTRL + →: Turn right by 10°")
    print("  ALT + →: Turn right by 25°")
    print("Press 'Esc' to exit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input
        keys = pygame.key.get_pressed()
        action = None

        # Define modifiers
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        alt_pressed = keys[pygame.K_LALT] or keys[pygame.K_RALT]

        # Determine action
        if keys[pygame.K_UP]:  # Forward movements
            if alt_pressed:
                action = 4  # Move forward by 25 cm
            elif ctrl_pressed:
                action = 3  # Move forward by 10 cm
            elif shift_pressed:
                action = 2  # Move forward by 5 cm
            else:
                action = 1  # Move forward by 1 cm
        elif keys[pygame.K_LEFT]:  # Left turns
            if alt_pressed:
                action = len(env.move_steps) + 4  # Turn left by 25°
            elif ctrl_pressed:
                action = len(env.move_steps) + 3  # Turn left by 10°
            elif shift_pressed:
                action = len(env.move_steps) + 2  # Turn left by 5°
            else:
                action = len(env.move_steps) + 1  # Turn left by 1°
        elif keys[pygame.K_RIGHT]:  # Right turns
            if alt_pressed:
                action = len(env.move_steps) + len(env.turn_angles) + 4  # Turn right by 25°
            elif ctrl_pressed:
                action = len(env.move_steps) + len(env.turn_angles) + 3  # Turn right by 10°
            elif shift_pressed:
                action = len(env.move_steps) + len(env.turn_angles) + 2  # Turn right by 5°
            else:
                action = len(env.move_steps) + len(env.turn_angles) + 1  # Turn right by 1°
        elif keys[pygame.K_ESCAPE]:
            running = False

        # Execute action if defined
        if action is not None:
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Position: {obs[:2]}, Angle: {obs[2]}°, Reward: {reward}")

            if done:
                if reward > 0:
                    print("Goal reached!")
                else:
                    print("Collision detected!")
                running = False
        time.sleep(0.01)

    env.close()
    print("Simulation ended.")


if __name__ == "__main__":
    human_control()
