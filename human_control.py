from environment import AGVEnvironment
import pygame


def human_control():
    env = AGVEnvironment()
    obs = env.reset()

    print("Use arrow keys to control the AGV:")
    print("  ↑: Move forward")
    print("  ←: Turn left")
    print("  →: Turn right")
    print("Press 'Esc' to exit.")

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
        elif keys[pygame.K_ESCAPE]:
            running = False

        # Execute action if defined
        if action is not None:
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Position: {obs[:2]}, Angle: {obs[2]}°")

            if done:
                print("Goal reached!")
                running = False

    env.close()
    print("Simulation ended.")


if __name__ == "__main__":
    human_control()
