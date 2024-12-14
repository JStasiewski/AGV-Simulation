from env import AGVLineFollowingEnvironment
import pygame
import time

def human_control():
    env = AGVLineFollowingEnvironment()
    obs = env.reset()

    print("Use arrow keys and modifiers to control the AGV:")
    print("Forward actions:")
    print("  ↑: Move forward by 0.01 m")
    print("  SHIFT + ↑: Move forward by 0.05 m")
    print("  CTRL + ↑: Move forward by 0.1 m")
    print("  ALT + ↑: Move forward by 0.25 m")
    print("  (Press multiple times for incremental moves.)")
    print("Turn actions:")
    print("  ←: Turn left by smallest angle (5°)")
    print("  SHIFT + ←: Left by next angle (10°)")
    print("  CTRL + ←: Left by next angle (15°)")
    print("  ALT + ←: Left by next angle (30°) - continue pattern as needed")
    print("  Similarly for → (right turns).")
    print("Press 'Esc' to exit.")

    # For convenience, define a mapping from key/modifiers to action indices
    # Forward moves: Actions are [0,...,num_turn_actions+num_move_actions-1]
    # In AGVLineFollowingEnvironment:
    #    action_space = Discrete(num_turn_actions + num_move_actions)
    #    num_turn_actions = len(turn_angles)*2
    #    num_move_actions = len(move_steps)
    #
    # move_steps = [0.01, 0.05, 0.1, 0.25, 1]
    # turn_angles = [5, 10, 15, 30, 45]
    #
    # Actions (assuming same indexing as before):
    # Move actions: 0 to (num_move_actions-1)
    # Turn actions: move_actions_end to move_actions_end+(num_turn_actions)-1
    #
    # Let's say:
    # Move indexes:
    #   0 -> 0.01 m
    #   1 -> 0.05 m
    #   2 -> 0.1 m
    #   3 -> 0.25 m
    #   4 -> 1.0 m
    #
    # Turn indexes:
    # left turns: 
    #   num_move_actions + 0,2,4,6,8 for left turns (corresponding to angles 5,10,15,30,45)
    # right turns:
    #   num_move_actions + 1,3,5,7,9 for right turns
    #
    # We'll just pick consistent keys like in the original code.

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = None

        # Modifiers
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        alt_pressed = keys[pygame.K_LALT] or keys[pygame.K_RALT]

        # Determine forward movement action
        # Index in move_steps: 0->0.01, 1->0.05, 2->0.1, 3->0.25, 4->1
        if keys[pygame.K_UP]:
            if alt_pressed:
                move_index = 3  # 0.25 m
            elif ctrl_pressed:
                move_index = 2  # 0.1 m
            elif shift_pressed:
                move_index = 1  # 0.05 m
            else:
                move_index = 0  # 0.01 m
            action = move_index  # Move actions start at 0

        # Determine turning action
        # turn_angles = [5, 10, 15, 30, 45]
        # Left turn actions start at num_move_actions (which is 5)
        # Pattern for left turns (even indices): num_move_actions + 0*2 = 5 for 5°
        # Actually, we must follow the environment logic. For that environment:
        #
        # num_turn_actions = len(turn_angles)*2 = 10
        # so turn actions from 5 to 14 (if 5 move steps)
        # indexing:
        # left turns: 5->5°, 7->10°, 9->15°, 11->30°, 13->45°
        # right turns: 6->5°, 8->10°, 10->15°, 12->30°, 14->45°
        #
        # We'll choose the same pattern as original:
        #
        # Actually, let's simplify the logic:
        # We'll pick the angle index based on modifiers:
        #   no modifier: 5°
        #   shift: 10°
        #   ctrl: 15°
        #   alt: 30° (we won't do 45° for simplicity, but you can extend)
        #
        # Adjusting to the pattern:
        # We have 5 angles: 5°,10°,15°,30°,45° at indexes [0,1,2,3,4]
        #
        # For left turn:
        #  base index = num_move_actions (5)
        #  angle_idx: 
        #     no modifier: 5° -> angle_idx = 0 -> left turn action = 5 + (0*2) = 5 
        #     shift: 10° -> angle_idx = 1 -> left turn action = 5 + (1*2) = 7
        #     ctrl: 15° -> angle_idx = 2 -> left turn action = 5 + (2*2) = 9
        #     alt: 30° -> angle_idx = 3 -> left turn action = 5 + (3*2) = 11
        # For right turn, just add 1 more:
        #     no modifier: angle_idx = 0 -> right turn = 6
        #     shift: angle_idx = 1 -> right turn = 8
        #     ctrl: angle_idx = 2 -> right turn = 10
        #     alt: angle_idx = 3 -> right turn = 12
        #
        # If we wanted 45°, continue pattern:
        #   alt+ctrl or something else for 45°. For simplicity, just use alt for 30°, ctrl+alt for 45° if needed.
        # Let's just map alt to 30° and if ctrl+alt is pressed simultaneously, do 45°.
        
        if keys[pygame.K_LEFT] and action is None:
            angle_idx = 0
            if shift_pressed:
                angle_idx = 1
            if ctrl_pressed:
                angle_idx = 2
            if alt_pressed:
                angle_idx = 3
            # Check for ctrl+alt for 45° if desired:
            if ctrl_pressed and alt_pressed:
                angle_idx = 4  # 45°
            
            # left turn action
            # left_base = num_move_actions + 0-based steps for left = 5 for 5°
            left_action = 5 + (angle_idx * 2)
            action = left_action

        if keys[pygame.K_RIGHT] and action is None:
            angle_idx = 0
            if shift_pressed:
                angle_idx = 1
            if ctrl_pressed:
                angle_idx = 2
            if alt_pressed:
                angle_idx = 3
            # ctrl+alt for 45°:
            if ctrl_pressed and alt_pressed:
                angle_idx = 4
            
            # right turn action is left turn action + 1
            right_action = 6 + (angle_idx * 2)
            action = right_action

        if keys[pygame.K_ESCAPE]:
            running = False

        # Execute action if defined
        if action is not None:
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Position: {obs[0]:.2f}, {obs[1]:.2f}; Angle: {obs[2]:.2f}°; Reward: {reward:.2f}")
            if done:
                if reward > 0:
                    print("Goal reached!")
                else:
                    print("Episode ended.")
                running = False

        time.sleep(0.05)

    env.close()
    print("Simulation ended.")

if __name__ == "__main__":
    human_control()
